import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Must be set before the app settings and DeepEval are imported
os.environ.setdefault("APP_ENVIRONMENT", "eval")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import click  # noqa: E402
from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import (  # noqa: E402
    AsyncConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import AnthropicModel  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from langfuse import observe  # noqa: E402

from src.config import settings  # noqa: E402
from src.rag.schemas import LLMJobMatches  # noqa: E402
from src.rag.service import RAGService  # noqa: E402
from src.tracing import init_tracing  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)
logging.getLogger("deepeval").setLevel(logging.WARNING)

EVAL_DIR = Path(__file__).parent
QUERIES_FILE = EVAL_DIR / "queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
JUDGE_MODEL = "claude-sonnet-5-5"


def load_queries(limit: int | None) -> list[dict]:
    with QUERIES_FILE.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    return queries[:limit] if limit else queries


@observe(name="eval-case", capture_output=False)
def run_case(service: RAGService, query: str) -> dict:
    """Run the pipeline steps and keep what the metrics need."""
    filters = service.query_parser.parse(query)
    docs, filters_relaxed = service.retrieve(query, filters)
    docs = service.rerank(query, docs)
    matches = service.find_matches(query, docs) if docs else LLMJobMatches(matches=[])
    results = service.build_results(matches, docs)
    return {
        "filters": filters.model_dump(exclude_none=True),
        "filters_relaxed": filters_relaxed,
        "context": [doc.page_content for doc in docs],
        "results": results,
    }


def format_answer(results) -> str:
    """Format the API answer as plain text for the judge."""
    if not results:
        return "No matching jobs found."
    return "\n\n".join(
        f"{i}. {job.title} at {job.company}\n"
        f"Summary: {job.summary}\n"
        f"Why it matches: {job.match_reason}"
        for i, job in enumerate(results, start=1)
    )


def build_metrics() -> list:
    judge = AnthropicModel(model=JUDGE_MODEL, api_key=settings.anthropic_api_key)
    return [
        ContextualRelevancyMetric(model=judge),  # does retrieved jobs fit the query?
        FaithfulnessMetric(model=judge),  # does answer sticks to the job text?
        AnswerRelevancyMetric(model=judge),  # does answer addresses the query?
    ]


@click.command()
@click.option(
    "--limit",
    default=5,
    show_default=True,
    help="Run the first N queries (0 for all).",
)
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    help="Queries judged at the same time (keep low to avoid rate limits).",
)
def main(limit: int, concurrency: int):
    """Evaluate the job search pipeline with DeepEval, using LLM as the judge."""
    langfuse = init_tracing()
    service = RAGService()
    queries = load_queries(limit)

    # Run the pipeline for every query
    rows, test_cases = [], []
    for i, item in enumerate(queries, start=1):
        logger.info("[%d/%d] %s", i, len(queries), item["query"])
        out = run_case(service, item["query"])
        row = {
            **item,
            "filters": out["filters"],
            "filter_ok": out["filters"] == item["expected_filters"],
            "filters_relaxed": out["filters_relaxed"],
            "num_results": len(out["results"]),
            "answer": format_answer(out["results"]),
            "metrics": {},
        }
        rows.append(row)

        # Off-topic queries are checked by result count, not by the judge
        if item["expect_relevant"]:
            test_cases.append(
                LLMTestCase(
                    name=item["id"],
                    input=item["query"],
                    actual_output=row["answer"],
                    retrieval_context=out["context"],
                )
            )
    langfuse.flush()

    # Judge the answers with DeepEval
    judge_cost = 0.0
    if test_cases:
        evaluation = evaluate(
            test_cases=test_cases,
            metrics=build_metrics(),
            async_config=AsyncConfig(max_concurrent=concurrency),
            display_config=DisplayConfig(print_results=False, inspect_after_run=False),
            # failed judge call marks that metric as errored instead of stopping the run
            error_config=ErrorConfig(ignore_errors=True),
        )
        by_id = {row["id"]: row for row in rows}
        for result in evaluation.test_results:
            for metric in result.metrics_data or []:
                by_id[result.name]["metrics"][metric.name] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "error": metric.error,
                }
                judge_cost += metric.evaluation_cost or 0

    summary = summarize(rows, judge_cost)
    print_summary(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, default=str)
    )
    print(f"\nSaved per-query scores and reasons to {out_file}")
    langfuse.shutdown()


def average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict], judge_cost: float) -> dict:
    relevant = [r for r in rows if r["expect_relevant"]]
    negative = [r for r in rows if not r["expect_relevant"]]

    # Average score per metric, overall and per tag
    scores, scores_by_tag = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in relevant:
        for name, data in row["metrics"].items():
            if data["score"] is not None:
                scores[name].append(data["score"])
                for tag in row["tags"]:
                    scores_by_tag[tag][name].append(data["score"])

    return {
        "queries": len(rows),
        "judge_model": JUDGE_MODEL,
        "llm_model": settings.llm_model,
        "filter_accuracy": average([float(r["filter_ok"]) for r in rows]),
        "filters_relaxed": sum(r["filters_relaxed"] for r in rows),
        "negative_queries_with_no_results": (
            f"{sum(r['num_results'] == 0 for r in negative)}/{len(negative)}"
        ),
        "metrics": {name: average(values) for name, values in scores.items()},
        "metrics_by_tag": {
            tag: {name: average(values) for name, values in metrics.items()}
            for tag, metrics in scores_by_tag.items()
        },
        "judge_cost_usd": round(judge_cost, 4),
    }


def print_summary(summary: dict) -> None:
    print("\n=== Eval summary ===")
    print(f"Queries:            {summary['queries']}")
    print(f"Filter accuracy:    {summary['filter_accuracy']}")
    print(f"Filters relaxed:    {summary['filters_relaxed']}")
    print(f"Off-topic -> empty: {summary['negative_queries_with_no_results']}")
    for name, score in summary["metrics"].items():
        print(f"{name + ':':<22}{score}")
    if summary["metrics_by_tag"]:
        print("\nBy tag:")
        for tag, metrics in summary["metrics_by_tag"].items():
            line = ", ".join(f"{name} {score}" for name, score in metrics.items())
            print(f"  {tag:<10} {line}")
    print(f"\nJudge cost:         ${summary['judge_cost_usd']}")


if __name__ == "__main__":
    main()
