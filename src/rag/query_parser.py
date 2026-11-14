import logging

import anthropic

from src.config import settings
from src.rag.schemas import QueryFilters

logger = logging.getLogger(__name__)

FILTER_PROMPT = """You extract search filters from a job search query. Set a field only when \
the query clearly asks for it; otherwise leave it null.

- work_type: remote, hybrid or onsite ("in office" means onsite).
- country: the country the jobs should be in. Map cities, states and abbreviations to their \
country (e.g. "NYC" or "USA" -> United States, "London" -> United Kingdom). Leave it null for \
a country that is not in the allowed values.
- seniority: the experience level asked for. "Entry level", "graduate" or "new grad" mean \
junior; "internship" means intern; "mid-level" means mid; "lead", "staff" or "principal" \
mean senior."""


class QueryParser:
    """Extract structured filters (work type, country, seniority) from a search query."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def parse(self, query: str) -> QueryFilters:
        """Return the filters in the query. Any failure means no filters, so search still works."""
        try:
            response = self.client.messages.parse(
                model=settings.filter_model,
                max_tokens=settings.filter_max_tokens,
                system=FILTER_PROMPT,
                messages=[{"role": "user", "content": query}],
                output_format=QueryFilters,
                output_config={"effort": settings.filter_effort},
            )
        except anthropic.APIError as e:
            logger.warning("Filter extraction failed, searching without filters: %s", e)
            return QueryFilters()

        if response.stop_reason != "end_turn" or response.parsed_output is None:
            logger.warning(
                "Filter extraction stopped (%s), searching without filters",
                response.stop_reason,
            )
            return QueryFilters()

        return response.parsed_output
