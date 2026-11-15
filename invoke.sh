#!/usr/bin/env bash
# Send a search query to the Job RAG API and print the formatted response.
#
# Usage:
#   ./invoke.sh "remote junior react jobs in Canada"
#   ./invoke.sh                                            # uses an example query
#   API_URL=http://localhost:8001 ./invoke.sh "python developer"
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
QUERY="${1:-remote python backend engineer}"

# Make sure the API is up (and can reach Qdrant) before searching
if ! HEALTH=$(curl -s -f "$API_URL/health"); then
    echo "API is not reachable or not healthy at $API_URL" >&2
    echo "Start it with: docker compose up -d" >&2
    exit 1
fi

# Build the JSON body safely, so quotes or special characters in the query don't break it
PAYLOAD=$(python3 -c 'import json, sys; print(json.dumps({"query": sys.argv[1]}))' "$QUERY")

RESPONSE_FILE=$(mktemp)
trap 'rm -f "$RESPONSE_FILE"' EXIT

echo "Searching: $QUERY"
read -r STATUS SECONDS_TAKEN < <(
    curl -s -o "$RESPONSE_FILE" -w "%{http_code} %{time_total}\n" \
        -X POST "$API_URL/search" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD"
)
echo "HTTP $STATUS in ${SECONDS_TAKEN}s"

# Pretty-print JSON; fall back to the raw body if it isn't JSON
python3 -m json.tool "$RESPONSE_FILE" 2>/dev/null || cat "$RESPONSE_FILE"

# Exit with an error code if the search failed, so other scripts can check it
[ "$STATUS" = "200" ]
