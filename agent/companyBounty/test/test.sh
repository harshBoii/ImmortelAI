#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

API_URL="${API_URL:-http://localhost:8000}"

echo "============================================"
echo "  Company Bounty — API Integration Test"
echo "============================================"
echo ""

# ─── Build request payload ───
PAYLOAD=$(cat <<'EOF'
{
  "company": {
    "name": "WATI",
    "website": "https://www.wati.io",
    "linkedin": "https://www.linkedin.com/company/watiglobal/"
  },
  "brandEntity": {
    "category": "WhatsApp CRM",
    "topics": [
      "whatsapp crm",
      "whatsapp automation",
      "customer messaging",
      "chatbot platform"
    ],
    "keywords": [
      "whatsapp business api",
      "whatsapp marketing",
      "whatsapp chatbot"
    ]
  },
  "competitors": ["Twilio", "Respond.io", "Gupshup", "MessageBird"],
  "models": ["gpt-4o"],
  "session_id": "bounty-test-001"
}
EOF
)

echo "▸ Endpoint: POST $API_URL/company/bounty"
echo "▸ Company : WATI"
echo "▸ Category: WhatsApp CRM"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
INPUT_FILE="$RESULTS_DIR/run_${TIMESTAMP}_input.json"
OUTPUT_FILE="$RESULTS_DIR/run_${TIMESTAMP}_output.json"

# Save input
echo "$PAYLOAD" | python3 -m json.tool > "$INPUT_FILE"
echo "💾 Input saved to $INPUT_FILE"
echo ""

echo "▸ Calling API (this may take a minute) ..."
echo ""

HTTP_CODE=$(curl -s -o "$OUTPUT_FILE" -w "%{http_code}" \
  -X POST "$API_URL/company/bounty" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "▸ HTTP status: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" -ne 200 ]; then
  echo "❌ TEST FAILED — API returned HTTP $HTTP_CODE"
  echo ""
  echo "Response body:"
  cat "$OUTPUT_FILE"
  exit 1
fi

# Pretty-print the output in place
python3 -m json.tool "$OUTPUT_FILE" > "${OUTPUT_FILE}.tmp" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"

echo "💾 Output saved to $OUTPUT_FILE"
echo ""

# ─── Print summary ───
python3 -c "
import json, sys

with open('$OUTPUT_FILE') as f:
    result = json.load(f)

summary = result.get('summary', {})
niches  = result.get('niches', [])

print('════════════════════════════════════════════')
print('                 RESULTS')
print('════════════════════════════════════════════')
print()
print(f'Total niches : {summary.get(\"total_niches\", 0)}')
print(f'Total prompts: {summary.get(\"total_prompts\", 0)}')
print()

for niche in niches:
    print(f'── {niche[\"topic\"]} ({niche[\"prompt_count\"]} prompts) ──')
    print(f'   {niche.get(\"description\", \"\")}')
    for j, p in enumerate(niche.get('prompts', []), 1):
        print(f'   {j}. {p}')
    print()

if summary.get('total_niches', 0) > 0 and summary.get('total_prompts', 0) > 0:
    print('✅ TEST PASSED — niches and prompts generated successfully')
else:
    print('❌ TEST FAILED — no niches or prompts were generated')
    sys.exit(1)
"
