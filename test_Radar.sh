#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/company/radar}"
OUTPUT_FILE="${OUTPUT_FILE:-radar_result.json}"

INPUT='{
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
  "competitors": [
    "Twilio",
    "Respond.io",
    "Gupshup",
    "MessageBird"
  ],
  "models": [
    "gpt-4o",
    "claude-3.5",
    "gemini-1.5"
  ]
}'

OUTPUT=$(curl -sS -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "$INPUT")

echo "$OUTPUT" | jq .

jq -n \
  --argjson input "$(echo "$INPUT" | jq -c .)" \
  --argjson output "$(echo "$OUTPUT" | jq -c .)" \
  '{ input: $input, output: $output }' > "$OUTPUT_FILE"

echo ""
echo "Saved input + output to $OUTPUT_FILE"
