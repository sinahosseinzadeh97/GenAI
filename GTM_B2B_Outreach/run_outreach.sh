#!/usr/bin/env bash
set -euo pipefail

API_BASE="http://localhost:3000"

# 1) شروع پایپلاین و گرفتن task_id
RESP=$(curl -s -X POST "$API_BASE/api/run" \
  -H "Content-Type: application/json" \
  -d '{
    "target_desc":"B2B SaaS CRM companies in fintech (EU, 11-200 employees)",
    "offering_desc":"AI outreach automation for SDRs (personalized emails from website + Reddit).",
    "sender_name":"Sina",
    "sender_company":"Our Company",
    "calendar_link":"https://cal.com/sina/15min",
    "num_companies":2,
    "email_style":"Professional"
  }')

TASK_ID=$(printf "%s" "$RESP" | sed -n 's/.*"task_id":"\([^"]*\)".*/\1/p')
echo "Task ID: $TASK_ID"

# 2) پیگیری پیشرفت با SSE
# 2) پیگیری پیشرفت با SSE (نسخه سازگار با macOS)
echo -n "Progress: "
curl -sN "$API_BASE/api/progress/$TASK_ID" | while read -r line; do
  if [[ $line == data:* ]]; then
    pct=$(echo "$line" | sed 's/[^0-9]*//g')
    if [[ -n "$pct" ]]; then
      echo -ne "\rProgress: ${pct}%"
    fi
  fi
done
echo ""
# 3) گرفتن نتیجه نهایی
echo "Fetching result..."
if command -v jq >/dev/null 2>&1; then
  curl -s "$API_BASE/api/result/$TASK_ID" | jq .
else
  curl -s "$API_BASE/api/result/$TASK_ID" | python3 -m json.tool
fi
