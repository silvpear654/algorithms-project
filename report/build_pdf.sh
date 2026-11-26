#!/bin/bash
# report.html → report.pdf (Google Chrome 헤드리스 인쇄, 수식은 KaTeX CDN으로 렌더링하므로 인터넷 필요)
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || CHROME=$(command -v google-chrome || command -v chromium)
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --virtual-time-budget=15000 --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null
echo "saved → $PWD/report.pdf"
