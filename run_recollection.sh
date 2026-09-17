#!/bin/bash
# Full trace re-collection via OpenRouter (pinned model IDs, preserved traces).
set -a; source .env; set +a
cd "$(dirname "$0")"
echo "== RE-COLLECTION: 6 models x 19 tasks x 10 runs =="
.venv/bin/python -m src.runners.run_experiment \
  --models gpt-4o-mini gpt-4o gpt-4.1 gpt-4.1-mini claude-sonnet-4 llama-3.3-70b \
  --runs 10 --output results
echo "== RE-COLLECTION COMPLETE =="
