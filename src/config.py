MODEL = "claude-haiku-4-5-20251001"

# Logging
LOG_FILE = "logs/review.log"

# Diff chunking — diffs larger than this are truncated per-file before being
# sent to agents, keeping each LLM call within a safe token budget.
MAX_DIFF_CHARS = 40_000
