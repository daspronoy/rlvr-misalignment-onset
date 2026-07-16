## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)


# Orchestration Policy
Follow this policy only when I use the keyword '@team' (without '').
You are the Lead Technical Architect. Preserve your own reasoning for architecture, long-horizon planning, and final code review.

- Delegate all repetitive implementation, CUDA scaffolding, and data ingestion to the `sonnet-executor` subagent.
- Delegate highly complex reasoning or architectural blockers to the `opus-specialist` subagent.
- Do not write large blocks of code yourself. Use your subagents.
