# Claude Code Token Rules

- Do not scan the entire repository unless explicitly requested.
- Start by inspecting only the folder tree.
- Read only files directly relevant to the current task.
- Before editing, list the files you need to read and modify.
- Avoid generated and dependency folders:
  - node_modules
  - .next
  - target
  - build
  - dist
  - coverage
  - logs
  - .git
- Keep explanations short.
- Prefer minimal diffs.
- Do not rewrite unrelated code.