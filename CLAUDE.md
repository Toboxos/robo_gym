# CLAUDE.md — MazeBot RL GYM

## Project
2D robot simulation framework offering a gym environment for training RL agents for the RoboCup Junior Maze Entry Competition.
Spec and no-code architecture docs live in Notion.
Tasks are tracked as GitHub Issues.

## Key references
- Notion:           https://www.notion.so/MazeBot-Simulator-v1-0-Functional-Specification-3244de87ad09811b9863c37c4edf80d3
- GitHub Issues:    https://github.com/Toboxos/robo_gym/issues

## Environment — mandatory
- Python 3.13+
- Use `uv` for everything: venv creation, package installation, running scripts
  - Create venv:       uv venv
  - Install packages:  uv add <package>
  - Run scripts:       uv run python <script.py>
  - Do NOT use pip, poetry, or conda directly
- Lock file (uv.lock) must be committed after any dependency change

## Code conventions
- Type hints on all public functions and dataclasses
- `@dataclass` over plain dicts for structured data
- Docstring on every public function (one-liner is fine)
- No print() in library code — use the stdlib `logging` module

## Before starting a task
1. Read the Notion spec page linked in the GitHub Issue description
2. Check the Notion decision log for relevant prior decisions
3. Run `uv run pytest` — confirm green baseline before touching anything

## When finishing a task
1. Ensure `uv run pytest` is green
2. Update uv.lock if dependencies changed (`uv lock`)
3. Leave a comment on the GitHub Issue summarising what was done
   and any decisions made — keep it to 3–5 bullet points