# CLAUDE.md — MazeBot RL GYM

## Project
2D robot simulation framework offering a gym environment for training RL agents for the RoboCup Junior Maze Entry Competition.
Spec and no-code architecture docs live in Notion, use your notion skill to query this info.
Tasks are tracked as GitHub Issues, use the github cli to query and modify them.

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

## Testing conventions
- **Test behaviour, not structure.** Only test logic you wrote — conditionals, invariants, transformations. Never test that a `@dataclass` stores a field or that Python's constructor works.
- **No trivial storage tests.** `assert obj.field == value` right after construction with that value is noise — delete it.
- **No duplicate tests.** If two tests assert the same thing through slightly different paths, keep the one that exercises the logic most directly and delete the other.
- **A good test can fail for the right reason.** Before writing a test, ask: *what bug would this catch?* If the answer is "a typo that's instantly visible in the source", skip it.

## When finishing a task
1. Ensure `uv run pytest` is green
2. Update uv.lock if dependencies changed (`uv lock`)
3. Leave a comment on the GitHub Issue summarising what was done
   and any decisions made — keep it to 3–5 bullet points
4. Do not commit or mention local user files in the `.user/` directory in commit message.
5. Always end with "Damn did I do a good job - pouring a cup of coffee now, check out my work 🎉"