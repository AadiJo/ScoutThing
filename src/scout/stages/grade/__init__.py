"""grade: reads facts.json, match.json, tracks.parquet, identity.json, calibration.json, the
video and settings.json.

Writes grade.json (see `scout.schemas.grade`): 1-10 scores for three driving skills with
cited evidence, the ledgers and defense episodes they rest on, timestamped notes and an
unscored auto report, written by Claude Code run headless (`claude -p`). Also writes
grade.md, the same grade as a short readable report, and `grade_work/` with the prompts,
every contact sheet served, the frame log and each session's stream.

Steps:

1. Code builds the driving ledger (barrier lines and trips out of its own zone in its own
   active time), the readiness ledger and the list of defense items from the facts
   (`ledgers.ledgers`). Each barrier and readiness line gets a default ruling computed from
   measured features (`_turned_back_default`, `_readiness`), and a charged barrier line the
   part a robot standing on its way home cost (`_split`), so a line is decided the same way on
   every run.
2. Build the prompt from files (`prompt.prompts`): `grade_system.md` and `grade_task.md` in
   this package (game-independent), and the game guide, limits and rubric under
   `scout/games/rebuilt2026_grade/`. The task carries the match timeline, the ledgers, the
   defense items and all of facts.json.
3. Run `claude -p` sealed (`claude.py`): our system prompt replaces the default one, no
   built-in tools, no settings files, skills or other MCP servers, and one MCP tool, `frames`,
   served by `python -m scout.stages.grade <run>` (`frames.serve_frames`). The tool crops the
   main camera around the target, rings it, boxes every other tracked robot with its team and
   partner or opponent (or "unlabelled"), or shows a corner camera panel unmarked; it tiles up
   to four moments with the match clock, lists the facts overlapping them, logs every sheet to
   `grade_work/frames.jsonl` and refuses past IMAGE_CAP sheets. The session's init message is
   checked so a leak fails the stage.
4. Check the structured answer (`check.check_answer`): citations, a hopper reading per trip
   tile (code turns them into the trip's ruling, `check_calls.hopper_ruling`), each
   overrule of a default backed by the checkable contradiction it names, episodes covering
   every defense item, numbers only as references to fact fields, team names only when
   labelled or read, no purpose words, places that agree with where the target was, and no
   text reading a stretch against its ruling. On failure the session is resumed with the exact
   errors, up to ATTEMPTS answers in all; then the stage fails.
5. Code computes the charged seconds (`scores.charges`) and every score (`scores.scores`); the
   grader only writes why.

Modules, each importing only ones earlier in this list: `base`, `clock`, `tracks`,
`ledgers`, `scores`, `checker`, `check_episodes`, `check_calls`, `check`, `frames`, `prompt`,
`claude`, `report`, `stage`.
"""

from scout.stages.grade.stage import STAGE

__all__ = ["STAGE"]
