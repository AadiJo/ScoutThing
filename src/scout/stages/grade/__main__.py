"""`python -m scout.stages.grade <run>`: the `frames` MCP server Claude Code starts for a
grader session (see `frames.serve_frames`)."""

from __future__ import annotations

import sys
from pathlib import Path

from scout.stages.grade.frames import serve_frames

if __name__ == "__main__":
    serve_frames(Path(sys.argv[1]))
