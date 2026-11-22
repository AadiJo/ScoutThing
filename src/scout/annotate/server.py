"""The local server behind `scout annotate`: one page, the source video with HTTP range
support, the run's match and pipeline data, and the truth file it edits.

    serve(RunDir(Path("runs/Qualification_16")), truth_path, host="0.0.0.0", port=8765)

Routes:
- `GET /` the page (`page.html`), `GET /video` the source video (range requests).
- `GET /api/session` video facts, roster, panels, timeline and every robot's truth.
- `GET /api/tracks` the pipeline's measured boxes and team labels, for the faint guide.
- `PUT /api/robot/<team>` replace one robot's samples. The body names the `base` hash the
  page last saw; a different hash on disk (another tab saved) is refused with 409, so two
  pages never overwrite each other silently. Every accepted save rewrites the truth file
  atomically.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from scout.games.rebuilt2026 import BUMPER_CENTER_HEIGHT_M
from scout.run import RunDir
from scout.schemas.calibration import Calibration
from scout.schemas.common import VIEW_NAMES, ViewName
from scout.schemas.identity import Identity
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks
from scout.schemas.truth import Truth, TruthRobot, TruthSample
from scout.schemas.video import VideoInfo
from scout.video import frame_time

PAGE = Path(__file__).with_name("page.html")
CHUNK = 1 << 20
INSTRUCTION = (
    "Follow the centre of the robot's bumper ring: the middle of the robot at bumper height."
)

SampleRow = tuple[int, float, float, bool]
"""(frame, u, v, estimated): the compact sample the page sends and receives."""


class RobotPut(BaseModel):
    """Body of `PUT /api/robot/<team>`."""

    model_config = ConfigDict(extra="forbid")

    base: str | None
    samples: list[SampleRow]


class ConflictError(Exception):
    """The truth on disk changed since the page loaded it."""


class Session:
    """The run and truth file one server edits. Handler threads share it under `lock`."""

    def __init__(self, run: RunDir, truth_path: Path) -> None:
        self.run = run
        self.info = VideoInfo.load(run.video_json)
        self.match = MatchInfo.load(run.match_json)
        if "main" not in Calibration.load(run.calibration_json).views:
            raise ValueError(f"{run.calibration_json} has no calibrated main view")
        self.truth_path = truth_path
        self.lock = threading.Lock()
        name = Path(self.info.source_path).name
        if truth_path.exists():
            self.truth = Truth.load(truth_path)
            if self.truth.video_sha256 != self.info.sha256:
                raise ValueError(
                    f"{truth_path} is for video sha256 {self.truth.video_sha256[:12]}..., "
                    f"but {run.path} holds {self.info.sha256[:12]}..."
                )
        else:
            self.truth = Truth(
                video_sha256=self.info.sha256,
                video_name=name,
                source="hand",
                click_height_m=BUMPER_CENTER_HEIGHT_M,
                notes="",
                robots=[],
            )

    def view_of(self, u: float, v: float) -> ViewName | None:
        """The broadcast panel holding full-frame pixel (u, v)."""
        for name in VIEW_NAMES:
            r = self.match.views.get(name)
            if r and r.x <= u < r.x + r.w and r.y <= v < r.y + r.h:
                return name
        return None

    def session_json(self) -> dict[str, object]:
        """Everything the page needs at load."""
        m, i = self.match, self.info
        with self.lock:
            robots = {str(r.team): _rows_json(r) for r in self.truth.robots}
        return {
            "video": {
                "name": Path(i.source_path).name,
                "fps": i.fps,
                "frame_count": i.frame_count,
                "width": i.width,
                "height": i.height,
                "sha256": i.sha256,
            },
            "match": {
                "roster": [s.model_dump() for s in m.roster],
                "views": {k: v.model_dump() for k, v in m.views.items()},
                "target_team": m.target_team,
                "auto_start_t": m.auto_start_t,
                "auto_end_t": m.auto_end_t,
                "teleop_start_t": m.teleop_start_t,
                "teleop_end_t": m.teleop_end_t,
            },
            "truth_path": str(self.truth_path),
            "click_height_m": self.truth.click_height_m,
            "instruction": INSTRUCTION,
            "has_tracks": self.run.tracks.exists() and self.run.identity_json.exists(),
            "robots": robots,
        }

    def tracks_json(self) -> list[dict[str, object]]:
        """Measured pipeline samples per track with their labelled stretches."""
        df = read_tracks(self.run.tracks).filter(state="measured").sort("track_id", "frame")
        labels: dict[int, list[tuple[int, int, int]]] = {}
        for s in Identity.load(self.run.identity_json).segments:
            if s.team is not None:
                labels.setdefault(s.track_id, []).append((s.frame_start, s.frame_end, s.team))
        out: list[dict[str, object]] = []
        for (tid,), g in df.group_by("track_id", maintain_order=True):
            track_id = int(str(tid))
            boxes = g.select("x1", "y1", "x2", "y2").to_numpy().round(1).tolist()
            out.append(
                {
                    "id": track_id,
                    "color": str(g["color"][0]),
                    "frames": g["frame"].to_list(),
                    "boxes": boxes,
                    "labels": labels.get(track_id, []),
                }
            )
        return out

    def put_robot(self, team: int, body: RobotPut) -> dict[str, object]:
        """Replace `team`'s samples and rewrite the truth file. Raises ConflictError or
        ValueError (bad team, frame or pixel)."""
        if team not in {s.team for s in self.match.roster}:
            raise ValueError(f"team {team} is not in this match's roster")
        samples: list[TruthSample] = []
        for frame, u, v, est in sorted(body.samples):
            if not 0 <= frame < self.info.frame_count:
                raise ValueError(f"frame {frame} is outside the video")
            view = self.view_of(u, v)
            if view is None:
                raise ValueError(f"pixel ({u}, {v}) is in no camera panel")
            samples.append(
                TruthSample(
                    frame=frame,
                    t=frame_time(self.info, frame),
                    view=view,
                    u=round(u, 1),
                    v=round(v, 1),
                    state="estimated" if est else "visible",
                )
            )
        robot = TruthRobot(team=team, samples=samples)
        with self.lock:
            current = self.truth.robot(team)
            if body.base != (_hash(current) if current else None):
                raise ConflictError(
                    f"team {team} was saved from another page since this one loaded"
                )
            others = [r for r in self.truth.robots if r.team != team]
            kept = [*others, robot] if samples else others
            order = [s.team for s in self.match.roster]
            kept.sort(key=lambda r: order.index(r.team) if r.team in order else len(order))
            truth = self.truth.model_copy(update={"robots": kept})
            self.truth_path.parent.mkdir(parents=True, exist_ok=True)
            truth.save(self.truth_path)
            self.truth = truth
        return {"hash": _hash(robot) if samples else None, "count": len(samples)}


def _hash(robot: TruthRobot) -> str:
    """Content hash of one robot's samples, the `base` a page saves against."""
    rows = [(s.frame, s.u, s.v, s.state) for s in robot.samples]
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:16]


def _rows_json(robot: TruthRobot) -> dict[str, object]:
    rows = [[s.frame, s.u, s.v, s.state == "estimated"] for s in robot.samples]
    return {"samples": rows, "hash": _hash(robot)}


def _handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/video":
                self._video()
            elif path == "/api/session":
                self._json(HTTPStatus.OK, session.session_json())
            elif path == "/api/tracks" and session.run.tracks.exists():
                self._json(HTTPStatus.OK, session.tracks_json())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {path}"})

        def do_PUT(self) -> None:
            m = re.fullmatch(r"/api/robot/(\d+)", self.path)
            if not m:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                result = session.put_robot(int(m[1]), RobotPut.model_validate_json(raw))
            except ConflictError as e:
                self._json(HTTPStatus.CONFLICT, {"error": str(e)})
            except (ValidationError, ValueError) as e:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            else:
                self._json(HTTPStatus.OK, result)

        def _json(self, status: HTTPStatus, obj: object) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _video(self) -> None:
            """Serve the source video, honouring a single `Range: bytes=a-b` request."""
            path = session.run.video.resolve()
            size = path.stat().st_size
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
            partial = bool(m and (m[1] or m[2]))
            if m and partial:
                if m[1]:
                    start, end = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
                else:
                    start = max(0, size - int(m[2]))
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0 and (buf := f.read(min(CHUNK, left))):
                        self.wfile.write(buf)
                        left -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Handler


def serve(run: RunDir, truth_path: Path, host: str, port: int) -> None:
    """Serve the page until Ctrl+C. Refuses a truth file made on another video."""
    session = Session(run, truth_path)
    server = ThreadingHTTPServer((host, port), _handler(session))
    server.daemon_threads = True
    n = sum(len(r.samples) for r in session.truth.robots)
    print(f"truth file: {truth_path} ({len(session.truth.robots)} robots, {n} samples)", flush=True)
    print(
        f"annotate at http://localhost:{port}/ (listening on {host}:{port}; Ctrl+C stops)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
