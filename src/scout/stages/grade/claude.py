"""The sealed `claude -p` session: find a logged-in Claude Code (`claude`), build the command
line with our prompt and our one tool (`command`), run it with the parent session's variables
dropped and check from its init message that nothing else loaded (`run_claude`)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from scout.run import RunDir
from scout.schemas.grade import GradeAnswer
from scout.stages.grade.base import JSON, SERVER, TOOL

EFFORT = "medium"
"""Claude Code effort level. Medium keeps a grade between 40 and 120 s on Opus."""
MAX_BUDGET_USD = 8.0
"""Stops a runaway session; one grade lists at about 0.3 to 1 USD."""
TIMEOUT_S = 600.0
TOOL_ID = f"mcp__{SERVER}__{TOOL}"
SESSION_TOOLS = (TOOL_ID, "StructuredOutput")
"""The only tools a grader session may load: ours, and the one `--json-schema` adds."""

ENV_KEEP = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CONFIG_DIR",
)
"""Claude and Anthropic variables the child keeps: credentials and where they live. Every
other `CLAUDE*`, `ANTHROPIC_*` and agent marker from a parent session is dropped."""
ENV_DROP_PREFIXES = ("CLAUDE", "ANTHROPIC_", "AI_AGENT", "CODEX_", "MCP_")


def _child_env() -> dict[str, str]:
    """This process's environment without the variables a parent Claude Code session
    leaks (session ids, entrypoint, effort, messaging sockets), keeping credentials."""
    return {
        k: v for k, v in os.environ.items() if k in ENV_KEEP or not k.startswith(ENV_DROP_PREFIXES)
    }


def claude() -> str:
    """Path to `claude`, checked to be installed and logged in."""
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError(
            "grade needs Claude Code: `claude` is not on PATH. Install it "
            "(https://docs.claude.com/claude-code) and run `claude` once to log in."
        )
    out = subprocess.run(
        [exe, "auth", "status"], capture_output=True, text=True, env=_child_env(), timeout=60
    )
    try:
        status = TypeAdapter(dict[str, object]).validate_json(out.stdout)
    except ValidationError:
        status = {}
    if status.get("loggedIn") is not True:
        raise RuntimeError(
            "grade needs Claude Code logged in: `claude auth status` says it is not. "
            "Run `claude` and log in (or set ANTHROPIC_API_KEY), then rerun."
        )
    return exe


def command(exe: str, model: str, run: RunDir, system: str, resume: str | None) -> list[str]:
    """The sealed `claude -p` command line: our prompt and our one tool, no built-in tools,
    settings files, skills or other MCP servers. `resume` continues a session for the retry."""
    mcp = {
        "mcpServers": {
            SERVER: {
                "command": sys.executable,
                "args": ["-m", "scout.stages.grade", str(run.path.resolve())],
            }
        }
    }
    cmd = [
        exe, "-p",
        "--model", model,
        "--effort", EFFORT,
        "--output-format", "stream-json", "--verbose",
        "--json-schema", json.dumps(GradeAnswer.model_json_schema()),
        "--system-prompt", system,
        "--tools", "",
        "--mcp-config", json.dumps(mcp), "--strict-mcp-config",
        "--allowedTools", TOOL_ID,
        "--permission-mode", "dontAsk",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--max-budget-usd", str(MAX_BUDGET_USD),
    ]  # fmt: skip
    if resume is not None:
        cmd += ["--resume", resume]
    return cmd


@dataclass(frozen=True)
class _Session:
    """What one `claude -p` call returned."""

    session_id: str
    answer: object
    result: JSON
    init: JSON


def as_dict(v: object) -> JSON:
    return TypeAdapter(dict[str, object]).validate_python(v) if isinstance(v, dict) else {}


def run_claude(cmd: list[str], prompt: str, cwd: Path, log: Path) -> _Session:
    """Run one session, save its stream to `log`, check the sealing, return the result."""
    with log.open("w") as out:
        proc = subprocess.run(
            cmd,
            input=prompt,
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=_child_env(),
            timeout=TIMEOUT_S,
        )
    adapter = TypeAdapter(dict[str, object])
    init: JSON = {}
    result: JSON = {}
    for line in log.read_text().splitlines():
        if not line.startswith("{"):
            continue
        msg = adapter.validate_json(line)
        if msg.get("type") == "system" and msg.get("subtype") == "init":
            init = msg
        elif msg.get("type") == "result":
            result = msg
    if not result:
        raise RuntimeError(
            f"claude exited {proc.returncode} without a result; stderr: {proc.stderr[-2000:]}"
        )
    _check_sealed(init)
    if result.get("is_error") or result.get("subtype") != "success":
        raise RuntimeError(
            f"claude session failed ({result.get('subtype')}): {str(result.get('result'))[:500]}"
        )
    return _Session(str(result.get("session_id")), result.get("structured_output"), result, init)


def _check_sealed(init: JSON) -> None:
    """Fail unless the session loaded exactly SESSION_TOOLS, our MCP server and no skills,
    slash commands or user plugins."""
    tools = init.get("tools")
    servers = init.get("mcp_servers")
    plugins = init.get("plugins")
    problems: list[str] = []
    if not isinstance(tools, list) or sorted(map(str, tools)) != sorted(SESSION_TOOLS):
        problems.append(f"tools {tools}")
    if not isinstance(servers, list) or [as_dict(s).get("name") for s in servers] != [SERVER]:
        problems.append(f"mcp servers {servers}")
    elif as_dict(servers[0]).get("status") != "connected":
        problems.append(f"frame tool not connected: {servers}")
    if init.get("skills") or init.get("slash_commands"):
        problems.append(f"skills {init.get('skills')} commands {init.get('slash_commands')}")
    if isinstance(plugins, list) and any(as_dict(p).get("path") != "builtin" for p in plugins):
        problems.append(f"plugins {plugins}")
    if problems:
        raise RuntimeError("grader session is not sealed: " + "; ".join(problems))
