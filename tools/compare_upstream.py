"""Compare immutable upstream owner files without changing runtime qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from rlreplay.adapters.trl import SUPPORTED_REVISION as TRL_REVISION
from rlreplay.adapters.verl import SUPPORTED_REVISION as VERL_REVISION

SOURCES = {
    "trl": (
        "huggingface/trl",
        TRL_REVISION,
        ("trl/trainer/grpo_trainer.py", "trl/trainer/utils.py"),
    ),
    "verl": (
        "verl-project/verl",
        VERL_REVISION,
        (
            "verl/trainer/ppo/ray_trainer.py",
            "verl/trainer/ppo/core_algos.py",
            "verl/utils/groupwise.py",
        ),
    ),
}
SHA = re.compile(r"[0-9a-fA-F]{40}")


def fetch(url: str, *, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "RLReplay-source-watch"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_authenticated(url: str, *, timeout: float) -> bytes:
    """Let GitHub CLI own credentials and return exact response bytes over GET."""
    parsed = urlsplit(url)
    headers = []
    if parsed.scheme != "https":
        raise ValueError("authenticated requests require GitHub HTTPS URLs")
    if parsed.netloc == "api.github.com":
        endpoint = parsed.path.lstrip("/")
        if parsed.query:
            endpoint += "?" + parsed.query
    elif parsed.netloc == "raw.githubusercontent.com":
        owner, repository, revision, path = parsed.path.lstrip("/").split("/", 3)
        endpoint = f"repos/{owner}/{repository}/contents/{path}?ref={revision}"
        headers = ["--header", "Accept: application/vnd.github.raw+json"]
    else:
        raise ValueError("authenticated requests are limited to GitHub API/source URLs")
    try:
        process = subprocess.run(
            [
                "gh",
                "api",
                "--hostname",
                "github.com",
                "--method",
                "GET",
                endpoint,
                *headers,
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "GH_DEBUG": ""},
        )
    except subprocess.TimeoutExpired as error:
        raise OSError("GitHub CLI request timed out") from error
    if process.returncode:
        reason = process.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"GitHub CLI exited {process.returncode}: {reason}")
    return process.stdout


def resolve(repository: str, revision: str, *, timeout: float, fetcher=None) -> str:
    if SHA.fullmatch(revision):
        return revision.lower()
    url = (
        f"https://api.github.com/repos/{repository}/commits/{quote(revision, safe='')}"
    )
    value = json.loads((fetcher or fetch)(url, timeout=timeout))
    sha = value["sha"]
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        raise ValueError("upstream response did not contain a full commit SHA")
    return sha.lower()


def compare(
    framework: str, revision: str, *, timeout: float = 15.0, auth="anonymous"
) -> dict:
    """Hash source bytes only; MATCH does not establish execution compatibility."""
    if not revision.strip():
        raise ValueError("candidate revision must not be empty")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if auth not in {"anonymous", "gh"}:
        raise ValueError("auth must be anonymous or gh")
    fetcher = fetch_authenticated if auth == "gh" else fetch
    repository, baseline, paths = SOURCES[framework]
    result = {
        "format": "rlreplay-upstream-source-comparison/1",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "framework": framework,
        "auth": auth,
        "repository": repository,
        "baseline_revision": baseline,
        "candidate_ref": revision,
        "candidate_revision": None,
        "source_compare": "UNAVAILABLE",
        "runtime_qualification": "NOT_RUN",
        "files": [],
    }
    try:
        candidate = resolve(repository, revision, timeout=timeout, fetcher=fetcher)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["resolution_error"] = {
            "error_type": type(error).__name__,
            "reason": str(error),
        }
        return result
    result["candidate_revision"] = candidate
    retrieved = {}

    def source(sha, path):
        url = f"https://raw.githubusercontent.com/{repository}/{sha}/{path}"
        if url not in retrieved:
            try:
                payload = fetcher(url, timeout=timeout)
            except OSError as error:
                retrieved[url] = {
                    "url": url,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            else:
                retrieved[url] = {
                    "url": url,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
        return retrieved[url]

    for path in paths:
        original = source(baseline, path)
        proposed = source(candidate, path)
        state = (
            "UNAVAILABLE"
            if "sha256" not in original or "sha256" not in proposed
            else ("MATCH" if original["sha256"] == proposed["sha256"] else "CHANGED")
        )
        result["files"].append(
            {
                "path": path,
                "source_compare": state,
                "baseline": original,
                "candidate": proposed,
            }
        )
    states = {item["source_compare"] for item in result["files"]}
    result["source_compare"] = (
        "UNAVAILABLE"
        if "UNAVAILABLE" in states
        else ("CHANGED" if "CHANGED" in states else "MATCH")
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=tuple(SOURCES))
    parser.add_argument(
        "revision", help="candidate commit, tag or branch (such as main)"
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="seconds per HTTP request"
    )
    parser.add_argument(
        "--output", type=Path, help="new JSON file retaining the report"
    )
    parser.add_argument(
        "--auth",
        choices=("anonymous", "gh"),
        default="anonymous",
        help="use gh to query with the GitHub CLI's active login",
    )
    args = parser.parse_args(argv)
    try:
        result = compare(
            args.framework, args.revision, timeout=args.timeout, auth=args.auth
        )
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
    except (OSError, ValueError) as error:
        print(json.dumps({"state": "ERROR", "reason": str(error)}), file=sys.stderr)
        return 4
    print(rendered, end="")
    return 4 if result["source_compare"] == "UNAVAILABLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
