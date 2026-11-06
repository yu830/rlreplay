"""Source comparisons must never be promoted into runtime qualification."""

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/compare_upstream.py"
CANDIDATE = "c" * 40


@pytest.fixture
def tool():
    return runpy.run_path(str(TOOL))


def test_mutable_revision_is_frozen_and_downloaded_source_is_not_executed(
    tool, monkeypatch, tmp_path
):
    marker = tmp_path / "must-not-exist"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
    calls = []

    def fetch(url, *, timeout):
        calls.append((url, timeout))
        return (
            json.dumps({"sha": CANDIDATE}).encode()
            if "api.github.com" in url
            else payload
        )

    monkeypatch.setitem(tool["compare"].__globals__, "fetch", fetch)
    result = tool["compare"]("trl", "release/candidate", timeout=3.0)
    assert result["candidate_revision"] == CANDIDATE
    assert result["source_compare"] == "MATCH"
    assert result["runtime_qualification"] == "NOT_RUN"
    assert calls[0][0].endswith("/commits/release%2Fcandidate")
    assert len(calls) == 5
    assert all(timeout == 3.0 for _, timeout in calls)
    assert all("/release/" not in url for url, _ in calls[1:])
    assert (
        result["files"][0]["candidate"]["sha256"] == hashlib.sha256(payload).hexdigest()
    )
    assert not marker.exists()


def test_changed_owner_does_not_claim_runtime_failure(tool, monkeypatch):
    def fetch(url, *, timeout):
        return (
            b"changed owner"
            if CANDIDATE in url and url.endswith("groupwise.py")
            else b"same"
        )

    monkeypatch.setitem(tool["compare"].__globals__, "fetch", fetch)
    result = tool["compare"]("verl", CANDIDATE)
    assert result["source_compare"] == "CHANGED"
    assert [item["source_compare"] for item in result["files"]] == [
        "MATCH",
        "MATCH",
        "CHANGED",
    ]
    assert result["runtime_qualification"] == "NOT_RUN"


@pytest.mark.parametrize("framework", ["trl", "verl"])
def test_baseline_is_imported_from_adapter_and_same_sha_is_fetched_once(
    tool, monkeypatch, framework
):
    from rlreplay.adapters import trl, verl

    expected = {"trl": trl.SUPPORTED_REVISION, "verl": verl.SUPPORTED_REVISION}[
        framework
    ]
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        return b"same"

    monkeypatch.setitem(tool["compare"].__globals__, "fetch", fetch)
    result = tool["compare"](framework, expected)
    assert result["baseline_revision"] == expected
    assert result["candidate_revision"] == expected
    assert len(calls) == len(result["files"]) == len(set(calls))
    assert result["source_compare"] == "MATCH"
    assert all("api.github.com" not in url for url in calls)


def test_missing_owner_keeps_partial_comparisons_and_error(tool, monkeypatch):
    def fetch(url, *, timeout):
        if CANDIDATE in url and url.endswith("utils.py"):
            raise OSError("connection unavailable")
        return b"same"

    monkeypatch.setitem(tool["compare"].__globals__, "fetch", fetch)
    result = tool["compare"]("trl", CANDIDATE)
    assert result["source_compare"] == "UNAVAILABLE"
    assert result["runtime_qualification"] == "NOT_RUN"
    assert [item["source_compare"] for item in result["files"]] == [
        "MATCH",
        "UNAVAILABLE",
    ]
    assert result["files"][1]["candidate"]["reason"] == "connection unavailable"
    assert "sha256" in result["files"][1]["baseline"]


@pytest.mark.parametrize("response", [b'{"sha":"short"}', b"not JSON"])
def test_failed_resolution_does_not_fall_back_to_requested_ref(
    tool, monkeypatch, response
):
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        return response

    monkeypatch.setitem(tool["compare"].__globals__, "fetch", fetch)
    result = tool["compare"]("trl", "main")
    assert result["candidate_revision"] is None
    assert result["source_compare"] == "UNAVAILABLE"
    assert result["runtime_qualification"] == "NOT_RUN"
    assert result["files"] == []
    assert result["resolution_error"]
    assert len(calls) == 1


def test_cli_retains_unavailable_report_and_refuses_overwrite(
    tool, monkeypatch, tmp_path, capsys
):
    def fetch(url, *, timeout):
        raise OSError("offline")

    monkeypatch.setitem(tool["compare"].__globals__, "fetch", fetch)
    destination = tmp_path / "source-watch.json"
    arguments = ["trl", "main", "--output", str(destination)]
    assert tool["main"](arguments) == 4
    result = json.loads(capsys.readouterr().out)
    assert result["source_compare"] == "UNAVAILABLE"
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    original = destination.read_bytes()
    assert tool["main"](arguments) == 4
    assert json.loads(capsys.readouterr().err)["state"] == "ERROR"
    assert destination.read_bytes() == original


def test_authenticated_source_uses_cli_raw_response_without_reading_tokens(
    tool, monkeypatch
):
    calls = []
    payload = b"print('source')\r\n"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, payload, b"")

    monkeypatch.setattr(subprocess, "run", run)
    actual = tool["fetch_authenticated"](
        f"https://raw.githubusercontent.com/huggingface/trl/{CANDIDATE}/trl/trainer/utils.py",
        timeout=3,
    )
    assert actual == payload
    command, options = calls[0]
    assert command[:6] == ["gh", "api", "--hostname", "github.com", "--method", "GET"]
    assert (
        command[6]
        == f"repos/huggingface/trl/contents/trl/trainer/utils.py?ref={CANDIDATE}"
    )
    assert command[-2:] == ["--header", "Accept: application/vnd.github.raw+json"]
    assert options["env"]["GH_DEBUG"] == ""
    assert options["timeout"] == 3


def test_authentication_applies_to_resolution_and_all_source_requests(
    tool, monkeypatch
):
    calls = []

    def authenticated(url, *, timeout):
        calls.append(url)
        return (
            json.dumps({"sha": CANDIDATE}).encode()
            if "api.github.com" in url
            else b"source"
        )

    def anonymous(*args, **kwargs):
        raise AssertionError("authenticated mode must not fall back to anonymous")

    monkeypatch.setitem(
        tool["compare"].__globals__, "fetch_authenticated", authenticated
    )
    monkeypatch.setitem(tool["compare"].__globals__, "fetch", anonymous)
    result = tool["compare"]("trl", "main", auth="gh")
    assert result["auth"] == "gh"
    assert result["source_compare"] == "MATCH"
    assert result["runtime_qualification"] == "NOT_RUN"
    assert len(calls) == 5


@pytest.mark.parametrize("failure", ["login", "timeout", "missing-cli"])
def test_authentication_failure_is_retained_without_anonymous_retry(
    tool, monkeypatch, failure
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 2)
        if failure == "missing-cli":
            raise FileNotFoundError("gh not found")
        return subprocess.CompletedProcess(command, 1, b"", b"GitHub login required")

    monkeypatch.setattr(subprocess, "run", run)
    result = tool["compare"]("trl", "main", auth="gh", timeout=2)
    assert result["source_compare"] == "UNAVAILABLE"
    assert result["resolution_error"]
    assert result["runtime_qualification"] == "NOT_RUN"
    assert len(calls) == 1


def test_authenticated_fetch_does_not_send_requests_to_other_hosts(tool, monkeypatch):
    def run(*args, **kwargs):
        raise AssertionError("must reject before invoking GitHub CLI")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="limited to GitHub"):
        tool["fetch_authenticated"]("https://example.com/source", timeout=2)
