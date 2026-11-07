"""Qualify pinned real CPU framework boundaries with synthetic local inputs."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTORY = "examples.trl_native_replay:make_trainer"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, label, arguments, *, expected=0, candidate=None):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(str(p) for p in (candidate, ROOT) if p),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        WANDB_DISABLED="true",
        TOKENIZERS_PARALLELISM="false",
        ACCELERATE_USE_CPU="true",
    )
    command = [sys.executable, *map(str, arguments)]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (output / f"{label}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != expected:
        raise AssertionError(
            f"{label}: exit {result.returncode}, expected {expected}; see {output}"
        )
    return result


def candidate_copy(output, package):
    spec = importlib.util.find_spec(package)
    assert spec and spec.submodule_search_locations
    source = Path(next(iter(spec.submodule_search_locations)))
    target = output / package
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return target


def replace_once(path, old, new):
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"candidate source anchor changed in {path}"
    path.write_text(source.replace(old, new), encoding="utf-8")


def run_verl(output):
    example = ROOT / "examples/verl_observer.py"
    run(output, "healthy", [example, output / "healthy"])
    healthy = json.loads((output / "healthy/result.json").read_text(encoding="utf-8"))
    assert (
        healthy["finding"]["verdict"]
        == healthy["native_replay_finding"]["verdict"]
        == "PASS"
    )
    run(
        output,
        "known-no-std",
        [example, output / "no-std", "--case", "known-no-std"],
        expected=2,
    )
    fault = json.loads((output / "no-std/result.json").read_text(encoding="utf-8"))
    assert (
        fault["finding"]["verdict"]
        == fault["native_replay_finding"]["verdict"]
        == "SUSPECTED_MISMATCH"
    )
    capsule = Path(fault["capsule"])
    before = digest(capsule)
    regression = output / "regression"
    run(output, "export", ["-m", "rlreplay", "export-test", capsule, regression])
    pytest = [
        "-m",
        "pytest",
        "-q",
        "-c",
        "/dev/null",
        "--rootdir",
        regression,
        "--confcutdir",
        regression,
        "-p",
        "no:cacheprovider",
        regression,
    ]
    failure = run(output, "original-regression", pytest, expected=1)
    assert "SUSPECTED_MISMATCH" in failure.stdout and "1 failed" in failure.stdout

    candidate_root = output / "candidate"
    candidate_root.mkdir()
    candidate = candidate_copy(candidate_root, "verl")
    driver = candidate / "trainer/ppo/ray_trainer.py"
    anchor = '        if "uid" in data.non_tensor_batch:  # optional\n'
    replace_once(
        driver,
        anchor,
        '        if adv_estimator == AdvantageEstimator.GRPO_VECTORIZED:\n            adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo\n'
        + anchor,
    )
    fixed = run(output, "candidate-regression", pytest, candidate=candidate_root)
    assert "1 passed" in fixed.stdout
    assert digest(capsule) == before
    return {
        "healthy": "PASS",
        "known_no_std": "SUSPECTED_MISMATCH",
        "candidate_fix": "PASS",
        "original_capsule_sha256": before,
    }


def verify_no_forward(bundle):
    from examples.trl_native_replay import make_trainer

    from rlreplay.trl_replay import read_trl, run_trl

    trainer = make_trainer()
    methods = {
        name: getattr(trainer, name)
        for name in (
            "_generate",
            "_calculate_rewards",
            "_get_per_token_logps_and_entropies",
        )
    }
    ownership = {name: name in trainer.__dict__ for name in methods}
    calls = {"forward": 0, "reward": 0}

    def forbidden(name):
        def fail(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"unexpected {name} call")

        return fail

    trainer.model.forward = forbidden("forward")
    trainer.reward_funcs = [forbidden("reward")]
    result = run_trl(read_trl(bundle), trainer)
    assert result["finding"]["verdict"] == "PASS" and result["native_observation_equal"]
    assert calls == {"forward": 0, "reward": 0}
    assert all(getattr(trainer, name) == method for name, method in methods.items())
    assert all((name in trainer.__dict__) == flag for name, flag in ownership.items())
    print(json.dumps({"calls": calls, "provider_calls": result["provider_calls"]}))


def run_trl(output):
    run(output, "healthy", [ROOT / "examples/trl_native_replay.py", output / "healthy"])
    healthy = json.loads((output / "healthy/result.json").read_text(encoding="utf-8"))
    assert healthy["replay"]["finding"]["verdict"] == "PASS"
    assert healthy["replay"]["native_observation_equal"]
    bundle = Path(healthy["bundle"])
    before = digest(bundle)
    run(output, "no-forward", [__file__, "--no-forward", bundle])
    regression = output / "regression"
    run(
        output,
        "export",
        [
            "-m",
            "rlreplay",
            "export-test",
            bundle,
            regression,
            "--runner",
            "trl",
            "--trainer-factory",
            FACTORY,
        ],
    )
    pytest = [
        "-m",
        "pytest",
        "-q",
        "-c",
        "/dev/null",
        "--rootdir",
        regression,
        "--confcutdir",
        regression,
        "-p",
        "no:cacheprovider",
        regression,
    ]
    replay = [
        "-m",
        "rlreplay",
        "replay",
        bundle,
        "--runner",
        "trl",
        "--trainer-factory",
        FACTORY,
        "--allow-revision-change",
    ]
    candidates = (
        (
            "epsilon",
            "advantages = advantages / (std_rewards + 1e-4)",
            "advantages = advantages / (std_rewards + 1e-2)",
            2,
            "SUSPECTED_MISMATCH",
        ),
        (
            "mask",
            '"completion_mask": completion_mask,',
            '"completion_mask": torch.zeros_like(completion_mask),',
            1,
            "CONTRACT_VIOLATION",
        ),
    )
    for label, old, new, exitcode, verdict in candidates:
        candidate_root = output / f"candidate-{label}"
        candidate_root.mkdir()
        owner = candidate_copy(candidate_root, "trl") / "trainer/grpo_trainer.py"
        replace_once(owner, old, new)
        changed = run(
            output,
            f"{label}-replay",
            replay,
            expected=exitcode,
            candidate=candidate_root,
        )
        finding = json.loads(changed.stdout)
        assert finding["finding"]["verdict"] == verdict
        assert not finding["native_observation_equal"]
        if label == "mask":
            assert finding["finding"]["reason"] == "trl_frozen_generation_layout"
            assert "completion_mask" in finding["layout_differences"]
        failed = run(
            output, f"{label}-regression", pytest, expected=1, candidate=candidate_root
        )
        assert verdict in failed.stdout and "1 failed" in failed.stdout
    pristine = run(output, "pristine-regression", pytest)
    assert "1 passed" in pristine.stdout
    assert digest(bundle) == before
    return {
        "healthy": "PASS",
        "injected_epsilon": "SUSPECTED_MISMATCH",
        "injected_zero_mask": "CONTRACT_VIOLATION",
        "pristine_restored": "PASS",
        "original_bundle_sha256": before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("verl", "trl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-forward", type=Path)
    args = parser.parse_args()
    if args.no_forward:
        verify_no_forward(args.no_forward)
        return
    if not args.framework or not args.output:
        parser.error("--framework and --output are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = run_verl(output) if args.framework == "verl" else run_trl(output)
    result["framework"] = args.framework
    result["scope"] = (
        "Pinned real framework CPU methods; synthetic inputs; injected candidates are controls"
    )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
