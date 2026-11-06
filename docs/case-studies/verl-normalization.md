# A no-std request lost at the verl driver boundary

At verl revision `31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe`, the vectorized
GRPO driver omits `norm_adv_by_std_in_grpo` when dispatching to its helper. A
caller requesting centered rewards without standard-deviation normalization
therefore receives normalized advantages. The values remain finite and have
plausible signs; the explicitly requested profile exposes the parameter mismatch.

This case captures that original driver result, reproduces it through the native
entry point, and exports a regression that fails before a small local fix and
passes after it. Status: locally reproduced and candidate-verified; upstream
review and acceptance remain unverified.

## The declared contract and observation

The example uses CPU float32 outcome rewards, two complete groups, stable sample
IDs, and explicit membership recorded after `DataProto.repeat()` and before
`DataProto.reorder()`. It requests `grpo_vectorized` with
`norm_adv_by_std_in_grpo=False`.

| Group | Rewards | Expected centered advantages | Pinned driver result, approximately |
|---|---|---|---|
| a | 1, 3 | −1, +1 | −0.7071063, +0.7071063 |
| b | 2, 6 | −2, +2 | −0.7071065, +0.7071065 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/verl-normalization-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../assets/verl-normalization.svg">
  <img alt="Four samples in two groups: recorded advantages near minus or plus 0.707 compared with requested centered advantages minus or plus 1 and 2." src="../assets/verl-normalization.svg">
</picture>

The figure uses the public `verl-3` fixture in
[public-cases.json](../../tests/fixtures/public-cases.json). Its g1/g2 display
labels correspond to groups a/b above. Expected values are reward minus group
mean; recorded values come from each sample's first valid advantage token.

The token view multiplies these values by each response mask. The independent
Oracle reports `SUSPECTED_MISMATCH`, with maximum absolute error approximately
1.2928935. This verdict describes a discrepancy under the requested profile;
the driver/helper comparison and parameter-flow inspection establish its cause.

## Reproduce the capture

Use an environment with RLReplay and the pinned verl source plus compatible
dependencies installed. The example checks the qualified driver, algorithm and
groupwise owner fingerprints. Install the framework dependencies separately from
the lightweight Core. This fixture runs locally on preconstructed CPU tensors.

### Set up an isolated Linux CPU environment

From an RLReplay source checkout, the following commands create a separate
Python environment and fetch the exact public verl revision. They install the
dependencies needed to import this CPU driver; full training backends have
additional environment requirements.

```bash
uv venv --python 3.12 ../rlreplay-verl-env
task_python="../rlreplay-verl-env/bin/python"
uv pip install --python "$task_python" torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
git init ../verl-pinned
git -C ../verl-pinned fetch --depth=1 https://github.com/verl-project/verl.git 31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe
git -C ../verl-pinned checkout --detach FETCH_HEAD
uv pip install --python "$task_python" --default-index https://pypi.org/simple -e ../verl-pinned pytest cachetools uvicorn fastapi
uv pip install --python "$task_python" --default-index https://pypi.org/simple .
source ../rlreplay-verl-env/bin/activate
```

At this revision, the driver imports `cachetools`, `uvicorn` and `fastapi` through
its rollout modules even for a preconstructed CPU batch. The base verl package
does not declare all three, so they are explicit in the installation command.
The fixture does not start a Ray cluster, download a model or execute rollout.

This path was exercised in a fresh WSL/Linux environment with Python 3.12.3,
CPU-only PyTorch 2.13.0, NumPy 2.5.4, Transformers 5.16.1, Ray 2.60.0 and
TensorDict 0.10.0. The framework dependencies are resolved at installation time;
record the resolved versions with the results, and keep them fixed when testing
the source-only candidate below.

From the RLReplay repository root, use new output directories:

```bash
python examples/verl_observer.py demo-verl-healthy
python examples/verl_observer.py demo-verl-no-std --case known-no-std
```

The healthy scalar/std call returns exit 0 with one COMPLETE capsule and `PASS`.
The known vectorized/no-std call returns **exit 2 intentionally** and saves one
COMPLETE capsule with `SUSPECTED_MISMATCH`. Both perform semantic checking and
same-source native replay; the latter reproduces the captured values exactly.
`result.json` records the capsule path, requested profile, findings and coverage.
`receipts.json` preserves the worker's completion/check status.

Discover the single JSON capsule and run the ordinary CLI:

```bash
CAPSULE="$(python -c 'from pathlib import Path; paths = list(Path("demo-verl-no-std/capsules").glob("*.json")); assert len(paths) == 1, paths; print(paths[0])')"
python -m rlreplay check "$CAPSULE"
python -m rlreplay replay "$CAPSULE"
python -m rlreplay replay "$CAPSULE" --runner verl
python -m rlreplay export-test "$CAPSULE" regression-no-std
python -m pytest -q regression-no-std
```

The first three commands return exit 2. Default replay checks the saved observation
against the independent reference; `--runner verl` executes the installed driver
and checks its newly computed output. Reproducing the mismatch is the expected
result against the original driver. The export succeeds; the generated pytest
then fails against the original driver because it asserts that the replayed
candidate satisfies the declared contract. Preserve this failure as evidence.

## Explain and verify the local fix

The [scalar GRPO branch](https://github.com/verl-project/verl/blob/31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe/verl/trainer/ppo/ray_trainer.py#L235-L245)
forwards the flag. The [generic estimator dispatch](https://github.com/verl-project/verl/blob/31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe/verl/trainer/ppo/ray_trainer.py#L248-L282)
does not include it, and the
[vectorized helper](https://github.com/verl-project/verl/blob/31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe/verl/trainer/ppo/core_algos.py#L335-L358)
defaults to normalization. Direct helper and scalar no-std controls honor the
request in the local verification.

In a separate reviewed candidate copy of that exact verl revision, immediately
after the generic `adv_kwargs` dictionary is built, add only:

```python
if adv_estimator == AdvantageEstimator.GRPO_VECTORIZED:
    adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
```

Keep the original checkout intact. Point the isolated reproduction environment
at the separate, reviewed candidate checkout, then rerun the **same exported
regression against the original capsule**:

```bash
uv pip install --python "$task_python" --no-deps --reinstall -e ../verl-candidate
python -m pytest -q regression-no-std
```

Keep the Python, PyTorch and NumPy versions and the other framework owners fixed
when testing this source-only candidate change. Record the original capsule
digest and loaded candidate source digest with the result. A failure to import
the framework or initialize its dependencies is an environment failure, not the
expected semantic regression failure; check the assertion and finding in the
pytest output as well as the exit code.

The local candidate passes. The exported test explicitly enables revision
changes for candidate verification; ordinary native replay requires matching
recorded owner/runtime fingerprints unless its revision-change option is
explicitly selected. Candidate verification retains the original capsule unchanged.
The capture example itself continues to require pristine qualified owners.

For source identification, the original `ray_trainer.py` SHA256 is
`eb41f99ccc66976eab14ebe35aa1aa5b7318e02469240ef1569ac4c884c94312`.
The locally verified candidate's driver SHA256 is
`f4ada6e45e80351fe28c4c2a512eb12624b35a36f36d4fe025dcd10c649a4a17`.
These identify the original and locally reviewed candidate source files.

This case qualifies function-level CPU replay and regression for a fixed
outcome-GRPO configuration. Full training correctness, model quality, production
overhead and other framework revisions require their own validation. TRL shares
the Evidence/Oracle/semantic-replay path and additionally supports
[native-method replay](../trl-native-replay.md) through a separate explicit bundle
with frozen generation and reward providers.
