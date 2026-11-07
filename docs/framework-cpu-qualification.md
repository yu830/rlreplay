# Real framework CPU qualification

The manually dispatched `Framework CPU qualification` GitHub Actions workflow
uses separate Linux x86-64/Python 3.12 environments for pinned verl and TRL.
Each environment installs its complete version lock from
`.github/framework-requirements/`, the CPU-only PyTorch wheel, and a freshly
built RLReplay wheel. Core installation keeps its zero required dependencies.

To reproduce one job from a source checkout with uv 0.12.3 and a new directory:

```bash
framework=trl  # or verl
uv venv --python 3.12 ../rlreplay-framework-env
task_python=../rlreplay-framework-env/bin/python
uv pip sync --python "$task_python" \
  --index https://download.pytorch.org/whl/cpu \
  --default-index https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  ".github/framework-requirements/${framework}-cpu.txt"
uv build --wheel --out-dir ../rlreplay-framework-wheels
uv pip install --python "$task_python" --no-deps ../rlreplay-framework-wheels/rlreplay-0.1.0-py3-none-any.whl
uv pip check --python "$task_python"
"$task_python" .github/scripts/framework_qualification.py \
  --framework "$framework" --output ../rlreplay-framework-results
```

The verl qualification exercises the healthy driver, the known no-std mismatch,
and the same exported regression against a separate source-only candidate fix.
The TRL qualification exercises frozen-provider native replay, injected epsilon
and zero-mask controls, and the original regression again under pristine TRL.
The script checks verdicts and semantic failure reasons, verifies that the
original artifact digest is unchanged, and checks that qualified TRL replay does
not call model forward or the original reward callback. The injected TRL changes
are test controls, not claims of upstream defects.

These are CPU method-boundary tests using synthetic data and a tiny local model.
They do not download a model, start a Ray cluster, use a GPU, or qualify an entire
training system. Package installation requires access to the public Python and
PyTorch indexes and the two fixed GitHub revisions recorded in the locks.
