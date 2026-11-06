# Diagram sources

The overview and detailed architecture diagrams are generated from Mermaid
sources. Light and dark resources carry the same graph. The normalization figure
reads the public synthetic framework fixture and plots recorded values against
the requested centered-reward contract.

Generate the diagrams with Mermaid CLI 12.2.0 from the repository root. Each
command can use the CLI's standard Puppeteer configuration option when a local
browser executable is required.

```bash
npx --yes @mermaid-js/mermaid-cli@12.2.0 -i docs/assets/overview.mmd -o docs/assets/overview.svg -c docs/assets/mermaid-light.json -b transparent
npx --yes @mermaid-js/mermaid-cli@12.2.0 -i docs/assets/overview.mmd -o docs/assets/overview-dark.svg -c docs/assets/mermaid-dark.json -b transparent
npx --yes @mermaid-js/mermaid-cli@12.2.0 -i docs/assets/architecture.mmd -o docs/assets/architecture.svg -c docs/assets/mermaid-light.json -b transparent
npx --yes @mermaid-js/mermaid-cli@12.2.0 -i docs/assets/architecture.mmd -o docs/assets/architecture-dark.svg -c docs/assets/mermaid-dark.json -b transparent
uv run --with matplotlib==3.11.2 python tools/render_case_figure.py
```

The numerical figure uses `tests/fixtures/public-cases.json`, case `verl-3`.
Expected values are computed directly as each reward minus its group mean,
matching the explicit no-std request. The case study describes the source and
native reproduction. [Case study](../case-studies/verl-normalization.md)
