# Security policy

RLReplay handles training observations and diagnostic artifacts. Supported
interfaces and runtime configurations are documented in the compatibility guide.
Report security concerns with the affected version or commit and the smallest
safe reproduction.

## Report privately

GitHub private vulnerability reporting is enabled for this repository. Use the
**Security** tab and **Report a vulnerability**, or open the private reporting
form. [Report privately](https://github.com/yu830/rlreplay/security/advisories/new)

If a private report option is unavailable, ask a maintainer to establish a private
contact method before sending vulnerability details. A public request should
contain only that request. Keep exploit payloads, sensitive capsules, credentials
and real training data out of public issues, pull requests and logs.

## Data and execution boundaries

Review capsules and logs before sharing them. Numeric rewards, identities and
source metadata may be sensitive even when raw text is absent. Use an appropriately
restricted output directory and an explicit retention policy.

Explicit TRL native replay bundles also contain prompt/completion text and token
IDs. They are separate from the default numeric capture path. A Trainer factory
is trusted local Python code selected explicitly by the caller; a bundle cannot
select a factory or executable entry point. Review these bundles and generated
regressions before sharing or running them.

Capsule checksums establish content consistency. Native replay additionally
depends on the recorded producer, implementation and runtime evidence. Run native
framework integrations and candidate regressions in an environment whose code and
dependencies you trust. The runtime guide describes the supported storage and
failure-handling contracts. [Runtime guide](docs/production.md)
