# Security policy

RLReplay handles training observations and diagnostic artifacts. Its experimental
API, readers and local capture runtime are still being qualified. Report security
concerns with the affected version or commit and the smallest safe reproduction.

## Report privately

When GitHub private vulnerability reporting is enabled for this repository, use
the **Security** tab and **Report a vulnerability**. Availability of that channel
must be confirmed when the repository is created.

If a private report option is unavailable, ask a maintainer to establish a private
contact method before sending vulnerability details. A public request should
contain only that request. Keep exploit payloads, sensitive capsules, credentials
and real training data out of public issues, pull requests and logs.

## Data and execution boundaries

Review capsules and logs before sharing them. Numeric rewards, identities and
source metadata may be sensitive even when raw text is absent. Use an appropriately
restricted output directory and an explicit retention policy.

Capsule checksums establish content consistency. Native replay additionally
depends on the recorded producer, implementation and runtime evidence. Run native
framework integrations and candidate regressions in an environment whose code and
dependencies you trust. The runtime guide describes the supported storage and
failure-handling contracts. [Runtime guide](docs/production.md)
