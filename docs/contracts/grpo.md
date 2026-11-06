# Outcome-GRPO contract

The current contract checks group identity, response masks and token-view
advantages computed from one outcome reward per response. A capsule includes
observed output advantages and the producer's membership receipt, along with an
explicit semantic profile and source metadata.

For a group with at least two responses, the independent reference computes

```text
mean = sum(rewards) / group_size
std  = sqrt(sum((reward - mean)^2) / (group_size - 1))
advantage = (reward - mean) / (std + epsilon)
token_advantage = advantage * response_mask
```

With `normalize_std=False`, the scalar advantage is `reward - mean`. Constant
reward groups therefore produce zero advantages. The declared singleton policy
`mean0_std1` supplies mean 0 and standard deviation 1 for a one-member group.
The supported profiles use sample standard deviation (`ddof=1`) and epsilon
outside the square root.

| Profile setting | Qualified values |
|---|---|
| Profile ID | `grpo-outcome/sample-std-epsilon-outside/v1` |
| Epsilon | `1e-6` for the selected verl path, `1e-4` for standard TRL |
| Standard deviation correction | `ddof=1` |
| Singleton policy | `mean0_std1` |
| Numeric execution | Captured CPU float32 |
| Group completeness | Complete according to the supplied producer receipt |

Adapters further restrict their supported modes. The TRL integration requires at
least two generations, one finite reward function of weight 1, group scaling and
the remaining supported settings in its integration guide. Its native scalar
advantage is represented in Evidence as a derived `scalar * completion_mask`
view. [Adapters](../adapters.md)

## What is checked

- The captured sample identities are unique and match the producer receipt.
- Each captured sample belongs to the group recorded by that receipt.
- Masks contain binary values and align with the advantage shape.
- Observations are finite and within the supported float32 range.
- Padding positions have zero advantage.
- The captured advantages agree with the declared independent reference within
  the qualified numerical comparison.

The comparison uses `atol=1e-6` and `rtol=1e-5`. If the high-precision comparison
fails, a separately implemented sequential binary32 reference can explain some
rounding differences. This explanation returns INCONCLUSIVE, preserving the
remaining uncertainty about the execution path.

## Interpreting a finding

| Verdict | Meaning |
|---|---|
| PASS | Captured observations satisfy the supported contract and comparison |
| CONTRACT_VIOLATION | An observed structural or finite-value requirement is violated |
| SUSPECTED_MISMATCH | The independent numerical references do not explain the discrepancy |
| INCONCLUSIVE | Evidence or execution qualification is insufficient |

Interpret the verdict together with its reason, checked sample count and total
sample count. A source digest binds a local implementation record, while the
producer still owns the accuracy of identity and configuration observations.
Incomplete groups, missing source fields and unsupported profiles retain explicit
uncertainty. [Validation controls](../validation.md)
