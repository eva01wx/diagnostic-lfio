# Identity-case implementation amendment

The first run completed 1,296 source programs and 14,834 projections under
`results/feature_forgetting_20261016`. Its independent audit rejected the unused
feature identity control: a source program with 18 conditions reached the
50,000-subtraction limit while unnecessarily recomputing its own double
complement. The returned lower bound was safely false and labeled conservative;
there was no demonstrated unsound assigned prediction. Nevertheless, this
identity has a direct proof and should not require that computation.

Before re-evaluating, add exactly this construction shortcut: if none of the
forgotten coordinates occurs in the normalized source, both projections equal
that normalized source. Add a regression test with both limits set to one.
The data, source programs, selection policies, endpoints, 50,000-node and
4,096-box limits are unchanged. This is not a threshold or accuracy retuning.

The initial core, runner, protocol, tests, manifest and audit failure are
archived under `docs/research/archive/feature_forgetting_initial_2026-10-16/`.
Its experiment records remain in place. The amended runner writes an entirely
separate `results/feature_forgetting_20261016_v2` and locks this amendment along
with the source/protocol. Recompute every projection and policy there, then
require the full independent audit. Report the identity optimization and any
remaining conservative failures explicitly; do not relabel initial outputs as
outputs of the amended implementation.
