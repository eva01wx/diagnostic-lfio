# Diagnostic LFIO experiments

This directory contains the algorithm implementations, baseline adapters, tests,
input data, fixed protocols and experiment reports.

See the [repository README](../README.md) for installation, file verification and
record restoration. Run experiment scripts from this directory with the Python
environment activated. Detailed commands for frozen-record replay and fresh
training are in [REPRODUCE.md](docs/research/REPRODUCE.md).

- [Diagnostic allocation protocol](docs/research/MAJOR_REVISION_PROTOCOL_2026-10-15.md)
- [Rule generalization](docs/research/RULE_GENERALIZATION_2026-10-16.md)
- [Global feature forgetting](docs/research/FEATURE_FORGETTING_2026-10-16.md)
- [Contextual and cost-aware forgetting](docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md)

The bundled tests cover 54 cases after pinned NeuRules sources are restored;
the four generalization/forgetting modules cover 47 cases without that download.
