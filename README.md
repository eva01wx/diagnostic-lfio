# Diagnostic LFIO

Source code and reproducibility artifacts for **Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting**.

Diagnostic LFIO combines diagnostic allocation for compact rule distillation with explicit rule generalization and certified feature forgetting.

Repository: [eva01wx/diagnostic-lfio](https://github.com/eva01wx/diagnostic-lfio).

```bash
git clone https://github.com/eva01wx/diagnostic-lfio.git
cd diagnostic-lfio
python tools/verify_release.py
```

For SSH, use `git@github.com:eva01wx/diagnostic-lfio.git`.

**Authors:** Xu Wang and Zhiqiang Chen. **Corresponding author:** Xu Wang
(xwang@idt.eitech.edu.cn).
**Co-author email:** Zhiqiang Chen (zqchen@idt.eitech.edu.cn).
**Scientific release:** `contextual-forgetting-2026-10-16`.
**Repository copy prepared:** 19 October 2026.

The selected submission manuscript is the **34-page CAS single-column review version**:
[paper/submission/Manuscript_CAS_single_column.pdf](paper/submission/Manuscript_CAS_single_column.pdf).
It uses `\documentclass[a4paper,fleqn,review]{cas-sc}` from the official CAS package
linked by the author, with numeric citations. A 19-page CAS double-column alternative
and matching flat source ZIPs are in [paper/submission/](paper/submission/).
The earlier 62-page elsarticle review layout is retained in `paper/elsarticle-template-num.pdf`.
This is a prepared manuscript for Knowledge-Based Systems; this repository
does not claim journal submission, acceptance or publication.

| Directory | Contents |
| --- | --- |
| `paper/` | Previous elsarticle layout, development supplement and editable highlights |
| `paper/submission/` | Selected CAS single-column manuscript, double-column alternative and matching flat source ZIPs |
| `paper/source/` | Flat sources of the preserved elsarticle layout |
| `paper/historical/` | Previous 19-page and 34-page CAS PDFs, for historical reference |
| `reproduction/` | Expanded code, focused tests, pinned requirements, protocols, input data and reports |
| `artifacts/` | Three original, byte-identical source/code/experiment-record ZIPs |
| `provenance/` | Original checksums/audits plus the copy validation record |
| `tools/` | Dependency-free checksum verification and record restoration helpers |

The implementation covers diagnostic allocation, explicit rule generalization,
and certified feature forgetting, including background difference constraints
and cost-aware selection. Later extensions reuse the same six-dataset cohort;
they are not additional independent datasets. Logical certificates refer to the
source surrogate and declared input domain, not teacher correctness or human
comprehension. See the [implementation and limits](reproduction/docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md).

## Verify the distributed files

From this directory, with Python 3.12:

```bash
python tools/verify_release.py
python tools/restore_records.py --check
```

The first command checks the copied files and every payload in the three original ZIPs and the two CAS source ZIPs.
The second checks the record archive and any existing destination files without
extracting it. Scientific source bytes and recorded results are preserved.

## Install and run the focused tests

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements-major-revision.txt
cd reproduction
python -m pytest -q tests/test_contextual_forgetting.py tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
```

This four-module command runs 47 focused tests without downloading baseline
sources. All five bundled test modules contain 54 tests; the additional
diagnostic module requires the pinned NeuRules files restored below. The
historical full-workspace result of 167 tests includes application code outside
this research bundle; it is not a claim that all 167 tests are distributed here.
Copy validation reuses the installed environment, rather than testing a clean
installation.

## Restore frozen records and replay the current report

Starting again from this directory:

```bash
python tools/restore_records.py
cd reproduction
python fetch_prospective_dependencies.py
python fetch_prospective_dependencies.py --verify-only
python -m pytest -q tests
python generate_contextual_forgetting.py --workers 12
```

The frozen record archive is about 64 MB compressed and 1.03 GB expanded.
The restoration helper checks all hashes, preserves the code checksum index,
skips identical files and refuses conflicting existing files. Restored records
are ignored by Git; the original archive remains the distributed snapshot.

FastSRS sources and their license are bundled. Pinned NeuRules sources are
fetched and hash-checked by `fetch_prospective_dependencies.py`; the fetch step
requires network access. Third-party licenses and provenance are retained.
Record replay is distinct from fresh model training. Fresh execution commands,
source locks and historical stages are described in the
[reproduction guide](reproduction/docs/research/REPRODUCE.md).
Use a separate working copy for new experiments or regenerated reports.

Funding, authorship contributions and AI assistance are disclosed in the paper.
Submission-preparation worksheets inside the preserved code snapshot are drafts,
not completed journal declarations. The repository URL is https://github.com/eva01wx/diagnostic-lfio; no DOI has been assigned.
