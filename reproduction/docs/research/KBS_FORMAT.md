# Knowledge-Based Systems preparation — 19 October 2026

The author-selected submission entry is `elsarticle-template-num.tex`, based on
Elsevier's numbered-reference template, with these exact settings in the shared
manuscript:

```latex
\documentclass[review,12pt]{elsarticle}
\bibliographystyle{elsarticle-num}
```

The manuscript is single-column, 12pt, with the review option's expanded line
spacing. Natbib numeric sorting/compression is enabled through `\biboptions`.
The unchanged `elsarticle` 3.5 class (9 January 2026) and numbered bibliography
style come from the Elsevier-maintained CTAN distribution. Their origin and
license notices are retained in `docs/vendor/elsarticle/PROVENANCE.md`.

Current title: **Compact Rule Distillation with Diagnostic Allocation and
Certified Feature Forgetting**. Authors: Xu Wang and Zhiqiang Chen, in that order;
Xu Wang is corresponding author, and both share the three listed affiliations.
Funding remains Grant No. LQN26F020019, and both supplied CRediT statements and
the existing AI disclosure are retained.

| Output | Role |
| --- | --- |
| `elsarticle-template-num.pdf` | Preferred 62-page submission manuscript |
| `LFIO_Negation_Paper.pdf` | Same 62-page manuscript under its established filename |
| `LFIO_Negation_Paper_review.pdf` | Compatibility entry for the same review layout |
| `LFIO_Development_Supplement.pdf` | Separate 16-page historical development supplement |

```bash
venv/bin/python build_paper.py --all
venv/bin/python package_contextual_forgetting.py
```

In the workspace, the submission filename is a small wrapper around the shared
manuscript. In the flat source ZIP, `elsarticle-template-num.tex` contains the
complete main document and references only files at the same folder level.
Select that file as the main document in a LaTeX editor or submission system.
The equivalent compatibility filenames are not additional articles to submit.

The main front matter, keywords and floats now use `elsarticle` syntax.
The scientific body is preserved, with one inline quantity moved to display
math to fit the new review layout. Algorithm sources, data, numerical findings
and references are unchanged. Structural checks, visual review, independent
flat compilation and checksums are recorded in the current release audits.
No training or scientific report regeneration is needed for this formatting
change; earlier test reports retain their original dates and scope.

The scientific release remains `contextual-forgetting-2026-10-16`.
[CURRENT_RELEASE.md](CURRENT_RELEASE.md) lists current PDFs and packages.
The previous 19-page CAS double-column and 34-page CAS single-column versions,
old release packages and prior GitHub copy are preserved in
`archive/before_elsarticle_review_2026-10-19/`. The development supplement keeps
its original CAS template and separate title.

The format follows the author's instruction; it is not a claim that the live
journal guide has been independently verified. Journal-specific limits and
submission declarations remain subject to the actual submission system.
See [the checklist](SUBMISSION_CHECKLIST.md) and
[author-facts worksheet](AUTHOR_DECLARATIONS_DRAFT.md).
