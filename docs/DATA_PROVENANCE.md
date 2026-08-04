# Benchmark provenance

## Content verification

On 2026-08-04, `evaluation/data/eval_data.json` was compared row by row with
the following pinned Hugging Face repositories:

| Benchmark | Repository | Revision | Split | Exact rows |
|---|---|---|---|---:|
| MedQA | `openlifescienceai/MedQA-USMLE-4-options-hf` | `20a8f4d6b851f6391751f6e76c06bc3a26c83e0b` | test | 1273/1273 |
| MedMCQA | `openlifescienceai/medmcqa` | `91c6572c454088bf71b679ad90aa8dffcd0d5868` | validation | 4183/4183 |

Question text, all four option texts, and answer labels matched exactly. The
machine-readable report is `data/benchmark/provenance.json`; the verification
can be repeated with:

```bash
python data_pipeline/verify_benchmark_provenance.py
```

The verifier first checks that each remote repository is still at the pinned
revision. It then retrieves all official rows and fails on any row-count,
question, option, or answer difference.

## Licensing status

Content provenance and permission to redistribute are separate questions.
The current status is intentionally recorded as
`content_verified_license_review_required`:

- The [MedQA Hugging Face card](https://huggingface.co/datasets/openlifescienceai/MedQA-USMLE-4-options-hf)
  does not declare a license. The
  [original MedQA repository](https://github.com/jind11/MedQA) displays an MIT
  license, but the underlying examination-question rights still warrant manual
  review before redistribution or commercial use.
- The [MedMCQA Hugging Face card](https://huggingface.co/datasets/openlifescienceai/medmcqa)
  declares Apache-2.0, while the
  [original MedMCQA repository](https://github.com/medmcqa/medmcqa) displays an
  MIT license. This metadata conflict must be resolved before redistributing
  the dataset.

MEDTRACE-R1 therefore tracks hashes and transformation scripts but does not
claim that all benchmark redistribution rights have been fully cleared.
