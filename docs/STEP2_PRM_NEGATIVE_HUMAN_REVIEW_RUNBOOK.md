# Step 2: structured blind human review for the PRM negative canary

Approval: user-approved on 2026-08-27. This phase is offline only. It does not
authorize model/API calls, GPU inference, training-record creation, training
merge, or larger-scale generation.

## Frozen scoring protocol

- review all 24 cases without viewing validator outputs, candidate origin, or
  intended controlled-error position;
- `problem_status=ok`: assign trajectory label 0 or 1;
- positive trajectory: `first_error_step=null`;
- process-negative trajectory: `error_type=process` and `first_error_step` must
  be a valid zero-based step;
- answer-only negative: `error_type=answer_only` and `first_error_step=null`;
- `problem_status=ambiguous` or `bad_gold`: both label and first error are null;
- trajectory-label denominator: all human `problem_status=ok` cases;
- exact-first-error denominator: human `ok` process-negative cases;
- required accuracy: trajectory label at least 90%, exact first error at least 80%;
- candidate list policy: human ok+negative, validator strict process negative,
  and exact first-error agreement;
- a candidate list is metadata only and is never a training-data merge.

## 1. Synchronize and test

```bash
cd /root/autodl-tmp/MedTrace-R1
source /etc/network_turbo
source .venv/bin/activate

git status --short
git fetch origin \
  +refs/heads/step2_genCOTdata:refs/remotes/origin/step2_genCOTdata
git merge --ff-only origin/step2_genCOTdata
git rev-parse HEAD
git status --short

python -m unittest discover
```

## 2. Create the private structured template

Preview performs validation and writes nothing:

```bash
python -m data_pipeline.prepare_prm_negative_human_review \
  --config configs/cot/prm_negative_human_review_v2.yaml
```

Create the template exactly once:

```bash
python -m data_pipeline.prepare_prm_negative_human_review \
  --config configs/cot/prm_negative_human_review_v2.yaml \
  --prepare-template-24
```

Private inputs:

```text
results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/
human_review_blind.md

results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/
human_review_annotations_v2.jsonl
```

The JSONL template contains one metadata line and 24 case lines. It contains no
validator result, origin, intended mutation, question, or trajectory text.
Review each numbered case in `human_review_blind.md` and enter the decision in
the matching JSONL line. Do not change case order, keys, or schema version.

Metadata example:

```json
{"blinded_to_validator_outputs":true,"record_type":"review_metadata","review_completed_at_utc":"2026-08-27T20:00:00+08:00","reviewer_role":"independent medical reviewer","schema_version":"medtrace.prm-negative-human-annotation.v2"}
```

Valid case examples:

```json
{"case_number":1,"human_error_type":"process","human_first_error_step":2,"human_problem_status":"ok","human_trajectory_label":0,"notes":"","record_type":"case_annotation","schema_version":"medtrace.prm-negative-human-annotation.v2"}
{"case_number":2,"human_error_type":"answer_only","human_first_error_step":null,"human_problem_status":"ok","human_trajectory_label":0,"notes":"","record_type":"case_annotation","schema_version":"medtrace.prm-negative-human-annotation.v2"}
{"case_number":3,"human_error_type":null,"human_first_error_step":null,"human_problem_status":"ok","human_trajectory_label":1,"notes":"","record_type":"case_annotation","schema_version":"medtrace.prm-negative-human-annotation.v2"}
{"case_number":4,"human_error_type":null,"human_first_error_step":null,"human_problem_status":"ambiguous","human_trajectory_label":null,"notes":"reason kept private","record_type":"case_annotation","schema_version":"medtrace.prm-negative-human-annotation.v2"}
```

If the reviewer has already viewed either validator key, use another reviewer
who has not seen those files. Do not set the blind attestation to true unless it
is accurate.

## 3. Validate and lock the completed annotations

The preview validates all fields and writes nothing:

```bash
python -m data_pipeline.lock_prm_negative_human_review \
  --config configs/cot/prm_negative_human_review_v2.yaml
```

After it reports 24 contract-valid annotations and a true blind attestation,
write the immutable hash lock:

```bash
python -m data_pipeline.lock_prm_negative_human_review \
  --config configs/cot/prm_negative_human_review_v2.yaml \
  --lock-completed-review-24
```

Any later edit to the annotation file, source candidates, original validator
events, or recovery attempts invalidates the lock.

## 4. Score the locked review

Preview the aggregate scores without writing reports:

```bash
python -m data_pipeline.audit_prm_negative_human_review \
  --config configs/cot/prm_negative_human_review_v2.yaml
```

Write the aggregate report and, only when both quality gates pass, the private
candidate-only negative list:

```bash
python -m data_pipeline.audit_prm_negative_human_review \
  --config configs/cot/prm_negative_human_review_v2.yaml \
  --score-locked-review-24
```

The candidate file contains metadata and first-error indices, not SFT or PRM
training records. No merge is authorized.

## 5. Generate the recovered validator key after locking

Only after the annotation lock exists:

```bash
python -m data_pipeline.review_prm_negative_validator_recovery \
  --config configs/cot/prm_negative_validator_recovery_v1.yaml
```

This writes `human_review_key_recovered.md` without changing the blind review,
annotations, or lock. All private review files stay under ignored `results/`
paths and must not be committed or shared.
