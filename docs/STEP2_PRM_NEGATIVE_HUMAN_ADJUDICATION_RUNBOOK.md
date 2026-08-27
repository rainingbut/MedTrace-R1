# Step 2: post-lock PRM first-error adjudication

Approval: user-approved on 2026-08-27. This is an offline disagreement
adjudication. It preserves the locked blind review and its failed 9/12 raw
exact-first-error score. It does not authorize model/API calls, GPU inference,
training-record creation, training merge, or larger-scale generation.

## Frozen decision

The blind review achieved 19/19 trajectory-label agreement but only 9/12 exact
first-error agreement. Exactly three process-negative cases require post-lock
adjudication. The adjudicator has seen validator first-error indices, so this
phase is explicitly unblinded and must never be reported as a replacement blind
score.

The adjudication protocol is `earliest_unambiguously_incorrect_step`. Every
decision must choose either the original human index or the validator index and
include a rationale. The original annotations, annotation lock, recovered key,
and failed aggregate audit are hash-bound into a new adjudication lock.

## 1. Synchronize and test

```bash
cd /root/autodl-tmp/MedTrace-R1
source /etc/network_turbo
source .venv/bin/activate

git fetch origin \
  +refs/heads/step2_genCOTdata:refs/remotes/origin/step2_genCOTdata
git merge --ff-only origin/step2_genCOTdata
python -m unittest discover
```

## 2. Prepare the private three-case template

Preview:

```bash
python -m data_pipeline.prepare_prm_negative_human_adjudication \
  --config configs/cot/prm_negative_human_adjudication_v1.yaml
```

Write the template once:

```bash
python -m data_pipeline.prepare_prm_negative_human_adjudication \
  --config configs/cot/prm_negative_human_adjudication_v1.yaml \
  --prepare-template-3
```

Complete the private file at:

```text
results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/
human_review_adjudication_v1.jsonl
```

Do not modify the original annotations, original lock, recovered validator key,
or raw audit. The completed approved private file may instead be uploaded to
this exact path after the preview confirms three disagreements.

## 3. Validate and lock

```bash
python -m data_pipeline.lock_prm_negative_human_adjudication \
  --config configs/cot/prm_negative_human_adjudication_v1.yaml

python -m data_pipeline.lock_prm_negative_human_adjudication \
  --config configs/cot/prm_negative_human_adjudication_v1.yaml \
  --lock-completed-adjudication-3
```

The first command writes nothing. The second writes an immutable lock only
after all three records pass the conditional contract.

## 4. Audit and write candidate metadata

```bash
python -m data_pipeline.audit_prm_negative_human_adjudication \
  --config configs/cot/prm_negative_human_adjudication_v1.yaml

python -m data_pipeline.audit_prm_negative_human_adjudication \
  --config configs/cot/prm_negative_human_adjudication_v1.yaml \
  --score-locked-adjudication-3
```

The report must show both the immutable raw 9/12 score and the separately named
adjudicated score. A passing gate may write only
`human_adjudicated_negative_candidates_v1.jsonl`, which contains candidate
metadata and first-error indices. It is not PRM training data and is not a
training merge.

All adjudication artifacts remain private under the Git-ignored `results/cot/`
tree.
