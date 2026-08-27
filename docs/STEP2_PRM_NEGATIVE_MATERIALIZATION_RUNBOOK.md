# Step 2: materialize approved PRM negative candidates

Approval: user-approved on 2026-08-27. This phase is offline only. It converts
the 11 locked and adjudicated negative candidates into isolated prefix-level
PRM records and audits their label balance. It does not authorize model/API
calls, GPU inference, SFT changes, training use, or larger-scale generation.

## Frozen behavior

- Read exactly 11 candidates from the passed adjudication output.
- Recompute and verify every upstream lock, audit, and candidate list.
- Use strict integer prefix labels: `1` before the agreed first error and `0`
  from the first error onward.
- Reject non-contiguous steps, duplicate records, duplicate full trajectories,
  overlap with the strict source, non-train sources, and Boolean labels.
- Preserve the original 721-record strict PRM view and 107-record SFT view.
- Write a candidate-only PRM file and a separate enriched derivative; never
  overwrite an original or strict-source artifact.

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

## 2. Read-only materialization preview

```bash
python -m data_pipeline.materialize_prm_negative_candidates \
  --config configs/cot/prm_negative_materialization_v1.yaml
```

The preview must report 11 candidate trajectories, both benchmarks, at least
two origins, zero duplicates, zero overlap with the strict source, no SFT
records written, and no model/API/GPU use. Prefix-record counts depend on the
real trajectory lengths and first-error positions.

## 3. Write the isolated derivative

```bash
python -m data_pipeline.materialize_prm_negative_candidates \
  --config configs/cot/prm_negative_materialization_v1.yaml \
  --materialize-approved-11
```

Private outputs:

```text
results/cot/pilot_v1_real/prm_negative_enrichment_v1/
prm_negative_materialization_v1/process_train_negative_candidates_v1.jsonl

results/cot/pilot_v1_real/prm_negative_enrichment_v1/
prm_negative_materialization_v1/process_train_negative_enriched_canary_v1.jsonl

results/cot/pilot_v1_real/prm_negative_enrichment_v1/
prm_negative_materialization_v1/manifest.json
```

The enriched derivative is audit-only at this stage and is not authorized as a
training input.

## 4. Audit label balance

Preview without writing an audit:

```bash
python -m data_pipeline.audit_prm_negative_materialization \
  --config configs/cot/prm_negative_materialization_v1.yaml
```

Write aggregate reports after all gates pass:

```bash
python -m data_pipeline.audit_prm_negative_materialization \
  --config configs/cot/prm_negative_materialization_v1.yaml \
  --audit-materialized-11
```

The report shows strict-source, newly materialized, and enriched label counts;
negative-prefix increase; benchmark/origin coverage; duplicates; and source
immutability. It contains no question, ID, or trajectory text.

Stop after the audit. Use the observed balance to design the next generation
quota. Do not train on the derivative or start larger-scale generation without
a separate user decision.
