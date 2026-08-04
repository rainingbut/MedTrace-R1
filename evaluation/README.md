# Evaluation entry points

Use `run_eval.py` for all MEDTRACE-R1 baseline and ablation results. It uses the
fixed extraction policy in `answer_extractor.py` and the metrics in
`metrics.py`.

`eval.py` and `scorer.py` are retained unchanged from the HuatuoGPT-o1
snapshot for reference only. Their sampling and scoring behaviour is not an
accepted MEDTRACE-R1 evaluation protocol, and results produced by them must not
be reported as MEDTRACE-R1 baselines.
