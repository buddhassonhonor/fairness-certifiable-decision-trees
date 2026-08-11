# Fairness-Certifiable Decision Trees with Explicit SAT/UNSAT Guarantees

Public artifact repository for code, datasets, experiment scripts, and evaluation figures accompanying the research paper:

> **Fairness-Certifiable Decision Trees with Explicit SAT/UNSAT Guarantees**

## Repository Structure

- `experiment/`: Primary experiment runner (`run_fairness_workflow.py`), experiment results for Round 1 (benchmarking), Round 2 (scalability/ablation/stability), and Round 3 (robustness/case study), and run manifests.
- `figures/`: High-resolution evaluation plots generated across all three experiment rounds.
- `experiment_fairness.py`: Utility module for fairness metric evaluation and exact constraint checks.

## Reproducibility

To reproduce all three experimental rounds, run:

```bash
python experiment/run_fairness_workflow.py
```

Outputs will be saved in `experiment/round1/`, `experiment/round2/`, and `experiment/round3/`, and visual plots will be rendered into `figures/`.

## License

MIT License
