# Case Study (Round 3)

- Dataset: `case_study_extreme`
- Seed: `13`
- Fairness constraints: `DI >= 0.8`, `FPR gap <= 0.05`
- Utility floor: `min_accuracy >= 0.72`

## SAT/UNSAT Observation
- Depth 1 certifiable tree status: `UNSAT`
- Depth 2 certifiable tree status: `UNSAT`

## CART Rules
```text
|--- x0 <= 0.6162
|   |--- x0 <= -0.7439
|   |   |--- class: 0
|   |--- x0 >  -0.7439
|   |   |--- class: 0
|--- x0 >  0.6162
|   |--- x2 <= -0.5719
|   |   |--- class: 1
|   |--- x2 >  -0.5719
|   |   |--- class: 1

```

## Certifiable Tree Rules (Depth 2)
```text
UNSAT/TIMEOUT: no feasible tree under fairness + utility constraints.
```

## Metrics
```text
          method  accuracy  selection_rate_g0  selection_rate_g1       di   fpr_g0   fpr_g1  fpr_gap status
            cart  0.864815           0.023715           0.773519 0.030659 0.013453 0.347222 0.333769    SAT
cert_tree_depth1       NaN                NaN                NaN      NaN      NaN      NaN      NaN  UNSAT
cert_tree_depth2       NaN                NaN                NaN      NaN      NaN      NaN      NaN  UNSAT
```