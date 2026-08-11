from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


DI_MIN = 0.80
FPR_GAP_MAX = 0.05


@dataclass
class DatasetBundle:
    name: str
    source: str
    X: np.ndarray
    y: np.ndarray
    s: np.ndarray
    feature_names: list[str]


@dataclass
class SolverConfig:
    depth: int = 2
    max_features: int = 6
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    min_accuracy: float = 0.65
    di_min: float = DI_MIN
    fpr_gap_max: float = FPR_GAP_MAX
    use_cache: bool = True
    rank_features: bool = True
    time_limit_sec: float = 60.0


@dataclass
class SplitCandidate:
    feature: int
    threshold: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    s = s.astype(int)

    acc = float(np.mean(y_true == y_pred))

    mask0 = s == 0
    mask1 = s == 1
    sel0 = _safe_mean(y_pred[mask0])
    sel1 = _safe_mean(y_pred[mask1])
    hi = max(sel0, sel1)
    lo = min(sel0, sel1)
    di = 1.0 if hi == 0 else float(lo / hi)

    neg0 = mask0 & (y_true == 0)
    neg1 = mask1 & (y_true == 0)
    fpr0 = _safe_mean(y_pred[neg0] == 1)
    fpr1 = _safe_mean(y_pred[neg1] == 1)
    fpr_gap = abs(fpr0 - fpr1)

    return {
        "accuracy": acc,
        "selection_rate_g0": sel0,
        "selection_rate_g1": sel1,
        "di": di,
        "fpr_g0": fpr0,
        "fpr_g1": fpr1,
        "fpr_gap": fpr_gap,
    }


def is_fair(metrics: dict[str, float], di_min: float, fpr_gap_max: float) -> bool:
    return metrics["di"] >= di_min and metrics["fpr_gap"] <= fpr_gap_max


def rank_features_by_signal(X: np.ndarray, y: np.ndarray) -> list[int]:
    y_center = y - np.mean(y)
    y_std = np.std(y_center)
    scores: list[tuple[float, int]] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        std = np.std(col)
        if std < 1e-12 or y_std < 1e-12:
            score = 0.0
        else:
            corr = np.corrcoef(col, y_center)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            score = abs(float(corr))
        scores.append((score, j))
    scores.sort(reverse=True)
    return [j for _, j in scores]


def generate_split_candidates(X: np.ndarray, y: np.ndarray, cfg: SolverConfig) -> list[SplitCandidate]:
    if cfg.rank_features:
        feature_order = rank_features_by_signal(X, y)
    else:
        feature_order = list(range(X.shape[1]))
    selected = feature_order[: min(cfg.max_features, X.shape[1])]

    splits: list[SplitCandidate] = []
    for j in selected:
        qvals = np.quantile(X[:, j], cfg.quantiles)
        uniq = np.unique(np.round(qvals, 8))
        for thr in uniq:
            splits.append(SplitCandidate(feature=j, threshold=float(thr)))
    return splits


def predict_tree_depth1(X: np.ndarray, root: SplitCandidate, leaves: tuple[int, int]) -> np.ndarray:
    m = X[:, root.feature] <= root.threshold
    return np.where(m, leaves[0], leaves[1]).astype(int)


def predict_tree_depth2(
    X: np.ndarray,
    root: SplitCandidate,
    left: SplitCandidate,
    right: SplitCandidate,
    leaves: tuple[int, int, int, int],
) -> np.ndarray:
    mr = X[:, root.feature] <= root.threshold
    ml = X[:, left.feature] <= left.threshold
    mrr = X[:, right.feature] <= right.threshold
    idx = np.where(mr, np.where(ml, 0, 1), np.where(mrr, 2, 3))
    table = np.array(leaves, dtype=int)
    return table[idx]


def solve_cert_tree(
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    cfg: SolverConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    splits = generate_split_candidates(X, y, cfg)
    if not splits:
        return {
            "status": "UNSAT",
            "runtime_sec": time.perf_counter() - start,
            "tree": None,
            "checked": 0,
            "feasible": 0,
            "num_splits": 0,
            "best_train_accuracy": np.nan,
            "best_train_di": np.nan,
            "best_train_fpr_gap": np.nan,
        }

    mask_cache: np.ndarray | None = None
    if cfg.use_cache:
        feats = np.array([sp.feature for sp in splits], dtype=int)
        thrs = np.array([sp.threshold for sp in splits], dtype=float)
        mask_cache = X[:, feats] <= thrs

    def get_mask(i: int) -> np.ndarray:
        if mask_cache is not None:
            return mask_cache[:, i]
        sp = splits[i]
        return X[:, sp.feature] <= sp.threshold

    checked = 0
    feasible = 0
    timeout = False

    best: dict[str, Any] | None = None
    leaf2 = list(itertools.product([0, 1], repeat=2))
    leaf4 = list(itertools.product([0, 1], repeat=4))

    def eval_candidate(y_pred: np.ndarray, tree: dict[str, Any]) -> None:
        nonlocal checked, feasible, best
        checked += 1
        met = fairness_metrics(y, y_pred, s)
        ok = (
            met["accuracy"] >= cfg.min_accuracy
            and met["di"] >= cfg.di_min
            and met["fpr_gap"] <= cfg.fpr_gap_max
        )
        if not ok:
            return
        feasible += 1
        if best is None or met["accuracy"] > best["train_metrics"]["accuracy"]:
            best = {"tree": tree, "train_metrics": met}

    if cfg.depth == 1:
        for i, root in enumerate(splits):
            if time.perf_counter() - start > cfg.time_limit_sec:
                timeout = True
                break
            m = get_mask(i)
            for leaves in leaf2:
                y_pred = np.where(m, leaves[0], leaves[1]).astype(int)
                tree = {"depth": 1, "root": i, "leaves": tuple(int(v) for v in leaves)}
                eval_candidate(y_pred, tree)
    elif cfg.depth == 2:
        for i, root in enumerate(splits):
            if time.perf_counter() - start > cfg.time_limit_sec:
                timeout = True
                break
            mr = get_mask(i)
            for j, left in enumerate(splits):
                if time.perf_counter() - start > cfg.time_limit_sec:
                    timeout = True
                    break
                ml = get_mask(j)
                for k, right in enumerate(splits):
                    if time.perf_counter() - start > cfg.time_limit_sec:
                        timeout = True
                        break
                    mrr = get_mask(k)
                    idx = np.where(mr, np.where(ml, 0, 1), np.where(mrr, 2, 3))
                    for leaves in leaf4:
                        table = np.array(leaves, dtype=int)
                        y_pred = table[idx]
                        tree = {
                            "depth": 2,
                            "root": i,
                            "left": j,
                            "right": k,
                            "leaves": tuple(int(v) for v in leaves),
                        }
                        eval_candidate(y_pred, tree)
                if timeout:
                    break
            if timeout:
                break
    else:
        raise ValueError("Only depth=1 or depth=2 is supported.")

    runtime = time.perf_counter() - start
    status = "SAT" if best is not None else ("TIMEOUT" if timeout else "UNSAT")
    if best is None:
        return {
            "status": status,
            "runtime_sec": runtime,
            "tree": None,
            "checked": checked,
            "feasible": feasible,
            "num_splits": len(splits),
            "best_train_accuracy": np.nan,
            "best_train_di": np.nan,
            "best_train_fpr_gap": np.nan,
            "splits": splits,
        }
    return {
        "status": status,
        "runtime_sec": runtime,
        "tree": best["tree"],
        "checked": checked,
        "feasible": feasible,
        "num_splits": len(splits),
        "best_train_accuracy": best["train_metrics"]["accuracy"],
        "best_train_di": best["train_metrics"]["di"],
        "best_train_fpr_gap": best["train_metrics"]["fpr_gap"],
        "splits": splits,
    }


def predict_cert_tree(X: np.ndarray, solve_out: dict[str, Any]) -> np.ndarray:
    tree = solve_out["tree"]
    if tree is None:
        raise ValueError("Cannot predict for UNSAT/TIMEOUT tree.")
    splits: list[SplitCandidate] = solve_out["splits"]
    if tree["depth"] == 1:
        return predict_tree_depth1(
            X,
            splits[tree["root"]],
            tuple(tree["leaves"]),
        )
    return predict_tree_depth2(
        X,
        splits[tree["root"]],
        splits[tree["left"]],
        splits[tree["right"]],
        tuple(tree["leaves"]),
    )


def describe_cert_tree(solve_out: dict[str, Any], feature_names: list[str]) -> str:
    tree = solve_out["tree"]
    if tree is None:
        return "UNSAT/TIMEOUT: no feasible tree under fairness + utility constraints."
    splits: list[SplitCandidate] = solve_out["splits"]
    if tree["depth"] == 1:
        root = splits[tree["root"]]
        return (
            f"if {feature_names[root.feature]} <= {root.threshold:.4f}: predict {tree['leaves'][0]}\n"
            f"else: predict {tree['leaves'][1]}"
        )
    root = splits[tree["root"]]
    left = splits[tree["left"]]
    right = splits[tree["right"]]
    leaves = tree["leaves"]
    return (
        f"if {feature_names[root.feature]} <= {root.threshold:.4f}:\n"
        f"  if {feature_names[left.feature]} <= {left.threshold:.4f}: predict {leaves[0]}\n"
        f"  else: predict {leaves[1]}\n"
        f"else:\n"
        f"  if {feature_names[right.feature]} <= {right.threshold:.4f}: predict {leaves[2]}\n"
        f"  else: predict {leaves[3]}"
    )


def fit_group_threshold_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    depth: int,
    seed: int,
    di_min: float,
    fpr_gap_max: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    clf.fit(X_train, y_train)
    p_train = clf.predict_proba(X_train)[:, 1]

    grid = np.linspace(0.05, 0.95, 19)
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None

    for t0 in grid:
        for t1 in grid:
            yhat = np.where(s_train == 0, p_train >= t0, p_train >= t1).astype(int)
            met = fairness_metrics(y_train, yhat, s_train)
            violation = max(0.0, di_min - met["di"]) + max(0.0, met["fpr_gap"] - fpr_gap_max)
            cand = {"t0": float(t0), "t1": float(t1), "metrics": met, "violation": float(violation)}
            if met["di"] >= di_min and met["fpr_gap"] <= fpr_gap_max:
                if best is None or met["accuracy"] > best["metrics"]["accuracy"]:
                    best = cand
            if fallback is None:
                fallback = cand
            else:
                if cand["violation"] < fallback["violation"] - 1e-12:
                    fallback = cand
                elif abs(cand["violation"] - fallback["violation"]) < 1e-12 and met["accuracy"] > fallback["metrics"]["accuracy"]:
                    fallback = cand

    chosen = best if best is not None else fallback
    if chosen is None:
        raise RuntimeError("Threshold search failed.")

    runtime = time.perf_counter() - start
    return {
        "clf": clf,
        "t0": chosen["t0"],
        "t1": chosen["t1"],
        "runtime_sec": runtime,
        "status": "SAT" if best is not None else "NEAR-SAT",
    }


def predict_group_threshold(model: dict[str, Any], X: np.ndarray, s: np.ndarray) -> np.ndarray:
    p = model["clf"].predict_proba(X)[:, 1]
    return np.where(s == 0, p >= model["t0"], p >= model["t1"]).astype(int)


def make_synthetic(
    name: str,
    n: int,
    p: int,
    bias_strength: float,
    label_noise: float,
    seed: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, size=n)
    X = rng.normal(0, 1, size=(n, p))

    shift = (2 * s - 1).astype(float)
    X[:, 0] += bias_strength * shift
    X[:, 1] += 0.6 * bias_strength * shift + rng.normal(0, 0.3, size=n)

    aux_idx = 2 if p >= 3 else (1 if p >= 2 else 0)
    logits = 1.25 * X[:, 0] + 0.75 * X[:, aux_idx] - 0.25 * shift + rng.normal(0, label_noise, size=n)
    y = (logits > np.quantile(logits, 0.55)).astype(int)

    flip_mask = rng.random(n) < 0.03
    y[flip_mask] = 1 - y[flip_mask]

    X_full = np.column_stack([X, s.astype(float)])
    fns = [f"x{j}" for j in range(p)] + ["sensitive_attr"]
    return DatasetBundle(name=name, source="synthetic", X=X_full, y=y, s=s.astype(int), feature_names=fns)


def load_proxy_datasets() -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []

    bc = load_breast_cancer()
    X = bc.data.astype(float)
    y = bc.target.astype(int)
    s = (X[:, 0] > np.median(X[:, 0])).astype(int)
    bundles.append(
        DatasetBundle(
            name="breast_cancer_proxy",
            source="real_proxy",
            X=np.column_stack([X, s.astype(float)]),
            y=y,
            s=s,
            feature_names=list(bc.feature_names) + ["sensitive_proxy"],
        )
    )

    wine = load_wine()
    Xw = wine.data.astype(float)
    yw = (wine.target > 0).astype(int)
    sw = (Xw[:, 0] > np.median(Xw[:, 0])).astype(int)
    bundles.append(
        DatasetBundle(
            name="wine_proxy",
            source="real_proxy",
            X=np.column_stack([Xw, sw.astype(float)]),
            y=yw,
            s=sw,
            feature_names=list(wine.feature_names) + ["sensitive_proxy"],
        )
    )

    dia = load_diabetes()
    Xd = dia.data.astype(float)
    yd = (dia.target > np.median(dia.target)).astype(int)
    sd = (Xd[:, 1] > np.median(Xd[:, 1])).astype(int)
    bundles.append(
        DatasetBundle(
            name="diabetes_proxy",
            source="real_proxy",
            X=np.column_stack([Xd, sd.astype(float)]),
            y=yd,
            s=sd,
            feature_names=[f"d{i}" for i in range(Xd.shape[1])] + ["sensitive_proxy"],
        )
    )
    return bundles


def build_round1_datasets(seed: int) -> list[DatasetBundle]:
    synth = [
        make_synthetic("synthetic_mild", n=1600, p=8, bias_strength=0.7, label_noise=0.9, seed=seed * 11 + 1),
        make_synthetic("synthetic_medium", n=1600, p=8, bias_strength=1.0, label_noise=0.8, seed=seed * 11 + 2),
        make_synthetic("synthetic_extreme", n=1600, p=8, bias_strength=1.4, label_noise=0.7, seed=seed * 11 + 3),
    ]
    return synth + load_proxy_datasets()

def run_methods_on_split(
    ds: DatasetBundle,
    seed: int,
    cert_cfg: SolverConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        ds.X,
        ds.y,
        ds.s,
        test_size=0.3,
        random_state=seed,
        stratify=ds.y,
    )

    cart_start = time.perf_counter()
    cart = DecisionTreeClassifier(max_depth=2, random_state=seed)
    cart.fit(X_train, y_train)
    cart_runtime = time.perf_counter() - cart_start
    cart_pred = cart.predict(X_test).astype(int)
    cart_met = fairness_metrics(y_test, cart_pred, s_test)
    rows.append(
        {
            "dataset": ds.name,
            "source": ds.source,
            "seed": seed,
            "method": "cart",
            "status": "SAT",
            "runtime_sec": cart_runtime,
            "checked": 0,
            "num_splits": 0,
            "min_accuracy_target": np.nan,
            **cart_met,
            "fair": is_fair(cart_met, DI_MIN, FPR_GAP_MAX),
        }
    )

    post = fit_group_threshold_tree(
        X_train=X_train,
        y_train=y_train,
        s_train=s_train,
        depth=2,
        seed=seed,
        di_min=DI_MIN,
        fpr_gap_max=FPR_GAP_MAX,
    )
    post_pred = predict_group_threshold(post, X_test, s_test)
    post_met = fairness_metrics(y_test, post_pred, s_test)
    rows.append(
        {
            "dataset": ds.name,
            "source": ds.source,
            "seed": seed,
            "method": "cart_group_threshold",
            "status": post["status"],
            "runtime_sec": post["runtime_sec"],
            "checked": 361,
            "num_splits": np.nan,
            "min_accuracy_target": np.nan,
            **post_met,
            "fair": is_fair(post_met, DI_MIN, FPR_GAP_MAX),
        }
    )

    cert = solve_cert_tree(X_train, y_train, s_train, cert_cfg)
    if cert["status"] == "SAT":
        cert_pred = predict_cert_tree(X_test, cert)
        cert_met = fairness_metrics(y_test, cert_pred, s_test)
        fair_flag = is_fair(cert_met, DI_MIN, FPR_GAP_MAX)
    else:
        cert_met = {
            "accuracy": np.nan,
            "selection_rate_g0": np.nan,
            "selection_rate_g1": np.nan,
            "di": np.nan,
            "fpr_g0": np.nan,
            "fpr_g1": np.nan,
            "fpr_gap": np.nan,
        }
        fair_flag = False
    rows.append(
        {
            "dataset": ds.name,
            "source": ds.source,
            "seed": seed,
            "method": "cert_tree",
            "status": cert["status"],
            "runtime_sec": cert["runtime_sec"],
            "checked": cert["checked"],
            "num_splits": cert["num_splits"],
            "min_accuracy_target": cert_cfg.min_accuracy,
            **cert_met,
            "fair": fair_flag,
        }
    )

    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "method"], dropna=False)
        .agg(
            runs=("seed", "count"),
            sat_rate=("status", lambda x: float(np.mean(np.asarray(x) == "SAT"))),
            fair_rate=("fair", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            di_mean=("di", "mean"),
            di_std=("di", "std"),
            fpr_gap_mean=("fpr_gap", "mean"),
            runtime_mean=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    return out


def plot_round1(summary: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    agg = (
        summary.groupby("method", dropna=False)
        .agg(
            accuracy=("accuracy_mean", "mean"),
            di=("di_mean", "mean"),
            fair_rate=("fair_rate", "mean"),
            sat_rate=("sat_rate", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(agg["method"], agg["accuracy"], color=["#3a7", "#5a9", "#c85"])
    axes[0].set_title("Round1 Mean Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(agg["method"], agg["fair_rate"], color=["#59a", "#4c8", "#d77"])
    axes[1].set_title("Round1 Fairness Satisfaction Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "round1_accuracy_fairness.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(agg["accuracy"], agg["di"], s=90)
    for _, r in agg.iterrows():
        ax2.text(r["accuracy"] + 0.002, r["di"] + 0.002, r["method"], fontsize=8)
    ax2.set_xlabel("Accuracy (mean)")
    ax2.set_ylabel("DI (mean)")
    ax2.set_title("Round1 Accuracy-Fairness Tradeoff")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "round1_tradeoff_scatter.png", dpi=200)
    plt.close(fig2)


def run_round1(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    out_dir = exp_root / "round1"
    ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []

    seeds = list(range(10))
    for seed in seeds:
        for ds in build_round1_datasets(seed):
            baseline = DecisionTreeClassifier(max_depth=2, random_state=seed)
            X_train, _, y_train, _, s_train, _ = train_test_split(
                ds.X, ds.y, ds.s, test_size=0.3, random_state=seed, stratify=ds.y
            )
            baseline.fit(X_train, y_train)
            train_acc = float(np.mean(baseline.predict(X_train) == y_train))
            maj = max(float(np.mean(y_train == 0)), float(np.mean(y_train == 1)))
            min_acc = max(0.58, min(0.90, max(maj + 0.05, train_acc - 0.10)))
            if ds.name == "synthetic_extreme":
                min_acc = min(0.92, min_acc + 0.06)
            cfg = SolverConfig(
                depth=2,
                max_features=6,
                quantiles=(0.25, 0.50, 0.75),
                min_accuracy=min_acc,
                di_min=DI_MIN,
                fpr_gap_max=FPR_GAP_MAX,
                use_cache=True,
                rank_features=True,
                time_limit_sec=60.0,
            )
            rows.extend(run_methods_on_split(ds, seed, cfg))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    summary = summarize_results(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_round1(summary, fig_root)

    return {
        "results_path": str(out_dir / "results.csv"),
        "summary_path": str(out_dir / "summary.csv"),
        "num_rows": int(len(df)),
    }


def run_scalability(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    out_dir = exp_root / "round2"
    ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []
    seeds = [0, 1, 2]

    for depth in [1, 2]:
        for k in [2, 4, 6, 8, 10]:
            for seed in seeds:
                ds = make_synthetic(
                    name=f"scale_k{k}",
                    n=900,
                    p=k,
                    bias_strength=1.0,
                    label_noise=0.8,
                    seed=5000 + 97 * seed + 13 * k + depth,
                )
                X_train, _, y_train, _, s_train, _ = train_test_split(
                    ds.X, ds.y, ds.s, test_size=0.3, random_state=seed, stratify=ds.y
                )
                cfg = SolverConfig(
                    depth=depth,
                    max_features=min(k + 1, ds.X.shape[1]),
                    quantiles=(0.25, 0.5, 0.75),
                    min_accuracy=0.62,
                    use_cache=True,
                    rank_features=True,
                    time_limit_sec=70.0,
                )
                sol = solve_cert_tree(X_train, y_train, s_train, cfg)
                rows.append(
                    {
                        "analysis": "scalability",
                        "depth": depth,
                        "feature_count": k,
                        "seed": seed,
                        "status": sol["status"],
                        "runtime_sec": sol["runtime_sec"],
                        "checked": sol["checked"],
                        "num_splits": sol["num_splits"],
                    }
                )

    scale_df = pd.DataFrame(rows)
    scale_df.to_csv(out_dir / "scalability.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for depth in [1, 2]:
        sub = scale_df[scale_df["depth"] == depth]
        grp = sub.groupby("feature_count")["runtime_sec"].mean().reset_index()
        ax.plot(grp["feature_count"], grp["runtime_sec"], marker="o", label=f"depth={depth}")
    ax.set_xlabel("Feature count (k)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Scalability: Runtime vs Feature Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_root / "round2_scalability_runtime.png", dpi=200)
    plt.close(fig)

    return {"path": str(out_dir / "scalability.csv"), "rows": int(len(scale_df))}

def run_ablation(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    out_dir = exp_root / "round2"
    ensure_dir(out_dir)

    variants = [
        ("full", dict(use_cache=True, rank_features=True, quantiles=(0.25, 0.50, 0.75))),
        ("no_cache", dict(use_cache=False, rank_features=True, quantiles=(0.25, 0.50, 0.75))),
        ("no_ordering", dict(use_cache=True, rank_features=False, quantiles=(0.25, 0.50, 0.75))),
    ]

    rows: list[dict[str, Any]] = []
    for seed in range(8):
        ds = make_synthetic(
            name="ablation_medium",
            n=1200,
            p=8,
            bias_strength=1.0,
            label_noise=0.75,
            seed=6000 + seed * 29,
        )
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            ds.X, ds.y, ds.s, test_size=0.3, random_state=seed, stratify=ds.y
        )
        for variant_name, opts in variants:
            cfg = SolverConfig(
                depth=2,
                max_features=6,
                quantiles=opts["quantiles"],
                min_accuracy=0.65,
                di_min=DI_MIN,
                fpr_gap_max=FPR_GAP_MAX,
                use_cache=opts["use_cache"],
                rank_features=opts["rank_features"],
                time_limit_sec=25.0,
            )
            sol = solve_cert_tree(X_train, y_train, s_train, cfg)
            if sol["status"] == "SAT":
                pred = predict_cert_tree(X_test, sol)
                met = fairness_metrics(y_test, pred, s_test)
                fair_flag = is_fair(met, DI_MIN, FPR_GAP_MAX)
            else:
                met = {
                    "accuracy": np.nan,
                    "selection_rate_g0": np.nan,
                    "selection_rate_g1": np.nan,
                    "di": np.nan,
                    "fpr_g0": np.nan,
                    "fpr_g1": np.nan,
                    "fpr_gap": np.nan,
                }
                fair_flag = False
            rows.append(
                {
                    "analysis": "ablation",
                    "variant": variant_name,
                    "seed": seed,
                    "status": sol["status"],
                    "runtime_sec": sol["runtime_sec"],
                    "checked": sol["checked"],
                    "num_splits": sol["num_splits"],
                    **met,
                    "fair": fair_flag,
                }
            )

    ab_df = pd.DataFrame(rows)
    ab_df.to_csv(out_dir / "ablation.csv", index=False)
    ab_sum = (
        ab_df.groupby("variant")
        .agg(runtime_mean=("runtime_sec", "mean"), fair_rate=("fair", "mean"), sat_rate=("status", lambda x: float(np.mean(np.asarray(x) == "SAT"))))
        .reset_index()
    )
    ab_sum.to_csv(out_dir / "ablation_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(ab_sum["variant"], ab_sum["runtime_mean"], color=["#4b9", "#c96", "#69b", "#a66"])
    axes[0].set_title("Ablation Runtime")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(ab_sum["variant"], ab_sum["fair_rate"], color=["#4b9", "#c96", "#69b", "#a66"])
    axes[1].set_title("Ablation Fair Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_root / "round2_ablation.png", dpi=200)
    plt.close(fig)

    return {"path": str(out_dir / "ablation.csv"), "rows": int(len(ab_df))}


def run_stability(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    out_dir = exp_root / "round2"
    ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []

    for seed in range(24):
        datasets = [
            make_synthetic("stability_synth", n=1400, p=8, bias_strength=1.0, label_noise=0.8, seed=7000 + 19 * seed),
            load_proxy_datasets()[0],
        ]
        for ds in datasets:
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                ds.X, ds.y, ds.s, test_size=0.3, random_state=seed, stratify=ds.y
            )

            cart = DecisionTreeClassifier(max_depth=2, random_state=seed)
            cart.fit(X_train, y_train)
            cart_pred = cart.predict(X_test).astype(int)
            cart_met = fairness_metrics(y_test, cart_pred, s_test)
            rows.append(
                {
                    "analysis": "stability",
                    "dataset": ds.name,
                    "seed": seed,
                    "method": "cart",
                    "status": "SAT",
                    **cart_met,
                    "fair": is_fair(cart_met, DI_MIN, FPR_GAP_MAX),
                }
            )

            post = fit_group_threshold_tree(X_train, y_train, s_train, depth=2, seed=seed, di_min=DI_MIN, fpr_gap_max=FPR_GAP_MAX)
            post_pred = predict_group_threshold(post, X_test, s_test)
            post_met = fairness_metrics(y_test, post_pred, s_test)
            rows.append(
                {
                    "analysis": "stability",
                    "dataset": ds.name,
                    "seed": seed,
                    "method": "cart_group_threshold",
                    "status": post["status"],
                    **post_met,
                    "fair": is_fair(post_met, DI_MIN, FPR_GAP_MAX),
                }
            )

            cfg = SolverConfig(
                depth=2,
                max_features=6,
                quantiles=(0.25, 0.50, 0.75),
                min_accuracy=0.64,
                time_limit_sec=25.0,
            )
            sol = solve_cert_tree(X_train, y_train, s_train, cfg)
            if sol["status"] == "SAT":
                pred = predict_cert_tree(X_test, sol)
                met = fairness_metrics(y_test, pred, s_test)
                fair_flag = is_fair(met, DI_MIN, FPR_GAP_MAX)
            else:
                met = {
                    "accuracy": np.nan,
                    "selection_rate_g0": np.nan,
                    "selection_rate_g1": np.nan,
                    "di": np.nan,
                    "fpr_g0": np.nan,
                    "fpr_g1": np.nan,
                    "fpr_gap": np.nan,
                }
                fair_flag = False
            rows.append(
                {
                    "analysis": "stability",
                    "dataset": ds.name,
                    "seed": seed,
                    "method": "cert_tree",
                    "status": sol["status"],
                    **met,
                    "fair": fair_flag,
                }
            )

    st_df = pd.DataFrame(rows)
    st_df.to_csv(out_dir / "stability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    methods = ["cart", "cart_group_threshold", "cert_tree"]
    acc_data = [st_df[st_df["method"] == m]["accuracy"].dropna().values for m in methods]
    di_data = [st_df[st_df["method"] == m]["di"].dropna().values for m in methods]
    axes[0].boxplot(acc_data, tick_labels=methods, showfliers=False)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Stability: Accuracy Distribution")
    axes[1].boxplot(di_data, tick_labels=methods, showfliers=False)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Stability: DI Distribution")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_root / "round2_stability_boxplot.png", dpi=200)
    plt.close(fig)

    return {"path": str(out_dir / "stability.csv"), "rows": int(len(st_df))}


def run_round2(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    ensure_dir(exp_root / "round2")
    out = {
        "scalability": run_scalability(exp_root, fig_root),
        "ablation": run_ablation(exp_root, fig_root),
        "stability": run_stability(exp_root, fig_root),
    }
    with (exp_root / "round2" / "round2_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out

def run_case_study(exp_root: Path) -> dict[str, Any]:
    out_dir = exp_root / "round3"
    ensure_dir(out_dir)

    seed = 13
    ds = make_synthetic("case_study_extreme", n=1800, p=8, bias_strength=1.5, label_noise=0.7, seed=8101)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        ds.X, ds.y, ds.s, test_size=0.3, random_state=seed, stratify=ds.y
    )

    cart = DecisionTreeClassifier(max_depth=2, random_state=seed)
    cart.fit(X_train, y_train)
    cart_pred = cart.predict(X_test).astype(int)
    cart_met = fairness_metrics(y_test, cart_pred, s_test)

    cfg_d1 = SolverConfig(depth=1, max_features=6, quantiles=(0.25, 0.5, 0.75), min_accuracy=0.72, time_limit_sec=50.0)
    cfg_d2 = SolverConfig(depth=2, max_features=6, quantiles=(0.25, 0.5, 0.75), min_accuracy=0.72, time_limit_sec=30.0)
    cert_d1 = solve_cert_tree(X_train, y_train, s_train, cfg_d1)
    cert_d2 = solve_cert_tree(X_train, y_train, s_train, cfg_d2)

    if cert_d2["status"] == "SAT":
        cert_pred = predict_cert_tree(X_test, cert_d2)
        cert_met = fairness_metrics(y_test, cert_pred, s_test)
    else:
        cert_met = {
            "accuracy": np.nan,
            "selection_rate_g0": np.nan,
            "selection_rate_g1": np.nan,
            "di": np.nan,
            "fpr_g0": np.nan,
            "fpr_g1": np.nan,
            "fpr_gap": np.nan,
        }

    metrics_df = pd.DataFrame(
        [
            {"method": "cart", **cart_met, "status": "SAT"},
            {
                "method": "cert_tree_depth1",
                **(
                    {**cert_met}
                    if cert_d1["status"] == "SAT"
                    else {
                        "accuracy": np.nan,
                        "selection_rate_g0": np.nan,
                        "selection_rate_g1": np.nan,
                        "di": np.nan,
                        "fpr_g0": np.nan,
                        "fpr_g1": np.nan,
                        "fpr_gap": np.nan,
                    }
                ),
                "status": cert_d1["status"],
            },
            {"method": "cert_tree_depth2", **cert_met, "status": cert_d2["status"]},
        ]
    )
    metrics_df.to_csv(out_dir / "case_study_metrics.csv", index=False)

    cart_rules = export_text(cart, feature_names=ds.feature_names, decimals=4)
    cert_rules = describe_cert_tree(cert_d2, ds.feature_names)
    text = [
        "# Case Study (Round 3)",
        "",
        f"- Dataset: `{ds.name}`",
        f"- Seed: `{seed}`",
        f"- Fairness constraints: `DI >= {DI_MIN}`, `FPR gap <= {FPR_GAP_MAX}`",
        f"- Utility floor: `min_accuracy >= {cfg_d2.min_accuracy}`",
        "",
        "## SAT/UNSAT Observation",
        f"- Depth 1 certifiable tree status: `{cert_d1['status']}`",
        f"- Depth 2 certifiable tree status: `{cert_d2['status']}`",
        "",
        "## CART Rules",
        "```text",
        cart_rules,
        "```",
        "",
        "## Certifiable Tree Rules (Depth 2)",
        "```text",
        cert_rules,
        "```",
        "",
        "## Metrics",
        "```text",
        metrics_df.to_string(index=False),
        "```",
    ]
    (out_dir / "case_study.md").write_text("\n".join(text), encoding="utf-8")

    return {"case_study_path": str(out_dir / "case_study.md")}


def run_noise_robustness(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    out_dir = exp_root / "round3"
    ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    for seed in range(10):
        base = make_synthetic("noise_medium", n=1500, p=8, bias_strength=1.0, label_noise=0.8, seed=9000 + 37 * seed)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            base.X, base.y, base.s, test_size=0.3, random_state=seed, stratify=base.y
        )
        cart = DecisionTreeClassifier(max_depth=2, random_state=seed)
        cart.fit(X_train, y_train)
        post = fit_group_threshold_tree(X_train, y_train, s_train, depth=2, seed=seed, di_min=DI_MIN, fpr_gap_max=FPR_GAP_MAX)
        cfg = SolverConfig(depth=2, max_features=6, quantiles=(0.25, 0.5, 0.75), min_accuracy=0.64, time_limit_sec=25.0)
        cert = solve_cert_tree(X_train, y_train, s_train, cfg)

        for sigma in noise_levels:
            rng = np.random.default_rng(10000 + 101 * seed + int(100 * sigma))
            Xn = X_test.copy()
            noise = rng.normal(0, sigma, size=Xn[:, :-1].shape)
            Xn[:, :-1] = Xn[:, :-1] + noise

            cart_pred = cart.predict(Xn).astype(int)
            cart_met = fairness_metrics(y_test, cart_pred, s_test)
            rows.append(
                {
                    "analysis": "noise",
                    "seed": seed,
                    "sigma": sigma,
                    "method": "cart",
                    "status": "SAT",
                    **cart_met,
                    "fair": is_fair(cart_met, DI_MIN, FPR_GAP_MAX),
                }
            )

            post_pred = predict_group_threshold(post, Xn, s_test)
            post_met = fairness_metrics(y_test, post_pred, s_test)
            rows.append(
                {
                    "analysis": "noise",
                    "seed": seed,
                    "sigma": sigma,
                    "method": "cart_group_threshold",
                    "status": post["status"],
                    **post_met,
                    "fair": is_fair(post_met, DI_MIN, FPR_GAP_MAX),
                }
            )

            if cert["status"] == "SAT":
                cert_pred = predict_cert_tree(Xn, cert)
                cert_met = fairness_metrics(y_test, cert_pred, s_test)
                rows.append(
                    {
                        "analysis": "noise",
                        "seed": seed,
                        "sigma": sigma,
                        "method": "cert_tree",
                        "status": "SAT",
                        **cert_met,
                        "fair": is_fair(cert_met, DI_MIN, FPR_GAP_MAX),
                    }
                )
            else:
                rows.append(
                    {
                        "analysis": "noise",
                        "seed": seed,
                        "sigma": sigma,
                        "method": "cert_tree",
                        "status": cert["status"],
                        "accuracy": np.nan,
                        "selection_rate_g0": np.nan,
                        "selection_rate_g1": np.nan,
                        "di": np.nan,
                        "fpr_g0": np.nan,
                        "fpr_g1": np.nan,
                        "fpr_gap": np.nan,
                        "fair": False,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "noise_robustness.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in ["cart", "cart_group_threshold", "cert_tree"]:
        sub = df[df["method"] == method]
        grp = sub.groupby("sigma")[["accuracy", "di"]].mean().reset_index()
        axes[0].plot(grp["sigma"], grp["accuracy"], marker="o", label=method)
        axes[1].plot(grp["sigma"], grp["di"], marker="o", label=method)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Noise Robustness: Accuracy")
    axes[0].set_xlabel("Gaussian noise sigma")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Noise Robustness: DI")
    axes[1].set_xlabel("Gaussian noise sigma")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_root / "round3_noise_robustness.png", dpi=200)
    plt.close(fig)

    return {"path": str(out_dir / "noise_robustness.csv"), "rows": int(len(df))}


def run_round3(exp_root: Path, fig_root: Path) -> dict[str, Any]:
    ensure_dir(exp_root / "round3")
    out = {
        "case_study": run_case_study(exp_root),
        "noise_robustness": run_noise_robustness(exp_root, fig_root),
    }
    with (exp_root / "round3" / "round3_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def build_q2_check(exp_root: Path) -> dict[str, Any]:
    r1 = pd.read_csv(exp_root / "round1" / "summary.csv")
    r2_scale = pd.read_csv(exp_root / "round2" / "scalability.csv")
    r2_ab = pd.read_csv(exp_root / "round2" / "ablation.csv")
    r2_st = pd.read_csv(exp_root / "round2" / "stability.csv")
    r3_case = exp_root / "round3" / "case_study.md"
    r3_noise = pd.read_csv(exp_root / "round3" / "noise_robustness.csv")

    checks = {
        "strong_baselines": int(r1["method"].nunique()) >= 3,
        "dataset_count_ge_5": int(r1["dataset"].nunique()) >= 5,
        "has_scalability_curve": not r2_scale.empty and int(r2_scale["depth"].nunique()) >= 2 and int(r2_scale["feature_count"].nunique()) >= 5,
        "has_ablation": not r2_ab.empty and int(r2_ab["variant"].nunique()) >= 3,
        "has_stability_distribution": not r2_st.empty and int(r2_st["seed"].nunique()) >= 20,
        "has_case_study": r3_case.exists(),
        "has_noise_robustness": not r3_noise.empty and int(r3_noise["sigma"].nunique()) >= 4,
    }

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness-certifiable tree experimental workflow.")
    parser.add_argument("--round", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--exp-root", default="experiment", help="Experiment output root directory.")
    parser.add_argument("--fig-root", default="figures", help="Figure output root directory.")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    fig_root = Path(args.fig_root)
    ensure_dir(exp_root)
    ensure_dir(fig_root)

    manifest: dict[str, Any] = {}

    if args.round in ("1", "all"):
        manifest["round1"] = run_round1(exp_root, fig_root)
    if args.round in ("2", "all"):
        manifest["round2"] = run_round2(exp_root, fig_root)
    if args.round in ("3", "all"):
        manifest["round3"] = run_round3(exp_root, fig_root)

    if args.round == "all":
        checks = build_q2_check(exp_root)
        manifest["q2_check"] = checks
        with (exp_root / "final_checklist.json").open("w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2)

    with (exp_root / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
