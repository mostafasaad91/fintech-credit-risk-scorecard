"""
FinTech Credit Risk — Probability of Default (PD) Scorecard, KS, IV, SHAP & Expected Loss
Author: Mostafa Saad
Dataset: UCI Taiwan Credit Card Default (30,000 real bank clients)
"""

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "uci_credit_card_default_30k.csv"
DB_PATH = ROOT / "data" / "credit_risk.db"
SQL_PATH = ROOT / "sql" / "01_credit_views.sql"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG = "#000000"
CARD = "#0A0C12"
CYAN = "#00E5FF"
VIOLET = "#A78BFA"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#F43F5E"
TEXT = "#FFFFFF"
MUTED = "#94A3B8"
GRID = "#1E2433"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": "#262D3F",
    "axes.labelcolor": MUTED,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "DejaVu Sans",
})


def ingest_and_run_sql():
    df_raw = pd.read_csv(RAW_CSV)
    conn = sqlite3.connect(DB_PATH)
    df_raw.to_sql("raw_credit_clients", conn, if_exists="replace", index=False)
    conn.executescript(SQL_PATH.read_text(encoding="utf-8"))
    qa = pd.read_sql_query("SELECT * FROM vw_credit_qa", conn).iloc[0].to_dict()
    df = pd.read_sql_query("SELECT * FROM vw_credit_features", conn)
    conn.close()
    return df, qa


def compute_iv(series, target, bins=8):
    """Information Value (IV) for banking credit risk feature screening."""
    s = pd.Series(series)
    if s.nunique() > bins:
        binned = pd.qcut(s.rank(method="first"), q=bins, duplicates="drop")
    else:
        binned = s
    tbl = pd.crosstab(binned, target)
    if tbl.shape[1] < 2:
        return 0.0
    good = (tbl[0] + 0.5) / (tbl[0].sum() + 0.5 * len(tbl))
    bad = (tbl[1] + 0.5) / (tbl[1].sum() + 0.5 * len(tbl))
    woe = np.log(good / bad)
    return float(np.sum((good - bad) * woe))


def ks_statistic(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_vals = tpr - fpr
    idx = int(np.argmax(ks_vals))
    return float(ks_vals[idx]), float(thresholds[idx])


def pd_to_credit_score(pd_prob, base_score=650, pdo=50, base_odds=10.0):
    """Industry-standard Points-to-Double-Odds (PDO) credit scorecard mapping (300-850)."""
    p = np.clip(pd_prob, 1e-4, 1 - 1e-4)
    odds_good = (1.0 - p) / p
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return np.clip(np.round(offset + factor * np.log(odds_good)), 300, 850)


def bootstrap_auc_ci(y_true, y_prob, n_boot=400, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        scores.append(roc_auc_score(yt[idx], yp[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    print("[1/5] Ingesting 30,000 UCI Credit clients into SQLite & building banking risk views...")
    df, qa = ingest_and_run_sql()
    print("QA Summary:", qa)

    # Additional Python behavioral ratios
    total_bill_6m = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    total_pay_6m = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["pay_to_bill_6m"] = np.clip(np.where(total_bill_6m > 0, total_pay_6m / total_bill_6m, 1.0), 0.0, 2.5)
    df["max_util_6m"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].max(axis=1) / np.maximum(df["LIMIT_BAL"], 1.0)
    df["zero_pay_months"] = (df[[f"PAY_AMT{i}" for i in range(1, 7)]] == 0).sum(axis=1)
    df["log_limit_bal"] = np.log1p(df["LIMIT_BAL"])

    num_cols = [
        "log_limit_bal", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "util_m1", "avg_util_6m", "max_util_6m", "months_delinquent_6m",
        "max_delay_6m", "delay_trend_6m", "pay_to_bill_6m", "zero_pay_months",
        "BILL_AMT1", "PAY_AMT1", "PAY_AMT2",
    ]
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    # Compute Information Value (IV) strictly on Train split after splitting (zero leakage!)
    X = df[num_cols + cat_cols].copy()
    y = df["default_flag"].values
    meta = df[["client_id", "LIMIT_BAL", "exposure_at_default", "age_band", "EDUCATION"]].copy()

    X_trainval, X_test, y_trainval, y_test, meta_trainval, meta_test = train_test_split(
        X, y, meta, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_trainval, y_trainval, meta_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    print(f"Splits -> Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    iv_scores = sorted(
        [{"feature": col, "iv": round(compute_iv(X_train[col], y_train), 4)} for col in num_cols + cat_cols],
        key=lambda d: d["iv"],
        reverse=True,
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    models = {
        "Naive Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Scorecard": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=260, max_depth=9, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=260, learning_rate=0.035, num_leaves=20, max_depth=5,
            min_child_samples=50, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=260, learning_rate=0.035, max_depth=4, min_child_weight=15,
            subsample=0.85, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

    print("[2/5] Training & benchmarking 5 Credit PD models (5-Fold Stratified CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    val_probs, test_probs, pipes = {}, {}, {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        pv = pipe.predict_proba(X_val)[:, 1]
        pt = pipe.predict_proba(X_test)[:, 1]
        val_probs[name], test_probs[name], pipes[name] = pv, pt, pipe

        auc_t = roc_auc_score(y_test, pt)
        gini_t = 2.0 * auc_t - 1.0
        ks_t, ks_thr = ks_statistic(y_test, pt)
        pr_t = average_precision_score(y_test, pt)
        brier_t = brier_score_loss(y_test, pt)
        ci_l, ci_h = bootstrap_auc_ci(y_test, pt) if name != "Naive Baseline" else (0.5, 0.5)

        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "val_roc_auc": round(float(roc_auc_score(y_val, pv)), 4),
            "test_roc_auc": round(float(auc_t), 4),
            "test_roc_auc_ci95": [round(ci_l, 4), round(ci_h, 4)],
            "test_gini": round(float(gini_t), 4),
            "test_ks": round(float(ks_t), 4),
            "test_pr_auc": round(float(pr_t), 4),
            "test_brier": round(float(brier_t), 4),
        }
        print(f"  {name:20s} | Val AUC={results[name]['val_roc_auc']:.4f} | Test AUC={auc_t:.4f} | Gini={gini_t:.4f} | KS={ks_t:.4f} | Brier={brier_t:.4f}")

    best_name = max([k for k in results if k != "Naive Baseline"], key=lambda k: results[k]["val_roc_auc"])
    print(f"Selected Best Credit PD Model (on Val AUC): {best_name}")

    # Basel Expected Loss (EL = PD * LGD * EAD, LGD = 45%) & Credit Score Tiers on Test Set
    LGD = 0.45
    test_df = meta_test.copy()
    test_df["default_flag"] = y_test
    test_df["pd_prob"] = test_probs[best_name]
    test_df["credit_score"] = pd_to_credit_score(test_df["pd_prob"])
    test_df["expected_loss_ntd"] = test_df["pd_prob"] * LGD * test_df["exposure_at_default"]
    test_df["risk_tier"] = pd.cut(
        test_df["credit_score"],
        bins=[299, 540, 620, 700, 851],
        labels=["High-Risk (<540)", "Subprime (540-620)", "Near-Prime (620-700)", "Prime (700+)"],
    )
    test_df["risk_decile"] = pd.qcut(test_df["pd_prob"].rank(method="first"), 10, labels=list(range(1, 11)))

    tier_summary = (
        test_df.groupby("risk_tier", observed=False)
        .agg(
            clients=("client_id", "count"),
            actual_default_pct=("default_flag", lambda s: round(100.0 * s.mean(), 2)),
            avg_pd_pct=("pd_prob", lambda s: round(100.0 * s.mean(), 2)),
            total_ead_m=("exposure_at_default", lambda s: round(s.sum() / 1e6, 2)),
            expected_loss_m=("expected_loss_ntd", lambda s: round(s.sum() / 1e6, 2)),
        )
        .reset_index()
    )
    top2_deciles_capture = float(
        test_df[test_df["risk_decile"].isin([9, 10])]["default_flag"].sum() / test_df["default_flag"].sum() * 100.0
    )

    # Slice audit by age_band
    slice_audit = []
    for band, grp in test_df.groupby("age_band"):
        slice_audit.append({
            "age_band": str(band),
            "n": int(len(grp)),
            "actual_default_pct": round(100.0 * grp["default_flag"].mean(), 2),
            "roc_auc": round(float(roc_auc_score(grp["default_flag"], grp["pd_prob"])), 4),
        })

    # [3/5] SHAP Explainability
    print("[3/5] Computing SHAP feature attributions on 6,000 test clients...")
    best_pipe = pipes[best_name]
    prep_fit = best_pipe.named_steps["prep"]
    clf_fit = best_pipe.named_steps["clf"]
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in prep_fit.get_feature_names_out()]
    X_test_t = prep_fit.transform(X_test)
    explainer = shap.TreeExplainer(clf_fit)
    sv = explainer.shap_values(X_test_t)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:12]

    # Save scorecard portfolio output & metrics JSON
    test_df.sort_values("pd_prob", ascending=False).head(100).to_csv(OUT_DIR / "high_risk_watchlist.csv", index=False)
    payload = {
        "dataset": "UCI Taiwan Credit Card Default (30,000 Real Bank Clients)",
        "qa_summary": qa,
        "splits": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "top_iv_features": iv_scores[:10],
        "models": results,
        "selected_model": best_name,
        "top2_deciles_default_capture_pct": round(top2_deciles_capture, 1),
        "total_test_expected_loss_ntd_m": round(float(test_df["expected_loss_ntd"].sum() / 1e6), 2),
        "risk_tiers": tier_summary.to_dict(orient="records"),
        "slice_audit_age": slice_audit,
        "top_shap_features": [{"feature": clean_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in top_idx],
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # [4/5] 4 AMOLED Executive Figures
    print("[4/5] Rendering 4 AMOLED banking risk visualizations...")

    # Fig 1: Information Value (IV) + Delinquency Streak Default Rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Credit Risk Screening — Information Value (IV) & Delinquency Cliff", color=TEXT, fontsize=14.5, fontweight="bold", y=0.97)

    top_iv = iv_scores[:10][::-1]
    iv_names = [d["feature"] for d in top_iv]
    iv_vals = [d["iv"] for d in top_iv]
    iv_cols = [EMERALD if v >= 0.30 else (CYAN if v >= 0.10 else VIOLET) for v in iv_vals]
    ax1.barh(iv_names, iv_vals, color=iv_cols, height=0.62)
    ax1.axvline(0.30, color=EMERALD, ls="--", lw=1.1, alpha=0.8)
    ax1.axvline(0.10, color=CYAN, ls=":", lw=1.1, alpha=0.8)
    for i, v in enumerate(iv_vals):
        ax1.text(v + 0.015, i, f"{v:.3f}", va="center", color=TEXT, fontsize=8.5, family="monospace")
    ax1.set_xlim(0, max(iv_vals) * 1.2)
    ax1.set_xlabel("Information Value (IV · Train Split)")
    ax1.set_title("Top 10 Predictive Features (IV ≥ 0.30 Strong)", color=CYAN, fontweight="bold")
    ax1.grid(axis="x", ls=":", alpha=0.5)

    delinq_rates = df.groupby("months_delinquent_6m")["default_flag"].mean() * 100
    ax2.plot(delinq_rates.index, delinq_rates.values, marker="o", color=ROSE, lw=2.6, markersize=8, markerfacecolor=BG, markeredgewidth=2.2)
    ax2.fill_between(delinq_rates.index, delinq_rates.values, color=ROSE, alpha=0.14)
    for x_v, y_v in zip(delinq_rates.index, delinq_rates.values):
        ax2.text(x_v, y_v + 2.5, f"{y_v:.1f}%", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 88)
    ax2.set_xlabel("Months Delinquent in Past 6 Months")
    ax2.set_ylabel("Observed Default Rate (%)")
    ax2.set_title("Default Probability Escalation by Delinquency Count", color=VIOLET, fontweight="bold")
    ax2.grid(ls=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "01_credit_risk_eda_iv.png", facecolor=BG)
    plt.close(fig)

    # Fig 2: ROC + KS Separation + Calibration
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.9), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Credit PD Benchmark — ROC/Gini, KS Separation & Calibration ({best_name})", color=TEXT, fontsize=14, fontweight="bold", y=0.97)
    pal = {"Logistic Scorecard": CYAN, "Random Forest": VIOLET, "LightGBM": EMERALD, "XGBoost": AMBER}
    for m_n, col in pal.items():
        p = test_probs[m_n]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax1.plot(fpr, tpr, color=col, lw=2.0, label=f"{m_n} (AUC {results[m_n]['test_roc_auc']:.3f})")
        pt_c, pp_c = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax3.plot(pp_c, pt_c, marker="o", color=col, lw=1.8, markersize=4, label=f"{m_n} ({results[m_n]['test_brier']:.3f})")

    ax1.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax1.set_title("ROC Curve (Test n=6,000)", color=CYAN, fontweight="bold")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.grid(ls=":", alpha=0.5); ax1.legend(frameon=False, fontsize=7.8)

    # KS Curve for selected best model
    p_best = test_probs[best_name]
    fpr_b, tpr_b, thr_b = roc_curve(y_test, p_best)
    ks_val, ks_t = ks_statistic(y_test, p_best)
    order = np.argsort(thr_b)
    ax2.plot(thr_b[order], tpr_b[order], color=ROSE, lw=2.2, label="Cumulative Defaults (Bad)")
    ax2.plot(thr_b[order], fpr_b[order], color=EMERALD, lw=2.2, label="Cumulative Non-Defaults (Good)")
    ax2.axvline(ks_t, color=CYAN, ls="--", lw=1.6, label=f"Max KS = {ks_val:.3f} @ {ks_t:.2f}")
    ax2.set_xlim(0, 0.95)
    ax2.set_title("Kolmogorov-Smirnov (KS) Separation", color=VIOLET, fontweight="bold")
    ax2.set_xlabel("Predicted Default Prob (PD)"); ax2.set_ylabel("Cumulative Rate")
    ax2.grid(ls=":", alpha=0.5); ax2.legend(frameon=False, fontsize=7.8)

    ax3.plot([0, 1], [0, 1], color="#FFFFFF", ls="--", lw=1, alpha=0.6)
    ax3.set_title("PD Calibration Reliability (Brier)", color=EMERALD, fontweight="bold")
    ax3.set_xlabel("Mean Predicted PD"); ax3.set_ylabel("Observed Default Rate")
    ax3.grid(ls=":", alpha=0.5); ax3.legend(frameon=False, fontsize=7.8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "02_scorecard_roc_ks.png", facecolor=BG)
    plt.close(fig)

    # Fig 3: SHAP Credit Risk Drivers
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    fig.patch.set_facecolor(BG)
    tf = [clean_names[i] for i in top_idx][::-1]
    tv = [mean_abs[i] for i in top_idx][::-1]
    cols_s = []
    for i in top_idx[::-1]:
        corr = np.corrcoef(X_test_t[:, i], sv[:, i])[0, 1]
        cols_s.append(ROSE if (not np.isnan(corr) and corr > 0) else EMERALD)
    ax.barh(tf, tv, color=cols_s, height=0.62, alpha=0.9)
    for idx_b, v in enumerate(tv):
        ax.text(v + 0.008, idx_b, f"{v:.3f}", va="center", color=TEXT, fontsize=8.6, family="monospace")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Default Log-Odds)")
    ax.set_title(f"Top 12 Credit Default Drivers ({best_name} · SHAP Explainability)", color=TEXT, fontsize=13.5, fontweight="bold", pad=12)
    ax.text(0.98, 0.06, "■ Red = Higher Default Risk   ■ Green = Protective Factor",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.8, family="monospace")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_xlim(0, max(tv) * 1.18)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_credit_shap_drivers.png", facecolor=BG)
    plt.close(fig)

    # Fig 4: Decile Default Capture & Basel Expected Loss by Credit Score Tier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.1), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Portfolio Risk Concentration — Decile Capture & Basel Expected Loss (LGD=45%)", color=TEXT, fontsize=14, fontweight="bold", y=0.97)

    dec_rates = test_df.groupby("risk_decile", observed=False)["default_flag"].mean() * 100
    d_cols = [EMERALD if i < 5 else (AMBER if i < 8 else ROSE) for i in range(10)]
    ax1.bar([f"D{i}" for i in range(1, 11)], dec_rates.values, color=d_cols, width=0.65)
    for i, v in enumerate(dec_rates.values):
        ax1.text(i, v + 1.5, f"{v:.0f}%", ha="center", color=TEXT, fontsize=8.2, family="monospace")
    ax1.set_ylim(0, max(dec_rates.values) * 1.18)
    ax1.set_ylabel("Observed Default Rate (%)")
    ax1.set_title(f"Risk Deciles (Top 2 Deciles Capture {top2_deciles_capture:.1f}% of Defaults)", color=CYAN, fontweight="bold")
    ax1.grid(axis="y", ls=":", alpha=0.5)

    t_labels = [str(r["risk_tier"]) for _, r in tier_summary.iterrows()]
    el_vals = [r["expected_loss_m"] for _, r in tier_summary.iterrows()]
    t_cols = [ROSE, AMBER, CYAN, EMERALD]
    ax2.bar(t_labels, el_vals, color=t_cols, width=0.56)
    for i, v in enumerate(el_vals):
        ax2.text(i, v + max(el_vals) * 0.03, f"{v:.1f}M", ha="center", color=TEXT, fontsize=9, fontweight="bold", family="monospace")
    ax2.set_ylim(0, max(el_vals) * 1.2)
    ax2.set_ylabel("Expected Credit Loss (NT$ Millions)")
    ax2.set_title("Expected Loss (EL = PD × 45% × EAD) by Score Tier", color=VIOLET, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=8.3)
    ax2.grid(axis="y", ls=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "04_expected_loss_deciles.png", facecolor=BG)
    plt.close(fig)

    print("[5/5] Credit Risk Scorecard pipeline complete!")


if __name__ == "__main__":
    main()
