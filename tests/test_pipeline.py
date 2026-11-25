import json
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_credit_risk_pipeline():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    assert m["qa_summary"]["total_clients"] == 30000
    assert m["splits"]["train"] + m["splits"]["val"] + m["splits"]["test"] == 30000

    best = m["selected_model"]
    assert m["models"][best]["test_roc_auc"] >= 0.76
    assert m["models"][best]["test_ks"] >= 0.40
    assert m["top2_deciles_default_capture_pct"] >= 45.0

    # Verify undocumented education/marriage codes cleaned in SQL view
    conn = sqlite3.connect(ROOT / "data" / "credit_risk.db")
    df = pd.read_sql_query("SELECT EDUCATION, MARRIAGE FROM vw_credit_features", conn)
    conn.close()
    assert set(df["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(df["MARRIAGE"].unique()).issubset({1, 2, 3})

    for fig in [
        "01_credit_risk_eda_iv.png",
        "02_scorecard_roc_ks.png",
        "03_credit_shap_drivers.png",
        "04_expected_loss_deciles.png",
    ]:
        p = ROOT / "outputs" / "figures" / fig
        assert p.exists() and p.stat().st_size > 20_000, f"Missing figure {fig}"

    print("ALL FINTECH CREDIT RISK TESTS PASSED!")


if __name__ == "__main__":
    test_credit_risk_pipeline()
