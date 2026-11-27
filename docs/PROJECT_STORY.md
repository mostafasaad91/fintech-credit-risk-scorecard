# Project Story — FinTech Credit Risk — Basel III Scorecard & Expected Loss Engine
## قصة بناء المشروع: ليه عملناه، إزاي اتبنى، وإيه الأسئلة والتحديات اللي واجهتنا؟

---

## 1. Why Did We Build This Project? (ليه عملنا المشروع ده؟)

In banking, BNPL (Buy Now Pay Later), and FinTech lending, predicting **"Will this client default next month?"** is only half the job. A real Risk Committee needs three things that typical toy ML notebooks skip:
1. **Calibrated Probabilities of Default ($PD$)** — not just high AUC, because a miscalibrated model destroys capital reserves.
2. **Banking-Standard Scorecard Metrics** — **Information Value (IV)**, **Kolmogorov-Smirnov (KS) statistic**, **Gini**, and a readable **300–850 Credit Score**.
3. **Money-Weighted Risk Quantification** — **Basel Expected Loss** ($EL = PD \times LGD \times EAD$) so leadership knows exactly how much money is exposed in each score tier.

**بالعربي:**
في البنوك وشركات الـ FinTech والـ BNPL، مينفعش تقول للمدير "الموديل جاب Accuracy 82%". مدير المخاطر (Risk Manager) بيسأل 3 أسئلة مباشرة:
1. **إيه احتمال التعثر الحقيقي ($PD$) لكل عميل؟** وهل الاحتمالات دي متعايرة صح (Calibrated) ولا مبالغ فيها؟
2. **الموديل بيفصل بين العميل الكويس والعميل المتعثر قد إيه؟** وده بيتقاس بـ **KS** و **Gini** مش بالـ Accuracy.
3. **هنخسر كام فلوس بالظبط؟** وده اللي بنحسبه بمعادلة بازل: `Expected Loss = PD × LGD × EAD`.

---

## 2. How Was It Built? (إزاي اتبنى خطوة بخطوة؟)

### Stage 1 — Real Banking Dataset & SQLite Data Mart (`sql/01_credit_views.sql`)
- Downloaded **30,000 real Taiwanese credit card clients** (`UCI Credit Card Default` via OpenML `data_id=42477`, default rate **22.12%**).
- Loaded raw records into a relational **SQLite** database (`credit_mart.db`) and engineered behavioral risk views directly in SQL:
  - `months_delinquent_6m`: how many of the last 6 months had a payment delay (`PAY_i >= 1`).
  - `max_delay_6m`: worst delinquency severity across 6 months.
  - `utilization_m1` & `is_over_limit`: credit line stress (`BILL_AMT1 / LIMIT_BAL`).
  - `pay_to_bill_ratio_m1`: repayment coverage ratio (`PAY_AMT1 / BILL_AMT1`).
  - `bill_trend_6m`: balance expansion (`BILL_AMT1 - BILL_AMT6`).

### Stage 2 — Zero-Leakage Partitioning & Train-Only IV Audit (`src/run_pipeline.py`)
- Strict **60 / 20 / 20 stratified split**: **18,000 Train / 6,000 Validation / 6,000 Held-Out Test**.
- Computed **Weight of Evidence (WoE) & Information Value (IV)** strictly on the training split to audit predictive power before modeling:
  - `months_delinquent_6m` (**IV = 0.8879**)
  - `PAY_0` (**IV = 0.8226**)
  - `max_delay_6m` (**IV = 0.7459**)
  - `PAY_2` (**IV = 0.4891**)

### Stage 3 — Model Benchmark & Pre-Registered Selection
- Benchmarked **Majority Baseline**, **Balanced Logistic Regression Scorecard**, **LightGBM**, and **XGBoost** using 5-fold stratified CV + validation AUC.
- **LightGBM** won on Validation AUC (**0.7885** vs Logistic **0.7521**, XGBoost **0.7874**), achieving held-out **Test AUC = 0.7818** (`95% CI: [0.7678, 0.7973]`), **KS = 0.4293**, **Gini = 0.5637**, and **Brier = 0.1344**.

### Stage 4 — 300–850 Credit Score, Basel Expected Loss, Slice Audit & SHAP
- Converted calibrated $PD$ into a consumer **300–850 Credit Score** (`score = 600 - (40 / ln 2) * ln(odds * 20)`).
- Computed **Basel Expected Loss** (`LGD = 45%`, `EAD = max(BILL_AMT1, 0)`): **NT$29.62M** across the 6,000 held-out accounts, with **56% (NT$16.58M)** concentrated in the `<540` High-Risk tier.
- Audited age fairness across 4 bands (`21–29`, `30–39`, `40–49`, `50+`): stable AUC (`0.7707–0.7877`).
- Verified every contract automatically via `tests/test_pipeline.py`.

---

## 3. Key Questions & Engineering Challenges We Faced (إيه الأسئلة والتحديات اللي واجهتنا؟)

### Challenge 1 — Undocumented Real-World Data Codes (`EDUCATION = 0, 5, 6` & `MARRIAGE = 0`)
- **The Question:** The official UCI schema only documents `EDUCATION` in `{1, 2, 3, 4}` and `MARRIAGE` in `{1, 2, 3}`. Why did our SQL QA check find **345 rows** with `EDUCATION` in `{0, 5, 6}` and **54 rows** with `MARRIAGE = 0`?
- **Our Decision:** Never silently drop 399 real customers or leave ghost categories that leak noise into one-hot encoders. We normalized them inside `sql/01_credit_views.sql` into the documented `Other` buckets (`4` and `3`) and logged exact QA counts into `metrics.json`.

### Challenge 2 — Over-Limit Accounts (`BILL_AMT1 > LIMIT_BAL`) & Negative Bills
- **The Question:** We discovered **2,115 clients** whose current bill exceeded their credit limit (`utilization > 1.0`), plus accounts with negative bills (credit overpayment). How should we handle `EAD` (Exposure at Default) and `utilization`?
- **Our Decision:** Clamped `EAD = max(BILL_AMT1, 0)` (a bank cannot lose money on a negative credit balance) and preserved over-limit stress explicitly via `is_over_limit = 1`, which turned out to be a strong behavioral risk flag.

### Challenge 3 — Why Balanced Logistic Regression Had Worse Brier Score (`0.2007` vs `0.1344`)
- **The Question:** Why did Logistic Regression with `class_weight="balanced"` get decent AUC (`0.7452`) and KS (`0.3902`), yet poor Brier Score (`0.2007`) — worse than even predicting the base rate?
- **The Insight:** `class_weight="balanced"` artificially inflates predicted probabilities toward `0.50`! While rank ordering (AUC/KS) survives, **probability calibration breaks completely**, making raw balanced logistic outputs dangerous for Expected Loss pricing unless recalibrated (Platt/Isotonic). LightGBM without class-reweighting naturally produced crisp, well-calibrated PDs (**Brier = 0.1344**): in the High-Risk tier, Avg Predicted PD was **55.65%** vs Actual Default Rate **54.64%**!

### Challenge 4 — LightGBM vs XGBoost Photo-Finish (`0.7885` Val vs `0.7847` Test)
- **The Question:** LightGBM won on Validation (`0.7885` vs `0.7874`), while XGBoost scored slightly higher on Test (`0.7847` vs `0.7818`). Should we retroactively switch the "winner" to XGBoost to show a bigger headline AUC?
- **Our Decision:** **Absolutely not.** Switching models after peeking at the test set is textbook data leakage / p-hacking. Both sit well inside each other's 95% bootstrap confidence intervals (`[0.7678, 0.7973]` vs `[0.7704, 0.7990]`). Keeping LightGBM as the validation-selected model proves real scientific discipline.

### Challenge 5 — Reliable Programmatic Data Acquisition
- **The Issue:** The legacy UCI `.xls` direct URL was slow and unstandardized.
- **The Fix:** Fetched the exact official verified 30,000-row dataset cleanly via `sklearn.datasets.fetch_openml(data_id=42477)` and persisted a reproducible CSV snapshot under `data/raw/`.

---

## 4. What Would We Build Next in Production? (الخطوة الجاية لو ده سيستم حقيقي في بنك)
- **Out-of-Time (OOT) Backtesting:** Validate scorecard PSI (Population Stability Index) across macroeconomic shifts (inflation/rate hikes).
- **Reject Inference:** Account for previously declined applicants whose default labels are unobserved.
- **Fair Lending / Adverse Action Codes:** Automatically translate each rejected applicant's top 3 negative SHAP drivers into legally compliant adverse action reason codes.
