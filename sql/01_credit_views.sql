-- ============================================================================
-- FinTech Credit Risk Scorecard — SQLite Banking Schema, QA & Risk Views
-- Dataset: UCI Taiwan Credit Card Default (30,000 real banking clients)
-- Author: Mostafa Saad
-- ============================================================================

DROP VIEW IF EXISTS vw_credit_qa;
CREATE VIEW vw_credit_qa AS
SELECT
    COUNT(*) AS total_clients,
    SUM(CASE WHEN CAST(x3 AS INTEGER) IN (0, 5, 6) THEN 1 ELSE 0 END) AS undocumented_education_codes,
    SUM(CASE WHEN CAST(x4 AS INTEGER) = 0 THEN 1 ELSE 0 END) AS undocumented_marriage_codes,
    SUM(CASE WHEN CAST(x12 AS REAL) > CAST(x1 AS REAL) THEN 1 ELSE 0 END) AS over_limit_clients_m1,
    SUM(CASE WHEN CAST(y AS INTEGER) = 1 THEN 1 ELSE 0 END) AS total_defaults,
    ROUND(100.0 * AVG(CAST(y AS REAL)), 2) AS default_rate_pct
FROM raw_credit_clients;

DROP VIEW IF EXISTS vw_credit_features;
CREATE VIEW vw_credit_features AS
SELECT
    rowid AS client_id,
    CAST(x1 AS REAL) AS LIMIT_BAL,
    CAST(x2 AS INTEGER) AS SEX,
    -- Map undocumented UCI education codes (0, 5, 6) -> 4 (Others)
    CASE WHEN CAST(x3 AS INTEGER) IN (1, 2, 3) THEN CAST(x3 AS INTEGER) ELSE 4 END AS EDUCATION,
    -- Map undocumented UCI marriage code (0) -> 3 (Others)
    CASE WHEN CAST(x4 AS INTEGER) IN (1, 2) THEN CAST(x4 AS INTEGER) ELSE 3 END AS MARRIAGE,
    CAST(x5 AS INTEGER) AS AGE,
    CASE
        WHEN CAST(x5 AS INTEGER) < 30 THEN '21-29'
        WHEN CAST(x5 AS INTEGER) < 40 THEN '30-39'
        WHEN CAST(x5 AS INTEGER) < 50 THEN '40-49'
        ELSE '50+'
    END AS age_band,
    CAST(x6 AS INTEGER) AS PAY_0,
    CAST(x7 AS INTEGER) AS PAY_2,
    CAST(x8 AS INTEGER) AS PAY_3,
    CAST(x9 AS INTEGER) AS PAY_4,
    CAST(x10 AS INTEGER) AS PAY_5,
    CAST(x11 AS INTEGER) AS PAY_6,
    CAST(x12 AS REAL) AS BILL_AMT1,
    CAST(x13 AS REAL) AS BILL_AMT2,
    CAST(x14 AS REAL) AS BILL_AMT3,
    CAST(x15 AS REAL) AS BILL_AMT4,
    CAST(x16 AS REAL) AS BILL_AMT5,
    CAST(x17 AS REAL) AS BILL_AMT6,
    CAST(x18 AS REAL) AS PAY_AMT1,
    CAST(x19 AS REAL) AS PAY_AMT2,
    CAST(x20 AS REAL) AS PAY_AMT3,
    CAST(x21 AS REAL) AS PAY_AMT4,
    CAST(x22 AS REAL) AS PAY_AMT5,
    CAST(x23 AS REAL) AS PAY_AMT6,
    -- Engineered Credit Risk Ratios
    ROUND(CAST(x12 AS REAL) / NULLIF(CAST(x1 AS REAL), 0), 4) AS util_m1,
    ROUND((CAST(x12 AS REAL)+CAST(x13 AS REAL)+CAST(x14 AS REAL)+CAST(x15 AS REAL)+CAST(x16 AS REAL)+CAST(x17 AS REAL)) / (6.0 * NULLIF(CAST(x1 AS REAL), 0)), 4) AS avg_util_6m,
    (
        (CASE WHEN CAST(x6 AS INTEGER) >= 1 THEN 1 ELSE 0 END) +
        (CASE WHEN CAST(x7 AS INTEGER) >= 1 THEN 1 ELSE 0 END) +
        (CASE WHEN CAST(x8 AS INTEGER) >= 1 THEN 1 ELSE 0 END) +
        (CASE WHEN CAST(x9 AS INTEGER) >= 1 THEN 1 ELSE 0 END) +
        (CASE WHEN CAST(x10 AS INTEGER) >= 1 THEN 1 ELSE 0 END) +
        (CASE WHEN CAST(x11 AS INTEGER) >= 1 THEN 1 ELSE 0 END)
    ) AS months_delinquent_6m,
    MAX(CAST(x6 AS INTEGER), CAST(x7 AS INTEGER), CAST(x8 AS INTEGER), CAST(x9 AS INTEGER), CAST(x10 AS INTEGER), CAST(x11 AS INTEGER)) AS max_delay_6m,
    (CAST(x6 AS INTEGER) - CAST(x11 AS INTEGER)) AS delay_trend_6m,
    CASE WHEN CAST(x12 AS REAL) > 0 THEN CAST(x12 AS REAL) ELSE 0.0 END AS exposure_at_default,
    CAST(y AS INTEGER) AS default_flag
FROM raw_credit_clients;
