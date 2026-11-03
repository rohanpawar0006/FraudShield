-- Schema Initialization for FraudShield

CREATE TABLE IF NOT EXISTS transactions (
    txn_id VARCHAR(64) PRIMARY KEY,
    account_id VARCHAR(64) NOT NULL,
    dest_id VARCHAR(64) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    type VARCHAR(32) NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    raw JSONB
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    txn_id VARCHAR(64) REFERENCES transactions(txn_id) ON DELETE CASCADE,
    xgb_score DOUBLE PRECISION NOT NULL,
    gnn_score DOUBLE PRECISION,
    blended_score DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    latency_ms DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_blended_score ON scores (blended_score DESC);
CREATE INDEX IF NOT EXISTS idx_scores_scored_at ON scores (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_scores_txn_id ON scores (txn_id);

CREATE TABLE IF NOT EXISTS cases (
    txn_id VARCHAR(64) PRIMARY KEY REFERENCES transactions(txn_id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL DEFAULT 'New', -- New, Under Review, Confirmed Fraud, False Positive
    assigned_to VARCHAR(64),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);

CREATE TABLE IF NOT EXISTS labels (
    id SERIAL PRIMARY KEY,
    txn_id VARCHAR(64) REFERENCES transactions(txn_id) ON DELETE CASCADE,
    label INTEGER NOT NULL, -- 1 = Fraud, 0 = Legitimate, -1 = Under Review
    disposition VARCHAR(32) NOT NULL, -- Confirmed Fraud, False Positive, Under Review
    previous_disposition VARCHAR(32),
    analyst VARCHAR(64) NOT NULL,
    comment TEXT,
    labeled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labels_txn_id ON labels (txn_id);
