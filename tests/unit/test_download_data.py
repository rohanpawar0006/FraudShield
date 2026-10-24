"""Unit tests for synthetic dataset generation."""

from src.ingestion.download_data import PAYSIM_COLUMNS, generate_synthetic_paysim


def test_generate_synthetic_paysim() -> None:
    """Verify synthetic PaySim generator matches schema and constraints."""
    df = generate_synthetic_paysim(num_records=200, fraud_rate=0.05, seed=42)

    assert len(df) == 200
    for col in PAYSIM_COLUMNS:
        assert col in df.columns

    # Verify fraud rate is within expectation
    fraud_count = df["isFraud"].sum()
    assert fraud_count > 0

    # Verify non-negative amounts
    assert (df["amount"] > 0).all()

    # Verify chronological ordering
    assert (df["step"].diff().dropna() >= 0).all()
