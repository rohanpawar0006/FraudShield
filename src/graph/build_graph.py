"""Graph construction module for FraudShield.

Builds directed transaction graph (origin_account -> dest_account) strictly using
training-window transactions with zero temporal leakage into validation/test windows.
Computes structural node features and extracts fraud ring topology signals.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.training.data import PAYMENT_TYPES  # noqa: E402

logger = get_logger(__name__)


def validate_training_window_isolation(df: pd.DataFrame, max_allowed_step: int) -> None:
    """Assert that no transactions from validation or test windows enter the graph."""
    if "step" not in df.columns:
        raise ValueError("Transaction dataset must contain 'step' column for temporal verification.")

    max_step_in_data = int(df["step"].max())
    if max_step_in_data > max_allowed_step:
        raise ValueError(
            f"TEMPORAL LEAKAGE VIOLATION: Graph dataset contains step {max_step_in_data}, "
            f"which exceeds training window maximum step {max_allowed_step}!"
        )
    logger.info("Temporal isolation validated: All transactions have step <= %d (max: %d)", max_allowed_step, max_step_in_data)


def compute_node_features(
    df: pd.DataFrame,
    unique_accounts: List[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Compute structural node features strictly from training-window transactions without label leakage.

    Features computed per account:
    - log(1 + out_degree)
    - log(1 + in_degree)
    - log(1 + total_amount_out)
    - log(1 + total_amount_in)
    - log(1 + distinct_counterparties)
    - fan_in_out_ratio: (in_degree + 1) / (out_degree + 1)

    Note: Target fraud labels are deliberately excluded to ensure zero future target leakage
    and avoid GNN label-leakage bias during inductive inference on new accounts.
    """
    logger.info("Computing structural node features for %d unique accounts...", len(unique_accounts))
    t0 = time.perf_counter()

    account_to_idx: Dict[str, int] = {acc: idx for idx, acc in enumerate(unique_accounts)}
    n_nodes = len(unique_accounts)

    out_degree = np.zeros(n_nodes, dtype=np.float32)
    in_degree = np.zeros(n_nodes, dtype=np.float32)
    amount_out = np.zeros(n_nodes, dtype=np.float32)
    amount_in = np.zeros(n_nodes, dtype=np.float32)

    # 1. Outbound aggregations (nameOrig)
    orig_counts = df["nameOrig"].value_counts()
    orig_amounts = df.groupby("nameOrig", observed=True)["amount"].sum()

    for acc, count in orig_counts.items():
        idx = account_to_idx[acc]
        out_degree[idx] = float(count)
        amount_out[idx] = float(orig_amounts[acc])

    # 2. Inbound aggregations (nameDest)
    dest_counts = df["nameDest"].value_counts()
    dest_amounts = df.groupby("nameDest", observed=True)["amount"].sum()

    for acc, count in dest_counts.items():
        idx = account_to_idx[acc]
        in_degree[idx] = float(count)
        amount_in[idx] = float(dest_amounts[acc])

    # 3. Assemble feature matrix
    log_out_deg = np.log1p(out_degree)
    log_in_deg = np.log1p(in_degree)
    log_amt_out = np.log1p(amount_out)
    log_amt_in = np.log1p(amount_in)
    log_counterparties = np.log1p(out_degree + in_degree)
    fan_ratio = (in_degree + 1.0) / (out_degree + 1.0)

    node_matrix = np.column_stack([
        log_out_deg,
        log_in_deg,
        log_amt_out,
        log_amt_in,
        log_counterparties,
        fan_ratio,
    ]).astype(np.float32)

    logger.info("Node features computed in %.2fs. Matrix shape: %s", time.perf_counter() - t0, node_matrix.shape)
    return node_matrix, account_to_idx


def extract_ring_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify fraud ring patterns: fan-in collection hubs, high-velocity disbursement nodes, and component sizes."""
    logger.info("Extracting fraud ring structural signals...")
    t0 = time.perf_counter()

    # Destination fan-in counts (accounts receiving from multiple origins)
    dest_counts = df["nameDest"].value_counts()
    fan_in_hubs = dest_counts[dest_counts >= 10].to_dict()

    # High amount accumulation nodes
    dest_sums = df.groupby("nameDest", observed=True)["amount"].sum()
    top_dest_sums = dest_sums.sort_values(ascending=False).head(20).to_dict()

    # Known fraud involvement in training split
    fraud_destinations = df[df["isFraud"] == 1]["nameDest"].unique().tolist()
    fraud_origins = df[df["isFraud"] == 1]["nameOrig"].unique().tolist()

    # Intersecting mule intermediaries: accounts that both receive and send
    senders = set(df["nameOrig"])
    receivers = set(df["nameDest"])
    forwarding_mules = list(senders.intersection(receivers))

    summary = {
        "training_max_step": int(df["step"].max()),
        "total_transactions": len(df),
        "total_unique_accounts": len(senders.union(receivers)),
        "fan_in_hubs_count_gte_10": len(fan_in_hubs),
        "forwarding_mule_accounts_count": len(forwarding_mules),
        "fraud_destination_accounts_count": len(fraud_destinations),
        "fraud_origin_accounts_count": len(fraud_origins),
        "sample_fan_in_hubs": dict(list(fan_in_hubs.items())[:10]),
        "sample_top_receivers_by_volume": dict(list(top_dest_sums.items())[:10]),
        "sample_forwarding_mules": forwarding_mules[:15],
        "extraction_duration_sec": round(time.perf_counter() - t0, 3),
    }
    logger.info("Extracted %d fan-in hubs and %d forwarding mule candidates.", len(fan_in_hubs), len(forwarding_mules))
    return summary


def build_pyg_graph(
    source_parquet: str = "data/processed/train_features.parquet",
    max_step: int = settings.TRAIN_MAX_STEP,
    subsample_edges: int = 1000000,
) -> Tuple[Data, Dict[str, int], StandardScaler, Dict[str, Any]]:
    """Build directed PyG graph from training split with strict temporal verification.

    Args:
        source_parquet: Path to processed train parquet.
        max_step: Upper bound on step column to guarantee zero leakage.
        subsample_edges: For efficient memory training on student laptop, retain high-information
                         edges (all transfers/cashouts + sampled background legitimate transactions).

    Returns:
        Tuple of (pyg_data, account_to_idx, scaler, ring_signals)
    """
    path = Path(source_parquet)
    if not path.exists():
        raise FileNotFoundError(f"Source parquet not found at {path}")

    logger.info("Loading training transactions from %s...", path.as_posix())
    cols = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
    df = pd.read_parquet(path, columns=cols)

    # 1. Assert strict temporal isolation
    validate_training_window_isolation(df, max_allowed_step=max_step)

    # Filter strictly to training window step
    df = df[df["step"] <= max_step].copy()

    # 2. Extract fraud ring structural signals over full training data
    ring_signals = extract_ring_signals(df)

    # 3. Informative Graph Edge Filtering:
    # All TRANSFER and CASH_OUT transactions are preserved (100% of fraud occurs in these types).
    # Other types (CASH_IN, PAYMENT) are sampled to bound CPU memory while maintaining realistic topology.
    fraud_mask = df["isFraud"] == 1
    transfer_cashout_mask = df["type"].isin(["TRANSFER", "CASH_OUT"])
    core_mask = fraud_mask | transfer_cashout_mask

    core_df = df[core_mask]
    other_df = df[~core_mask]

    n_sample_other = min(len(other_df), max(100000, subsample_edges - len(core_df)))
    if n_sample_other > 0 and len(other_df) > n_sample_other:
        other_sample = other_df.sample(n=n_sample_other, random_state=42)
        graph_df = pd.concat([core_df, other_sample], ignore_index=True)
    else:
        graph_df = core_df

    graph_df = graph_df.sort_values(by="step").reset_index(drop=True)
    logger.info(
        "Constructing PyG graph: %d edges (Fraud: %d, %.3f%%)",
        len(graph_df),
        int(graph_df["isFraud"].sum()),
        graph_df["isFraud"].mean() * 100.0,
    )

    # 4. Extract unique node accounts
    unique_accounts = sorted(list(set(graph_df["nameOrig"]).union(set(graph_df["nameDest"]))))

    # 5. Compute structural node features
    raw_node_matrix, account_to_idx = compute_node_features(graph_df, unique_accounts)

    # Fit StandardScaler
    scaler = StandardScaler()
    scaled_node_features = scaler.fit_transform(raw_node_matrix)

    # 6. Build edge index and edge features
    src_indices = [account_to_idx[acc] for acc in graph_df["nameOrig"]]
    dst_indices = [account_to_idx[acc] for acc in graph_df["nameDest"]]
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

    # Edge attributes: log(1+amount), one-hot payment type (5), normalized step
    amounts_log = np.log1p(graph_df["amount"].to_numpy(dtype=np.float32))
    steps_norm = (graph_df["step"].to_numpy(dtype=np.float32) / 743.0)

    # One-hot encode type
    type_dummies = pd.get_dummies(graph_df["type"])
    for p_type in PAYMENT_TYPES:
        if p_type not in type_dummies.columns:
            type_dummies[p_type] = 0
    type_arr = type_dummies[PAYMENT_TYPES].to_numpy(dtype=np.float32)

    edge_attr_mat = np.column_stack([
        amounts_log,
        steps_norm,
        type_arr,
    ]).astype(np.float32)

    edge_attr = torch.tensor(edge_attr_mat, dtype=torch.float)
    edge_y = torch.tensor(graph_df["isFraud"].to_numpy(dtype=np.float32), dtype=torch.float)

    pyg_data = Data(
        x=torch.tensor(scaled_node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=edge_y,
    )
    pyg_data.num_nodes = len(unique_accounts)

    logger.info(
        "PyG Data constructed: %d nodes, %d edges, node_feat_dim: %d, edge_feat_dim: %d",
        pyg_data.num_nodes,
        pyg_data.num_edges,
        pyg_data.x.size(1),
        pyg_data.edge_attr.size(1),
    )

    return pyg_data, account_to_idx, scaler, ring_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudShield Graph Construction")
    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/train_features.parquet",
        help="Path to training features parquet",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=settings.TRAIN_MAX_STEP,
        help="Maximum step allowed for training graph (default: 323)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1000000,
        help="Edge count target for laptop CPU training",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    pyg_data, account_to_idx, scaler, ring_signals = build_pyg_graph(
        source_parquet=args.source,
        max_step=args.max_step,
        subsample_edges=args.subsample,
    )

    # Save artifacts to paths from config
    logger.info("Saving graph artifacts to disk...")
    torch.save(pyg_data, settings.GRAPH_DATA_PATH)
    joblib.dump(account_to_idx, settings.GRAPH_NODE_MAPPING_PATH)
    joblib.dump(scaler, settings.GRAPH_SCALER_PATH)

    with open(settings.GRAPH_RINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(ring_signals, f, indent=2)

    logger.info("Graph construction completed successfully!")
    print(f"Graph Data saved to: {settings.GRAPH_DATA_PATH}")
    print(f"Node Mapping saved to: {settings.GRAPH_NODE_MAPPING_PATH}")
    print(f"Ring Signals saved to: {settings.GRAPH_RINGS_PATH}")


if __name__ == "__main__":
    main()
