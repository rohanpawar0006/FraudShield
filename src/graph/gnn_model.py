"""PyTorch Geometric GraphSAGE edge fraud classification model for FraudShield.

Learns neighborhood embeddings for financial accounts and classifies transaction edges.
Uses class-weighted loss for severe imbalance, evaluates against validation window,
and logs metrics and artifacts to MLflow as FraudShield-GNN.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.nn import SAGEConv  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.training.data import PAYMENT_TYPES  # noqa: E402

logger = get_logger(__name__)


class GraphSAGEEdgeClassifier(nn.Module):
    """2-layer GraphSAGE architecture with edge prediction head for transaction fraud classification.

    Design Rationale (Edge-level vs Node-level):
    1. In financial networks, accounts are not inherently fraudulent; transactions (edges u -> v) are.
    2. Edge prediction directly provides per-transaction risk scores that align with XGBoost for blending.
    3. The model jointly reasons over source neighborhood hu, destination neighborhood hv, and edge telemetry e_uv.
    """

    def __init__(
        self,
        node_in_channels: int = 6,
        edge_in_channels: int = 7,
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # GraphSAGE neighborhood aggregation layers
        self.conv1 = SAGEConv(node_in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Edge classification MLP head: [h_u || h_v || edge_attr]
        edge_mlp_in = hidden_channels * 2 + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node representations across 2-hop neighborhoods."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        eval_edges: torch.Tensor,
        eval_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode nodes and predict fraud logits for the given edges."""
        node_emb = self.encode_nodes(x, edge_index)
        src_nodes = eval_edges[0]
        dst_nodes = eval_edges[1]

        src_emb = node_emb[src_nodes]
        dst_emb = node_emb[dst_nodes]

        combined = torch.cat([src_emb, dst_emb, eval_edge_attr], dim=-1)
        logits = self.edge_mlp(combined).squeeze(-1)
        return logits


def load_validation_edges(
    val_parquet_path: str = "data/processed/val_features.parquet",
    account_to_idx: Optional[Dict[str, int]] = None,
    max_edges: int = 50000,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load validation split edges ($324 <= step <= 378$) with known node mapping for early stopping."""
    path = Path(val_parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation dataset not found at {path}")

    logger.info("Loading validation edges from %s...", path.as_posix())
    df = pd.read_parquet(path, columns=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"])

    # Temporal isolation assertion
    assert (df["step"] > settings.TRAIN_MAX_STEP).all(), "Validation edges overlap with training step!"
    assert (df["step"] <= settings.VAL_MAX_STEP).all(), "Validation edges exceed validation step!"

    # Filter to transactions where accounts are in training graph or map unseen to 0
    mapping = account_to_idx or {}
    unknown_node_idx = 0

    if max_edges > 0 and len(df) > max_edges:
        # Stratified sample keeping all fraud
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0].sample(n=max_edges - len(fraud_df), random_state=42)
        eval_df = pd.concat([fraud_df, legit_df], ignore_index=True)
    else:
        eval_df = df

    src_idx = [mapping.get(acc, unknown_node_idx) for acc in eval_df["nameOrig"]]
    dst_idx = [mapping.get(acc, unknown_node_idx) for acc in eval_df["nameDest"]]
    val_edges = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # Edge attributes
    amounts_log = np.log1p(eval_df["amount"].to_numpy(dtype=np.float32))
    steps_norm = eval_df["step"].to_numpy(dtype=np.float32) / 743.0
    type_dummies = pd.get_dummies(eval_df["type"])
    for p_type in PAYMENT_TYPES:
        if p_type not in type_dummies.columns:
            type_dummies[p_type] = 0
    type_arr = type_dummies[PAYMENT_TYPES].to_numpy(dtype=np.float32)

    val_edge_attr = torch.tensor(
        np.column_stack([amounts_log, steps_norm, type_arr]),
        dtype=torch.float,
    )
    val_y = eval_df["isFraud"].to_numpy(dtype=np.int64)

    logger.info("Prepared %d validation edges (Fraud: %d, %.3f%%)", len(eval_df), int(val_y.sum()), val_y.mean() * 100.0)
    return val_edges, val_edge_attr, val_y


def train_gnn_model(
    graph_data_path: str = settings.GRAPH_DATA_PATH,
    node_mapping_path: str = settings.GRAPH_NODE_MAPPING_PATH,
    val_parquet_path: str = "data/processed/val_features.parquet",
    epochs: int = 15,
    hidden_channels: int = 64,
    lr: float = 0.005,
    batch_edges: int = 65536,
) -> Dict[str, Any]:
    """Train GraphSAGE edge classifier on training graph with early stopping on validation split."""
    logger.info("Starting GraphSAGE edge classification training...")
    t_start = time.perf_counter()

    data: Data = torch.load(graph_data_path, weights_only=False)
    account_to_idx: Dict[str, int] = joblib.load(node_mapping_path)

    # Class weighting
    y_train = data.y
    n_pos = float((y_train == 1.0).sum())
    n_neg = float((y_train == 0.0).sum())
    pos_weight_val = max(n_neg / max(n_pos, 1.0), 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float)
    logger.info("Training edges: %d (Pos: %.0f, Neg: %.0f). pos_weight: %.2f", data.num_edges, n_pos, n_neg, pos_weight_val)

    # Validation edges
    val_edges, val_edge_attr, val_y = load_validation_edges(
        val_parquet_path=val_parquet_path,
        account_to_idx=account_to_idx,
    )

    model = GraphSAGEEdgeClassifier(
        node_in_channels=data.x.size(1),
        edge_in_channels=data.edge_attr.size(1),
        hidden_channels=hidden_channels,
        dropout=0.2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_pr_auc = 0.0
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None
    patience = 4
    no_improve_count = 0

    # Subsample training edges per epoch for memory and speed on CPU
    n_edges = data.edge_index.size(1)

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()

        # Edge sampling for epoch mini-batching
        perm = torch.randperm(n_edges)[: min(n_edges, batch_edges * 4)]
        train_edge_sub = data.edge_index[:, perm]
        train_attr_sub = data.edge_attr[perm]
        train_y_sub = data.y[perm]

        optimizer.zero_grad()
        logits = model(
            x=data.x,
            edge_index=train_edge_sub,  # Message passing graph
            eval_edges=train_edge_sub,  # Edges to score
            eval_edge_attr=train_attr_sub,
        )

        loss = criterion(logits, train_y_sub)
        loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(
                x=data.x,
                edge_index=train_edge_sub,
                eval_edges=val_edges,
                eval_edge_attr=val_edge_attr,
            )
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_prec, val_rec, _ = precision_recall_curve(val_y, val_probs)
        val_pr_auc = float(auc(val_rec, val_prec))
        val_roc_auc = float(roc_auc_score(val_y, val_probs))
        ep_duration = time.perf_counter() - t_ep

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Val PR-AUC: %.4f | Val ROC-AUC: %.4f",
            epoch, epochs, ep_duration, loss.item(), val_pr_auc, val_roc_auc
        )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info("Early stopping triggered at epoch %d (best epoch: %d)", epoch, best_epoch)
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    total_training_sec = time.perf_counter() - t_start
    logger.info("GNN training finished in %.1fs. Best Val PR-AUC: %.4f at epoch %d", total_training_sec, best_val_pr_auc, best_epoch)

    # Save model checkpoint
    torch.save(model.state_dict(), settings.GNN_MODEL_PATH)
    logger.info("GNN model checkpoint saved to %s", settings.GNN_MODEL_PATH)

    # Precompute GNN node risk lookup for ultra-fast serving
    logger.info("Precomputing GNN risk lookup for fast serving...")
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode_nodes(data.x, data.edge_index)
        # Compute node outbound risk norm
        node_risk_scores = torch.norm(node_embeddings, dim=-1).cpu().numpy()
        # Min-max normalize risk scores into [0, 1]
        min_r, max_r = node_risk_scores.min(), node_risk_scores.max()
        if max_r > min_r:
            node_risk_scores = (node_risk_scores - min_r) / (max_r - min_r)
        else:
            node_risk_scores = np.zeros_like(node_risk_scores)

    # Build account_id -> gnn_score dataframe
    accounts = list(account_to_idx.keys())
    indices = [account_to_idx[acc] for acc in accounts]
    gnn_scores_df = pd.DataFrame({
        "account_id": accounts,
        "gnn_risk_score": node_risk_scores[indices].astype(np.float32),
    })
    gnn_scores_df.to_parquet(settings.GNN_SCORES_LOOKUP_PATH, index=False)
    logger.info("Saved GNN score lookup for %d accounts to %s", len(gnn_scores_df), settings.GNN_SCORES_LOOKUP_PATH)

    # MLflow tracking
    import os

    import mlflow
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="gnn-graphsage"):
        mlflow.log_params({
            "model_architecture": "GraphSAGEEdgeClassifier",
            "hidden_channels": hidden_channels,
            "learning_rate": lr,
            "pos_weight": pos_weight_val,
            "best_epoch": best_epoch,
            "total_epochs": epochs,
            "training_nodes": data.num_nodes,
            "training_edges": data.num_edges,
        })
        mlflow.log_metrics({
            "val_pr_auc": best_val_pr_auc,
            "training_duration_sec": total_training_sec,
        })
        mlflow.log_artifact(settings.GNN_MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(settings.GRAPH_RINGS_PATH, artifact_path="graph")

    return {
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "training_duration_sec": total_training_sec,
        "model_path": settings.GNN_MODEL_PATH,
        "lookup_path": settings.GNN_SCORES_LOOKUP_PATH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudShield GNN Training")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = parser.parse_args()

    train_gnn_model(
        epochs=args.epochs,
        hidden_channels=args.hidden,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
