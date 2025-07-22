import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv
from torch_scatter import scatter
import traceback
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
    precision_recall_curve, auc
)
from layers import *


# --- Utility Functions ---

def do_compute_metrics(pred_probs, labels):
    """Calculate and return a complete set of binary classification metrics."""
    labels = labels.astype(int)
    pred_classes = (pred_probs > 0.5).astype(int)
    acc = accuracy_score(labels, pred_classes)
    if len(np.unique(labels)) < 2:
        return acc, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
    auroc = roc_auc_score(labels, pred_probs)
    f1 = f1_score(labels, pred_classes, zero_division=0)
    precision = precision_score(labels, pred_classes, zero_division=0)
    recall = recall_score(labels, pred_classes, zero_division=0)
    ap = average_precision_score(labels, pred_probs)
    prec, rec, _ = precision_recall_curve(labels, pred_probs)
    aupr = auc(rec, prec)
    return acc, auroc, f1, precision, recall, ap, aupr


def compute_enhanced_mpnp_loss(predictions, uncertainty, label, kl_weight, uncertainty_weight):
    """
   Calculate the loss for the enhanced MPNP model (with debugging information and an ultimate fix).
   """
    # ==================== 1. Force Debug Output ====================
    # If you see this 'DEBUG' message in the logs, it means your changes have taken effect.
    # print(f"\n--- DEBUG: Inside the NEW loss function! ---")
    # print(f"    Initial shapes -> predictions: {predictions.shape}, label: {label.shape}")
    # =======================================================

    try:
        # ==================== 2. Ultimate Shape Fix ====================
        # Using .reshape(-1) can force a tensor of any shape (scalar, vector) into a 1D vector.
        # This is more robust than the previous .squeeze() and .view().
        predictions_reshaped = predictions.reshape(-1)
        label_reshaped = label.reshape(-1)
        # =======================================================

        # Print again to confirm if the shape has been correctly modified.
        # print(f"    Reshaped shapes  -> predictions: {predictions_reshaped.shape}, label: {label_reshaped.shape}")

        # Use the reshaped tensors to calculate the loss.
        prediction_loss = F.binary_cross_entropy_with_logits(predictions_reshaped, label_reshaped)

        # KL divergence loss (ensure uncertainty is also a vector).
        uncertainty_reshaped = uncertainty.reshape(-1)
        kl_loss = torch.mean(uncertainty_reshaped) * kl_weight

        # Uncertainty regularization loss.
        uncertainty_loss = torch.mean(1.0 / (uncertainty_reshaped + 1e-8)) * uncertainty_weight

        # Total loss.
        total_loss = prediction_loss + kl_loss + uncertainty_loss

        # Calculate probabilities for evaluation metrics.
        probs = torch.sigmoid(predictions_reshaped)

        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'kl_loss': kl_loss,
            'uncertainty_loss': uncertainty_loss,
            'predictions': probs.detach(),
        }

    except Exception as e:
        print(f"Error in compute_enhanced_mpnp_loss: {e}")
        device = predictions.device if hasattr(predictions, 'device') else 'cpu'
        return {
            'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'prediction_loss': torch.tensor(0.0, device=device),
            'kl_loss': torch.tensor(0.0, device=device),
            'uncertainty_loss': torch.tensor(0.0, device=device),
            'predictions': torch.tensor([0.5], device=device),
        }


class AverageMeter(object):
    """A utility class to compute and store the average and current values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self):
        return self.avg


# --- Core Modules of the Enhanced Model (Refactored and Fixed) ---

class DynamicAttention(nn.Module):
    """Dynamic attention module that correctly handles PyG batching."""

    def __init__(self, in_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.PReLU(),
            nn.Linear(in_dim // 2, 1), nn.Sigmoid()
        )

    def forward(self, x, batch):
        graph_repr = global_mean_pool(x, batch)
        attn_scores = self.attention_net(graph_repr)
        node_level_attn = attn_scores[batch]
        return x * node_level_attn


class EdgeToEdgeMessagePassing(nn.Module):
    """Edge-to-edge message passing module, simplified and robust."""

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.node_to_edge_proj = nn.Linear(node_dim, edge_dim, bias=False) if node_dim != edge_dim else nn.Identity()
        self.edge_update_net = nn.Sequential(nn.Linear(edge_dim, edge_dim), nn.PReLU(), nn.BatchNorm1d(edge_dim))

    def forward(self, x, edge_index, edge_attr, line_graph_edge_index):
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        proj_src = self.node_to_edge_proj(x[src_nodes])
        proj_dst = self.node_to_edge_proj(x[dst_nodes])
        fused_edge_attr = edge_attr + (proj_src + proj_dst) / 2.0
        if line_graph_edge_index is not None and line_graph_edge_index.size(1) > 0:
            line_src, line_dst = line_graph_edge_index
            messages = fused_edge_attr[line_src]
            aggregated_messages = scatter(messages, line_dst, dim=0, dim_size=fused_edge_attr.size(0), reduce='mean')
            fused_edge_attr = fused_edge_attr + self.edge_update_net(aggregated_messages)
        node_updates = scatter(fused_edge_attr, dst_nodes, dim=0, dim_size=x.size(0), reduce='mean')
        return node_updates


class GNP_Block(nn.Module):
    """Graph Neural Process block, refactored."""

    def __init__(self, hidden_dim, kge_dim, n_iter=2):
        super().__init__()
        self.n_iter = n_iter
        self.e2e_mp = EdgeToEdgeMessagePassing(hidden_dim, hidden_dim)
        self.attention = DynamicAttention(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, kge_dim)

    def forward(self, x, edge_index, edge_attr, batch, line_graph_edge_index):
        hidden_states = x.unsqueeze(0)
        for _ in range(self.n_iter):
            node_updates = self.e2e_mp(x, edge_index, edge_attr, line_graph_edge_index)
            _, hidden_states = self.gru(node_updates.unsqueeze(0), hidden_states)
            x = hidden_states.squeeze(0)
        attended_x = self.attention(x, batch)
        graph_emb = global_add_pool(attended_x, batch)
        graph_emb = self.readout(graph_emb)
        return x, graph_emb


class SimplifiedUncertaintyEstimator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.PReLU(),
            nn.Linear(in_dim // 2, 1), nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)


# --- Top-Level Enhanced Model (Using Fixed Modules) ---

class Improved_MPNP_DDI(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, n_iter, kge_dim, rel_total, dropout=0.2):
        super().__init__()
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = 3
        self.hidden_dim = hidden_dim
        self.node_preprocessor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.edge_preprocessor = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.gnp_blocks = nn.ModuleList([
            GNP_Block(hidden_dim, kge_dim, n_iter=n_iter) for _ in range(self.n_blocks)
        ])
        self.co_attention = CoAttentionLayer(kge_dim)
        self.KGE = RESCAL(rel_total, kge_dim)
        self.uncertainty_estimator = SimplifiedUncertaintyEstimator(kge_dim * 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, head_graphs, tail_graphs, relations):
        head_x = self.node_preprocessor(head_graphs.x)
        tail_x = self.node_preprocessor(tail_graphs.x)

        edge_dim = self.edge_preprocessor.in_features
        if not hasattr(head_graphs, 'edge_attr') or head_graphs.edge_attr is None:
            head_graphs.edge_attr = torch.zeros((head_graphs.edge_index.size(1), edge_dim),
                                                device=head_graphs.x.device)
        if not hasattr(tail_graphs, 'edge_attr') or tail_graphs.edge_attr is None:
            tail_graphs.edge_attr = torch.zeros((tail_graphs.edge_index.size(1), edge_dim),
                                                device=tail_graphs.x.device)

        head_edge_attr = self.edge_preprocessor(head_graphs.edge_attr)
        tail_edge_attr = self.edge_preprocessor(tail_graphs.edge_attr)

        head_representations = []
        tail_representations = []
        current_head_x, current_tail_x = head_x, tail_x

        for block in self.gnp_blocks:
            updated_head_x, head_graph_emb = block(
                current_head_x, head_graphs.edge_index, head_edge_attr,
                head_graphs.batch, getattr(head_graphs, 'line_graph_edge_index', None)
            )
            updated_tail_x, tail_graph_emb = block(
                current_tail_x, tail_graphs.edge_index, tail_edge_attr,
                tail_graphs.batch, getattr(tail_graphs, 'line_graph_edge_index', None)
            )
            head_representations.append(head_graph_emb)
            tail_representations.append(tail_graph_emb)
            current_head_x = (current_head_x + updated_head_x) / 2.0
            current_tail_x = (current_tail_x + updated_tail_x) / 2.0

        head_multi_scale = torch.stack(head_representations, dim=1)  # Shape: [B, N_blocks, D]
        tail_multi_scale = torch.stack(tail_representations, dim=1)  # Shape: [B, N_blocks, D]

        # ========================================
        # 1. Compute attention weights
        attentions = self.co_attention(head_multi_scale, tail_multi_scale)  # Shape: [B, N_blocks, 2]

        # 2. Extract head and tail attention weights and adjust the shape for broadcasting
        head_attn_weights = attentions[:, :, 0].unsqueeze(-1)  # Shape: [B, N_blocks, 1]
        tail_attn_weights = attentions[:, :, 1].unsqueeze(-1)  # Shape: [B, N_blocks, 1]

        # 3. Use attention weights for a weighted sum to get the final single embedding
        final_head_emb = torch.sum(head_multi_scale * head_attn_weights, dim=1)  # Shape: [B, D]
        final_tail_emb = torch.sum(tail_multi_scale * tail_attn_weights, dim=1)  # Shape: [B, D]

        # 4. Pass the final single embedding to the KGE model (now with only 3 parameters)
        scores = self.KGE(final_head_emb, final_tail_emb, relations)
        # =======================================================

        # Uncertainty estimation can still use the mean of the multi-scale representations.
        combined_repr = torch.cat([head_multi_scale.mean(dim=1), tail_multi_scale.mean(dim=1)], dim=1)
        uncertainty = self.uncertainty_estimator(combined_repr)

        return scores, uncertainty.squeeze(-1)

    def get_atom_embeddings(self, data):
        """
       A helper function that only runs the GNN part to extract the final representations of atoms (nodes).
       Input: A PyG Data object (representing one or more molecules).
       Output: Feature embeddings of the nodes.
       """
        # 1. Unpack the required data from the input data object.
        #    If edge_attr does not exist, create a zero tensor to ensure code robustness.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_preprocessor.in_features), device=x.device)
        else:
            edge_attr = data.edge_attr

        # 2. Run the exact same preprocessing steps as in the forward method.
        x = self.node_preprocessor(x)
        edge_attr = self.edge_preprocessor(edge_attr)

        # 3. Fully replicate the message passing loop from the forward method.
        for block in self.gnp_blocks:
            # Each block will return the updated node features and graph-level embeddings.
            # Here, we are only interested in the node-level features (updated_x).
            updated_x, _ = block(
                x,
                edge_index,
                edge_attr,
                batch,
                getattr(data, 'line_graph_edge_index', None)
            )
            # Apply the exact same residual connection as in the forward method.
            x = (x + updated_x) / 2.0

        # 4. After processing through all blocks, x is the final atom (node) embedding we need.
        return x


class Ablation_MPNP_DDI_no_Relation(nn.Module):
    """
   This is an ablation version of Improved_MPNP_DDI.
   It removes the KGE/RESCAL module and replaces it with a standard multi-class classifier.
   This model no longer uses relation embeddings to guide predictions.
   """

    def __init__(self, in_dim, edge_dim, hidden_dim, n_iter, kge_dim, rel_total, dropout=0.2):
        super().__init__()
        # --- This part is identical to the original model and is reused directly ---
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = 3
        self.hidden_dim = hidden_dim
        self.node_preprocessor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.edge_preprocessor = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.gnp_blocks = nn.ModuleList([
            GNP_Block(hidden_dim, kge_dim, n_iter=n_iter) for _ in range(self.n_blocks)
        ])
        self.co_attention = CoAttentionLayer(kge_dim)
        # -------------------------------------------

        # !!! Core Change 1: Remove KGE, replace with an MLP classifier !!!
        # self.KGE = RESCAL(rel_total, kge_dim)  <-- This line is no longer needed

        # Add a new MLP classifier
        # The input dimension is the concatenated dimension of the two drug embeddings (kge_dim * 2).
        # The output dimension is the total number of DDI types (rel_total), which is 86.
        self.classifier = nn.Sequential(
            nn.Linear(kge_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rel_total)  # Output logits for 86 classes
        )
        # -------------------------------------------

        # The uncertainty estimator can be kept or removed; here we keep it for now.
        self.uncertainty_estimator = SimplifiedUncertaintyEstimator(kge_dim * 2)
        self._initialize_weights()

    def _initialize_weights(self):
        # This method remains unchanged.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, head_graphs, tail_graphs, relations):  # Note: The 'relations' parameter will no longer be used.
        # --- The GNN and Co-Attention parts remain completely unchanged ---
        # This part of the code is identical to the original model and is reused directly.
        head_x = self.node_preprocessor(head_graphs.x)
        tail_x = self.node_preprocessor(tail_graphs.x)

        edge_dim = self.edge_preprocessor.in_features
        if not hasattr(head_graphs, 'edge_attr') or head_graphs.edge_attr is None:
            head_graphs.edge_attr = torch.zeros((head_graphs.edge_index.size(1), edge_dim),
                                                device=head_graphs.x.device)
        if not hasattr(tail_graphs, 'edge_attr') or tail_graphs.edge_attr is None:
            tail_graphs.edge_attr = torch.zeros((tail_graphs.edge_index.size(1), edge_dim),
                                                device=tail_graphs.x.device)

        head_edge_attr = self.edge_preprocessor(head_graphs.edge_attr)
        tail_edge_attr = self.edge_preprocessor(tail_graphs.edge_attr)

        head_representations = []
        tail_representations = []
        current_head_x, current_tail_x = head_x, tail_x

        for block in self.gnp_blocks:
            updated_head_x, head_graph_emb = block(
                current_head_x, head_graphs.edge_index, head_edge_attr,
                head_graphs.batch, getattr(head_graphs, 'line_graph_edge_index', None)
            )
            updated_tail_x, tail_graph_emb = block(
                current_tail_x, tail_graphs.edge_index, tail_edge_attr,
                tail_graphs.batch, getattr(tail_graphs, 'line_graph_edge_index', None)
            )
            head_representations.append(head_graph_emb)
            tail_representations.append(tail_graph_emb)
            current_head_x = (current_head_x + updated_head_x) / 2.0
            current_tail_x = (current_tail_x + updated_tail_x) / 2.0

        head_multi_scale = torch.stack(head_representations, dim=1)
        tail_multi_scale = torch.stack(tail_representations, dim=1)

        attentions = self.co_attention(head_multi_scale, tail_multi_scale)
        head_attn_weights = attentions[:, :, 0].unsqueeze(-1)
        tail_attn_weights = attentions[:, :, 1].unsqueeze(-1)

        final_head_emb = torch.sum(head_multi_scale * head_attn_weights, dim=1)
        final_tail_emb = torch.sum(tail_multi_scale * tail_attn_weights, dim=1)
        # ------------------------------------------------

        # 1. Concatenate the final embeddings of the two drugs.
        combined_drug_emb = torch.cat([final_head_emb, final_tail_emb], dim=1)

        # 2. Feed the concatenated embedding into the MLP classifier to get the logits vector.
        logits = self.classifier(combined_drug_emb)
        # -------------------------------------------

        # Uncertainty estimation can still use the mean of the multi-scale representations.
        combined_repr = torch.cat([head_multi_scale.mean(dim=1), tail_multi_scale.mean(dim=1)], dim=1)
        uncertainty = self.uncertainty_estimator(combined_repr)

        # Return logits instead of scores.
        return logits, uncertainty.squeeze(-1)
