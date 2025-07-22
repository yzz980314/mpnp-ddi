import torch
from torch import nn
import torch.nn.functional as F


class CoAttentionLayer(nn.Module):
    """
   Co-attention layer (fixed).
   Computes attention between multi scale representations of head and tail entities,
   and generates a weight for each scale.
   """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        # Use a learnable parameter matrix to compute affinity
        self.W = nn.Parameter(torch.randn(emb_size, emb_size))
        nn.init.xavier_uniform_(self.W)

    def forward(self, head_reps, tail_reps):
        # head_reps, tail_reps shape: [batch_size, n_blocks, emb_size]

        # 1. Compute the affinity matrix C = h^T * W * t
        # (h^T * W) -> [B, N, D] @ [D, D] -> [B, N, D]
        h_W = torch.matmul(head_reps, self.W)
        # C = (h^T * W) * t^T -> [B, N, D] @ [B, D, N] -> [B, N, N]
        affinity_matrix = torch.bmm(h_W, tail_reps.transpose(1, 2))

        # 2. Compute attention weights for head and tail
        # Apply softmax to the rows of the affinity matrix to get head attention
        attn_h = torch.softmax(affinity_matrix, dim=2)  # Shape: [B, N, N]
        # Apply softmax to the columns of the affinity matrix to get tail attention
        attn_t = torch.softmax(affinity_matrix, dim=1)  # Shape: [B, N, N]

        # 3. Aggregate attention weights into a contribution score for each scale
        # For each head scale, sum its attention with all other tail scales
        head_scores = attn_h.sum(dim=2)  # Shape: [B, N]
        # For each tail scale, sum its attention with all other head scales
        tail_scores = attn_t.sum(dim=1)  # Shape: [B, N]

        # 4. Stack into the final attention weight tensor
        # Shape: [B, N, 2] -> [batch_size, n_blocks, (head_weight, tail_weight)]
        return torch.stack([head_scores, tail_scores], dim=2)


class RESCAL(nn.Module):
    """
   RESCAL scoring function (fixed and simplified).
   Only accepts final, single entity embeddings.
   """

    def __init__(self, rel_total, dim):
        super(RESCAL, self).__init__()
        self.rel_emb = nn.Embedding(rel_total, dim * dim)
        self.dim = dim
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, head, tail, relations):
        # head, tail shape: [batch_size, dim]
        # relations shape: [batch_size]

        # Reshape for batch matrix multiplication
        head = head.unsqueeze(1)  # -> [batch_size, 1, dim]
        tail = tail.unsqueeze(2)  # -> [batch_size, dim, 1]

        # Get relation matrices
        rel_matrices = self.rel_emb(relations).view(-1, self.dim, self.dim)  # -> [batch_size, dim, dim]

        # Calculate score: h^T * M_r * t
        score = torch.bmm(torch.bmm(head, rel_matrices), tail).squeeze()  # -> [batch_size]

        return score


# --- Other layers already in your file (keep them) ---

class MultiHeadCoAttentionLayer(nn.Module):
    def __init__(self, emb_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == emb_size, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, receiver, attendant):
        batch_size = receiver.size(0)
        n_blocks = receiver.size(1)
        q = self.q_proj(attendant).view(batch_size, n_blocks, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(receiver).view(batch_size, n_blocks, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(receiver).view(batch_size, n_blocks, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, n_blocks, self.emb_size)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights.mean(dim=1)


class EdgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=32):
        super(EdgeAttention, self).__init__()
        self.edge_attention = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        src_idx, dst_idx = edge_index
        src_features = x[src_idx]
        dst_features = x[dst_idx]
        edge_features = torch.cat([src_features, dst_features, edge_attr], dim=1)
        edge_weights = self.edge_attention(edge_features).squeeze(-1)
        return edge_weights


class EnhancedRESCAL(nn.Module):
    def __init__(self, rel_total, dim, dropout=0.2):
        super(EnhancedRESCAL, self).__init__()
        self.rel_emb = nn.Embedding(rel_total, dim * dim)
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.feature_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.relation_aware = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, relations, multi_scale_features=None, attentions=None):
        batch_size = heads.size(0)
        rel_matrices = self.rel_emb(relations).view(batch_size, self.dim, self.dim)
        if multi_scale_features is not None:
            scale_weights = F.softmax(torch.sum(multi_scale_features, dim=2), dim=1).unsqueeze(2)
            multi_scale_fusion = torch.sum(multi_scale_features * scale_weights, dim=1)
            heads_fusion = torch.cat([heads.mean(dim=1), multi_scale_fusion], dim=1)
            tails_fusion = torch.cat([tails.mean(dim=1), multi_scale_fusion], dim=1)
            heads_enhanced = self.feature_fusion(heads_fusion)
            tails_enhanced = self.feature_fusion(tails_fusion)
        else:
            heads_enhanced = heads.mean(dim=1)
            tails_enhanced = tails.mean(dim=1)
        if attentions is not None:
            weighted_heads = torch.bmm(attentions, heads)
            weighted_tails = torch.bmm(attentions.transpose(1, 2), tails)
            heads_enhanced = (heads_enhanced + weighted_heads.mean(dim=1)) / 2
            tails_enhanced = (tails_enhanced + weighted_tails.mean(dim=1)) / 2
        heads_aware = self.relation_aware(heads_enhanced)
        tails_aware = self.relation_aware(tails_enhanced)
        heads_aware = self.dropout(heads_aware)
        tails_aware = self.dropout(tails_aware)
        h_mr = torch.bmm(heads_aware.unsqueeze(1), rel_matrices)
        scores = torch.bmm(h_mr, tails_aware.unsqueeze(2)).squeeze()
        scores = torch.clamp(scores, min=-10, max=10)
        return scores
