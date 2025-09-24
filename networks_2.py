import torch
import torch.nn as nn
import math
from typing import Dict

class DiscreteCondProj(nn.Module):
    def __init__(self, in_dim=3, hid=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, y_onehot):  # [B,A,3] oder [B,3]
        return self.net(y_onehot)
class AgentSelfAttention(nn.Module):
    """Temporal self-attention within each agent over its own time steps with multi-head attention."""
    def __init__(self, input_dim=256, d_k=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_k = d_k
        self.dropout = dropout
        self.num_heads = 4
        self.d_k_per_head = self.d_k // self.num_heads  # 256 / 4 = 64

        # Linear layers for Q, K, V
        self.query = nn.Linear(input_dim, d_k)
        self.key = nn.Linear(input_dim, d_k)
        self.value = nn.Linear(input_dim, d_k)
        
        # Output projection after concatenating heads
        self.out_proj = nn.Linear(d_k, d_k)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_k)

    def forward(self, x, feature_mask):
        B, A, T, F = x.shape
        x = self.norm(x)
        # Compute Q, K, V: [B, A, T, d_k]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head: [B, A, T, num_heads, d_k_per_head]
        Q = Q.view(B, A, T, self.num_heads, self.d_k_per_head).transpose(2, 3)  # [B, A, num_heads, T, d_k_per_head]
        K = K.view(B, A, T, self.num_heads, self.d_k_per_head).transpose(2, 3)
        V = V.view(B, A, T, self.num_heads, self.d_k_per_head).transpose(2, 3)

        # Compute attention scores: [B, A, num_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k_per_head)

        # Apply feature mask: [B, A, T] -> [B, A, 1, 1, T]
        if feature_mask is not None:
            mask = feature_mask.unsqueeze(2).unsqueeze(2)  # Broadcast to heads
            scores = scores.masked_fill(~mask, -float('inf'))

        # Compute attention weights
        attn_weights = nn.functional.softmax(scores, dim=-1)  # [B, A, num_heads, T, T]

        # Handle NaN: Set weights to 0 where all keys are masked
        any_valid_key = feature_mask.any(dim=-1)  # [B, A]
        attn_weights = torch.where(
            any_valid_key.unsqueeze(2).unsqueeze(3).unsqueeze(4),
            attn_weights,
            torch.zeros_like(attn_weights)
        )

        # Compute attention output: [B, A, num_heads, T, d_k_per_head]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back: [B, A, num_heads, T, d_k_per_head] -> [B, A, T, d_k]
        attn_output = attn_output.transpose(2, 3).contiguous().view(B, A, T, self.d_k)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection + dropout + norm
        return x + self.dropout(attn_output)

class AgentCrossAttention(nn.Module):
    """Self-attention across agents at each time step with multi-head attention."""
    def __init__(self, input_dim=256, d_k=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_k = d_k
        self.dropout = dropout
        self.num_heads = 4
        self.d_k_per_head = self.d_k // self.num_heads  # 256 / 4 = 64

        # Linear layers for Q, K, V
        self.query = nn.Linear(input_dim, d_k)
        self.key = nn.Linear(input_dim, d_k)
        self.value = nn.Linear(input_dim, d_k)
        
        # Output projection
        self.out_proj = nn.Linear(d_k, d_k)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_k)

    def forward(self, x, feature_mask):
        B, A, T, F = x.shape
        x = self.norm(x)
        x_trans = x.transpose(1, 2)  # [B, T, A, F]
        
        # Compute Q, K, V: [B, T, A, d_k]
        Q = self.query(x_trans)
        K = self.key(x_trans)
        V = self.value(x_trans)

        # Reshape for multi-head: [B, T, A, num_heads, d_k_per_head]
        Q = Q.view(B, T, A, self.num_heads, self.d_k_per_head).transpose(2, 3)  # [B, T, num_heads, A, d_k_per_head]
        K = K.view(B, T, A, self.num_heads, self.d_k_per_head).transpose(2, 3)
        V = V.view(B, T, A, self.num_heads, self.d_k_per_head).transpose(2, 3)

        # Compute attention scores: [B, T, num_heads, A, A]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k_per_head)

        # Apply feature mask: [B, A, T] -> [B, T, A] -> [B, T, 1, 1, A]
        if feature_mask is not None:
            mask_trans = feature_mask.permute(0, 2, 1)  # [B, T, A]
            scores = scores.masked_fill(~mask_trans.unsqueeze(2).unsqueeze(3), -float('inf'))

        # Compute attention weights
        attn_weights = nn.functional.softmax(scores, dim=-1)  # [B, T, num_heads, A, A]

        # Handle NaN
        any_valid_key = mask_trans.any(dim=-1)  # [B, T]
        attn_weights = torch.where(
            any_valid_key.unsqueeze(2).unsqueeze(3).unsqueeze(4),
            attn_weights,
            torch.zeros_like(attn_weights)
        )

        # Compute attention output: [B, T, num_heads, A, d_k_per_head]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back: [B, T, num_heads, A, d_k_per_head] -> [B, T, A, d_k]
        attn_output = attn_output.transpose(2, 3).contiguous().view(B, T, A, self.d_k)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Transpose back: [B, A, T, d_k]
        attn_output = attn_output.transpose(1, 2)

        # Residual connection + dropout + norm
        return x + self.dropout(attn_output)

class AgentRoadGraphAttention(nn.Module):
    """Cross-attention where each agent attends to roadgraph features with multi-head attention."""
    def __init__(self, input_dim=256, d_k=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_k = d_k
        self.dropout = dropout
        self.num_heads = 4
        self.d_k_per_head = self.d_k // self.num_heads  # 256 / 4 = 64

        # Linear layers for Q, K, V
        self.query = nn.Linear(input_dim, d_k)
        self.key = nn.Linear(input_dim, d_k)
        self.value = nn.Linear(input_dim, d_k)
        
        # Output projection
        self.out_proj = nn.Linear(d_k, d_k)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_k)

    def forward(self, x, roadgraph_tensor, roadgraph_mask):
        B, A, T, F = x.shape
        RG = roadgraph_tensor.size(1)  # Number of roadgraph elements
        x = self.norm(x)
        # Compute Q from agents: [B, A, T, d_k]
        Q = self.query(x)
        # Compute K, V from roadgraph: [B, RG, d_k]
        K = self.key(roadgraph_tensor)
        V = self.value(roadgraph_tensor)

        # Reshape for multi-head
        Q = Q.view(B, A, T, self.num_heads, self.d_k_per_head).permute(0, 3, 1, 2, 4)  # [B, num_heads, A, T, d_k_per_head]
        K = K.view(B, RG, self.num_heads, self.d_k_per_head).permute(0, 2, 1, 3)       # [B, num_heads, RG, d_k_per_head]
        V = V.view(B, RG, self.num_heads, self.d_k_per_head).permute(0, 2, 1, 3)       # [B, num_heads, RG, d_k_per_head]

        # Compute attention scores: [B, num_heads, A, T, RG]
        scores = torch.einsum('bhatd,bhrd->bhatr', Q, K) / math.sqrt(self.d_k_per_head)

        # Apply roadgraph mask: [B, RG] -> [B, 1, 1, 1, RG]
        if roadgraph_mask is not None:
            scores = scores.masked_fill(~roadgraph_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), -float('inf'))

        # Compute attention weights
        attn_weights = nn.functional.softmax(scores, dim=-1)  # [B, num_heads, A, T, RG]

        # Handle NaN
        any_valid_rg = roadgraph_mask.any(dim=-1)  # [B]
        attn_weights = torch.where(
            any_valid_rg.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4),
            attn_weights,
            torch.zeros_like(attn_weights)
        )

        # Compute attention output: [B, num_heads, A, T, d_k_per_head]
        attn_output = torch.einsum('bhatr,bhrd->bhatd', attn_weights, V)

        # Reshape back: [B, num_heads, A, T, d_k_per_head] -> [B, A, T, d_k]
        attn_output = attn_output.permute(0, 2, 3, 1, 4).contiguous().view(B, A, T, self.d_k)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection + dropout + norm
        return x + self.dropout(attn_output)
    
class FeatureMLP(nn.Module):
    def __init__(self, input_dim=1408, output_dim=256, hidden_dim=1024):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.out = nn.Linear(hidden_dim, output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim)  # Residual path
    
    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        out = self.out(h)
        return out + self.shortcut(x)  # Add raw input back

class RoadGraphMLP(nn.Module):
    def __init__(self, input_dim=20, output_dim=256, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, roadgraph_tensor):
        return self.mlp(roadgraph_tensor)

class OutputMLP(nn.Module):
    def __init__(self, input_dim=256, output_dim=3, hidden_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.layer3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h_1 = self.layer1(x)
        h_2 = self.layer2(h_1)
        h_3 = self.layer3(h_2)
        return h_3

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1028, dropout_rate=0.1):
        super(FeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Add LayerNorm before the FFN
        self.norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Add dropout after the FFN computation
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Pre-layernorm: normalize input first
        norm_x = self.norm(x)
        # Compute FFN output
        ffn_output = self.ffn(norm_x)
        # Apply dropout and add residual
        return x + self.dropout(ffn_output)

class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.feature_mlp = FeatureMLP()
        self.roadgraph_mlp = RoadGraphMLP()
        self.output_mlp = OutputMLP()
        self.transformer = nn.ModuleList([
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentCrossAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentRoadGraphAttention(), 
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentCrossAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentRoadGraphAttention(), 
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentCrossAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN(),
            AgentSelfAttention(),
            FeedForwardNN()

        ])

    def forward(self, noisy_tensor, roadgraph_tensor, feature_mask, roadgraph_mask):
        B, A, T, F = noisy_tensor.shape
        B, RG, NUM, D = roadgraph_tensor.shape
        noisy_tensor = self.feature_mlp(noisy_tensor)
        x = noisy_tensor
        if roadgraph_tensor is not None:
            roadgraph_features = self.roadgraph_mlp(roadgraph_tensor.view(B, RG, -1))
        else:
            roadgraph_features = None

        for block in self.transformer:
            if isinstance(block, AgentSelfAttention):
                x = block(x, feature_mask=feature_mask)
            elif isinstance(block, AgentCrossAttention):
                x = block(x, feature_mask=feature_mask)
            elif isinstance(block, AgentRoadGraphAttention):
                x = block(x, roadgraph_tensor=roadgraph_features, roadgraph_mask=roadgraph_mask)
            elif isinstance(block, FeedForwardNN):
                x = block(x)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
        output = self.output_mlp(x)
        return output

if __name__ == "__main__":
    print("all good!")