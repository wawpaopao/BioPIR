import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from transformers import AutoModel,AutoModelForMaskedLM
import math

def exists(v):
    return v is not None

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def default(v, d):
    return v if exists(v) else d

def check_embedding_stats(embeddings):
    print(f"Mean: {embeddings.mean().item()}")
    print(f"Std: {embeddings.std().item()}")
    print(f"Min: {embeddings.min().item()}")
    print(f"Max: {embeddings.max().item()}")
    print(f"Contains NaN: {torch.isnan(embeddings).any().item()}")
    print(f"Contains Inf: {torch.isinf(embeddings).any().item()}")


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class RectifiedFlow1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        embed_dim=320,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l2',
        objective='pred_flow',  # Changed from pred_noise
        self_condition=False,
        device=None,
        clip_during_sampling=False,
        clip_values=(-1., 1.),
    ):
        super().__init__()
        self.model = model
        self.self_condition = self_condition
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.objective = objective
        self.device = device
        self.num_timesteps = timesteps
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        
        # Clipping parameters
        self.clip_during_sampling = clip_during_sampling
        self.clip_values = clip_values
        
        # Normalization functions
        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one
        
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def predict_flow(self, x, t, padding_mask=None,x_self_cond=None):
        """Predict the flow from x_t"""
            
        return self.model(x, t, padding_mask=padding_mask,x_self_cond=x_self_cond)

    @torch.no_grad()
    def sample(self, x_start, padding_mask=None, steps=None):
        """Sample from x_start (source) to x_end (target)"""
        steps = default(steps, self.sampling_timesteps)
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Start from source sequences
        x = x_start
        
        # Time steps for ODE integration
        times = torch.linspace(0., 1., steps + 1, device=device)
        
        for i in range(len(times) - 1):
            t_curr = times[i]
            t_next = times[i + 1]
            dt = t_next - t_curr
            
            # Get current time for batch
            t_batch = torch.full((batch_size,), t_curr, device=device)
            
            # Predict flow
            # flow = self.predict_flow(x, t_batch, padding_mask=padding_mask)
            flow = self.predict_flow(x, t_batch, padding_mask=None)
            # Update x using Euler method
            x = x + flow * dt
            
            # Optional clipping
            if self.clip_during_sampling:
                x = torch.clamp(x, *self.clip_values)
        
        return self.unnormalize(x)

    def get_optimal_transport_flow(self, x0, x1):
        """Compute optimal transport flow between x0 and x1"""
        return x1 - x0


    def compute_loss(self,predicted_v, target_v, padding_mask):
    # 只计算非padding位置的loss
        loss = F.mse_loss(predicted_v, target_v, reduction='none')  # (B, L, D)
        
        
        # 应用mask
        mask = padding_mask.unsqueeze(-1).expand_as(loss)  # (B, L, D)
        masked_loss = loss * mask
        
        # 计算平均loss（只对有效位置）
        total_loss = masked_loss.sum()
        valid_elements = mask.sum()
        
        return total_loss / (valid_elements + 1e-8)
    def forward(self, x1, padding_mask=None):
        """
        Forward pass for training
        x0: source sequences
        x1: target sequences
        """
        

        x1 = self.normalize(x1)
        batch_size = x1.shape[0]
        device = x1.device
        
        # Normalize inputs
        x0 = torch.randn_like(x1)
        
        # Sample random times
        # t = torch.rand(batch_size, device=device) 
        
        dist = torch.distributions.Beta(2.0, 2.0)  # 中段
        t = dist.sample((batch_size,)).to(device)
        # Interpolate between x0 and x1
        x_t = x0 * (1 - t.view(-1, 1, 1)) + x1 * t.view(-1, 1, 1)
        
        # Compute optimal transport flow
        flow = self.get_optimal_transport_flow(x0, x1)
        
        flow_pred = self.predict_flow(x_t, t, padding_mask=None)
        
        # Compute loss
        loss = self.compute_loss(flow_pred, flow, padding_mask)
        
        return loss.mean()




class SimpleTransformerDenoiser(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            pep_max_len: int = 30,
            dropout: float = 0.1,
            self_condition: bool = False,
    ):
        """
        """
        super().__init__()
        
        self.self_condition = self_condition
        self.pep_max_len = pep_max_len
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(320, hidden_dim)
        self.input_norm = nn.LayerNorm(320)
        self.pos_encoder = nn.Parameter(torch.zeros(1, pep_max_len, hidden_dim))
        nn.init.normal_(self.pos_encoder, std=0.02)
        
        time_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2,320),
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
            self,
            x: torch.Tensor,        # 带噪声的嵌入, 形状: (B, L, D)
            t: torch.Tensor,        # 时间步, 形状: (B,)
            padding_mask: torch.Tensor = None, # 注意力掩码, 形状: (B, L)
            x_self_cond: torch.Tensor = None
    ):
        """
        """
        B, L, D = x.shape
        
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = x + x_self_cond
        
        x = self.input_norm(x)
        h = self.input_projection(x)
        
        h = h + self.pos_encoder[:, :L, :]
        
        time_embedding = self.time_mlp(t) 
        h = h + time_embedding.unsqueeze(1) 
        
        src_key_padding_mask = padding_mask if padding_mask is not None else None
        
        # 5. 通过Transformer编码器
        h = self.transformer_encoder(
            h, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 6. 通过输出头得到速度预测
        predicted_v = self.output_head(h)
        
        return predicted_v
    



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype))
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        position_ids = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # attention_mask: [B, L] -> [B, 1, 1, L]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = RoPEAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        attn_output = self.attention(self.norm1(x), attention_mask)
        
        x = x + attn_output
        ff_output = self.feed_forward(self.norm2(x))
        
        x = x + ff_output
       
        return x


class SimpleTransformerDenoiserWithRoPE(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 8,
            pep_max_len: int = 28,
            dropout: float = 0,
            self_condition: bool = False,
    ):
        """
        使用RoPE的简单Transformer去噪网络。
        """
        super().__init__()
        
        self.self_condition = self_condition
        self.pep_max_len = pep_max_len
        self.hidden_dim = hidden_dim
        
        # 1. 输入投影层
        self.input_projection = nn.Linear(320, hidden_dim)
        
        # 2. 时间编码MLP
        time_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim),
            
        )
        
        # 3. Transformer blocks with RoPE
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 320),
        )
        
        # 5. 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight,gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
            self,
            x: torch.Tensor,        # 带噪声的嵌入, 形状: (B, L, D)
            t: torch.Tensor,        # 时间步, 形状: (B,)
            padding_mask: torch.Tensor = None, # 注意力掩码, 形状: (B, L)
            x_self_cond: torch.Tensor = None
    ):
        """
        前向传播过程。
        """
        B, L, D = x.shape
        
        # (可选) 自条件处理
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = x + x_self_cond
        
        # 1. 输入投影
        h = self.input_projection(x)
            
        # 2. 计算时间编码并注入
        time_embedding = self.time_mlp(t)  # 形状: (B, D)
       
        h = h + time_embedding.unsqueeze(1)  # 广播到所有位置
        
        # 3. 通过Transformer blocks（RoPE在attention内部处理）
        for block in self.transformer_blocks:
            h = block(h, padding_mask)
            
        # 4. 通过输出头得到速度预测
        predicted_v = self.output_head(h)
        
        return predicted_v