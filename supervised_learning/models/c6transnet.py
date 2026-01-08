import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 组件: SE-Block ---
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    让模型自动学习通道的重要性，增强关键特征（如斜线连珠特征）。
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --- 2. 组件: Dilated ResBlock ---
class DilatedResBlock(nn.Module):
    """
    带空洞卷积的残差块
    dilation: 空洞率。设置为 2 或 3 可以大幅增加感受野。
    """
    def __init__(self, channels, dilation=1):
        super(DilatedResBlock, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层卷积 (标准 3x3)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # SE 注意力
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se(out) # 加入注意力
        
        out += residual
        out = self.relu(out)
        return out

# --- 3. 组件: Relative Positional Encoding ---
class RelativeGlobalAttention(nn.Module):
    """
    带有 2D 相对位置编码的 Self-Attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 2D Relative Position Bias
        # 19x19 的棋盘，相对距离范围是 [-18, 18]，所以表的大小是 (2*18+1)^2
        self.window_size = 19
        self.num_relative_distance = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # [2*19-1 * 2*19-1, nH]

        # 生成相对坐标索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add Relative Position Bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# --- 4. 主模型: C6TransNet v2 ---
class C6TransNet(nn.Module):
    def __init__(self, input_planes=17, embed_dim=256, depth=6, num_heads=8):
        super(C6TransNet, self).__init__()
        
        self.board_size = 19
        self.num_points = 19 * 19
        
        # --- Stem (ResNet with Dilation) ---
        self.conv_in = nn.Conv2d(input_planes, embed_dim, 3, 1, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # 混合使用不同 Dilation 的 Block
        self.res_stack = nn.Sequential(
            DilatedResBlock(embed_dim, dilation=1),
            DilatedResBlock(embed_dim, dilation=2), # 增大感受野
            DilatedResBlock(embed_dim, dilation=3), # 进一步增大，看清长斜线
            DilatedResBlock(embed_dim, dilation=1),
            DilatedResBlock(embed_dim, dilation=1),
        )
        
        # --- Body (ViT with Relative Pos) ---
        # 手写 Transformer Block 以集成 Relative Position Bias
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': RelativeGlobalAttention(embed_dim, num_heads=num_heads),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            }) for _ in range(depth)
        ])
        
        # --- Auto-regressive Head ---
        # Step 1: Predict First Move
        self.head_move1 = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Move1 Embedding: 把第一步的选择变成向量，融入到第二步预测中
        self.move1_embed = nn.Embedding(self.num_points + 1, embed_dim) # +1 for padding/none
        
        # Step 2: Predict Second Move
        # 输入是: Global Features + Move1 Embedding
        self.head_move2 = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward_transformer(self, x):
        # x: (B, 361, C)
        for block in self.blocks:
            # Attention
            residual = x
            x = block['norm1'](x)
            x = block['attn'](x)
            x += residual
            
            # MLP
            residual = x
            x = block['norm2'](x)
            x = block['mlp'](x)
            x += residual
        return x

    def forward(self, x, move1_idx=None):
        """
        Args:
            x: (B, 17, 19, 19)
            move1_idx: (B,) Optional. 
                       如果在训练时，传入 Ground Truth 的第一步索引，用于 Teacher Forcing 训练第二步。
                       如果在推理时，先不传，拿到第一步预测后，再传进来预测第二步。
        Returns:
            policy1: (B, 361)
            policy2: (B, 361) or None
            value: (B, 1)
        """
        B = x.size(0)
        
        # 1. Stem
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        out = self.res_stack(out)
        
        # 2. Body
        # (B, C, H, W) -> (B, H*W, C)
        tokens = out.flatten(2).transpose(1, 2) 
        features = self.forward_transformer(tokens) # (B, 361, C)
        
        # 3. Heads
        
        # Value
        global_feat = features.mean(dim=1)
        value = self.value_head(global_feat)
        
        # Policy Step 1
        policy1_logits = self.head_move1(features).squeeze(-1) # (B, 361)
        
        # Policy Step 2 (Auto-regressive)
        policy2_logits = None
        
        if move1_idx is not None:
            # Training Mode or Inference Step 2
            # 将 move1 的位置信息加回到特征中
            # move1_idx: (B,) -> Embedding -> (B, C) -> Expand to (B, 361, C)
            m1_emb = self.move1_embed(move1_idx).unsqueeze(1) # (B, 1, C)
            
            # 融合策略：简单相加 或者 Concat
            # 这里选择相加，类似于 Position Embedding 的叠加
            features_v2 = features + m1_emb
            
            policy2_logits = self.head_move2(features_v2).squeeze(-1)
            
        return policy1_logits, policy2_logits, value

    def predict(self, x):
        """
        推理专用函数：自动执行两步预测
        """
        # Step 1
        p1_logits, _, value = self.forward(x, move1_idx=None)
        
        # Greedy Select Move 1
        # 注意：这里应该做 Masking (不能下在有子的地方)
        # 为简单起见，这里只返回 logits，由外部做 masking 和 selection
        # 但为了自回归，我们需要知道 move1 是啥
        
        # 这里假设外部会调用 forward 两次，或者我们在内部做 argmax
        # 为了灵活性，我们返回 p1_logits，用户选完后，再调 forward 拿 p2
        return p1_logits, value

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C6TransNet().to(device)
    print(f"Model v2 Created. Params: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(2, 17, 19, 19).to(device)
    
    # 1. 预测第一步
    p1, _, v = model(x)
    print(f"P1 shape: {p1.shape}")
    
    # 2. 假设第一步选了位置 100 和 200
    m1 = torch.tensor([100, 200]).to(device)
    
    # 3. 预测第二步 (Teacher Forcing)
    p1, p2, v = model(x, move1_idx=m1)
    print(f"P2 shape: {p2.shape}")
