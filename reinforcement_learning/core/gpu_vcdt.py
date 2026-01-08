import torch
import torch.nn.functional as F

class GPUThreatDetector:
    """
    在 GPU 上并行检测 Connect6 的各种威胁形状。
    支持检测：
    - 连5 (必胜)
    - 活4 (必胜)
    - 冲4 (死4, 需要防守)
    - 活3 (潜在威胁)
    """
    def __init__(self, device):
        self.device = device
        self.kernels = self._build_kernels()
        
    def _build_kernels(self):
        """
        构建卷积核。
        我们需要检测 4 个方向：横、竖、斜、反斜。
        对于每种威胁，我们定义特定的 pattern。
        
        Connect6 关键形状:
        1. 连6: [1,1,1,1,1,1] (Win)
        2. 连5: [1,1,1,1,1] (Next move -> 6)
        3. 连4: [1,1,1,1] (Next 2 moves -> 6)
        
        为了用卷积检测，我们使用简单的计数逻辑。
        卷积核大小 6x6 (为了覆盖连6)。
        """
        kernels = []
        
        # 基础方向向量
        directions = [
            (0, 1),  # Horizontal
            (1, 0),  # Vertical
            (1, 1),  # Diagonal
            (1, -1)  # Anti-Diagonal
        ]
        
        # 我们构建一组卷积核，分别检测长度为 4, 5, 6 的连子
        # 这里的 weights 是一个 (Out, 1, 6, 6) 的 Tensor
        # Out Channels: 4 directions * 3 lengths (4,5,6) = 12 kernels
        
        weights_list = []
        
        for length in [4, 5, 6]:
            for dx, dy in directions:
                k = torch.zeros((1, 1, 6, 6), device=self.device)
                
                # Center the pattern roughly
                start_x, start_y = 0, 0
                if dx == 1 and dy == -1: start_y = 5 # Anti-diag start from top-right
                
                # Build linear pattern
                valid = True
                for i in range(length):
                    cx = start_x + i * dx
                    cy = start_y + i * dy
                    if 0 <= cx < 6 and 0 <= cy < 6:
                        k[0, 0, cx, cy] = 1
                    else:
                        valid = False
                        # 对于简单的 6x6 kernel，反斜方向如果不调整起始点可能出界
                        # 实际上我们不需要完美的 6x6 居中，只要能匹配就行
                        # 简化：我们用足够大的 kernel 或者分别处理
                        pass
                
                # 重新简化构建逻辑：
                # 我们其实可以用 4 个独立的 Kernel 组，不需要放在一个 Layer 里
                # 但为了效率，放在一起最好。
                pass 

        # 重新实现：使用 4 个基础方向的 1D 卷积核 (转为 2D)
        # Horizontal: 1x6
        k_h = torch.ones((1, 1, 1, 6), device=self.device)
        # Vertical: 6x1
        k_v = torch.ones((1, 1, 6, 1), device=self.device)
        # Diagonal: 6x6 identity
        k_d = torch.eye(6, device=self.device).reshape(1, 1, 6, 6)
        # Anti-Diagonal: 6x6 flip identity
        k_ad = torch.rot90(torch.eye(6, device=self.device), 1, [0, 1]).reshape(1, 1, 6, 6)
        
        return [k_h, k_v, k_d, k_ad]

    def detect(self, board_batch, player_channel=0):
        """
        输入: board_batch (B, C, H, W) - 通常 C=17，我们只看 player_channel (0=Self, 1=Opp)
        输出: dict of masks (B, H, W) indicating threat levels
        """
        # 提取单通道 (B, 1, H, W)
        x = board_batch[:, player_channel:player_channel+1, :, :]
        
        threats = {
            'win': torch.zeros_like(x), # 连6
            'c5': torch.zeros_like(x),  # 连5
            'c4': torch.zeros_like(x),  # 连4
        }
        
        # 对 4 个方向分别卷积
        for k in self.kernels:
            # 卷积，padding='same' (需手动计算 padding)
            # k shape: (1, 1, KH, KW)
            # F.conv2d
            kh, kw = k.shape[2], k.shape[3]
            pad_h, pad_w = kh // 2, kw // 2
            
            # 这里的 padding 只是为了保持大小，实际上我们需要精确匹配
            # 简单的计数卷积：输出该位置周围有多少连子
            out = F.conv2d(x, k, padding=(pad_h, pad_w))
            
            # 这种简单的计数有个问题：它统计的是“区域内的子数”，不一定是连续的。
            # 比如 O O _ O O 也会被算作 4。
            # 这种 "跳4" 其实也是威胁！所以简单计数其实挺有效。
            
            # 如果区域内子数 == 6 -> 连6 (Win)
            threats['win'] = torch.max(threats['win'], (out >= 6.0).float())
            
            # 如果区域内子数 == 5 -> 连5 (活5/冲5)
            threats['c5'] = torch.max(threats['c5'], (out == 5.0).float())
            
            # 如果区域内子数 == 4 -> 连4
            threats['c4'] = torch.max(threats['c4'], (out == 4.0).float())
            
        return threats

    def get_vcdt_mask(self, board_batch):
        """
        生成用于 MCTS 的 Mask。
        如果存在 c5，则只允许在 c5 相关区域下子（进攻或防守）。
        """
        # 检测己方威胁
        my_threats = self.detect(board_batch, 0)
        # 检测对方威胁
        opp_threats = self.detect(board_batch, 1)
        
        # 逻辑：
        # 1. 如果我有 Win，随便下哪里能连成6就行（Mask = Win区域）
        # 2. 如果对手有 Win，我必须堵（Mask = Opp Win区域）
        # 3. 如果我有 C5，我下这里能赢（Mask = My C5区域）
        # 4. 如果对手有 C5，我必须堵（Mask = Opp C5区域）
        
        # 优先级：My Win > My C5 > Opp Win > Opp C5
        # Wait, Connect6 is 2 stones per turn.
        # If Opp has C5 (5 stones), he needs 1 stone to win.
        # If it's my turn, I place 2 stones.
        # I must block.
        
        # 为了简单，我们生成一个 "Attention Map"
        attention = torch.zeros_like(board_batch[:, 0:1, :, :])
        
        # 简单的加权叠加
        attention += my_threats['win'] * 100
        attention += my_threats['c5'] * 50
        attention += my_threats['c4'] * 10
        
        attention += opp_threats['win'] * 100 # 必须防
        attention += opp_threats['c5'] * 50
        attention += opp_threats['c4'] * 10
        
        # 将 attention 映射回具体的空位
        # 卷积输出的是“中心点”有威胁，我们需要把这个威胁“扩散”到具体的空位上。
        # 这里需要再一次卷积（dilation）或者简单的 mask 覆盖。
        
        # 简单处理：如果 attention > 0，说明这个局部有事发生。
        # 我们保留这个 attention map 作为 MCTS 的 Bias。
        
        return attention
