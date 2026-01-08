import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# 引入本地游戏逻辑
from .connect6_game import Connect6Game

class Connect6Dataset(Dataset):
    def __init__(self, csv_file, mode='train'):
        """
        Args:
            csv_file: 清洗后的 CSV 路径
            mode: 'train' 或 'val'。
        """
        self.df = pd.read_csv(csv_file)
        self.mode = mode
        
        # 过滤掉太短的对局（比如少于 5 步的没啥训练价值）
        # 这里我们动态处理
        self.data = self.df.to_dict('records')
        print(f"Dataset loaded: {len(self.data)} games.")

    def __len__(self):
        return len(self.data)

    def _coord_to_index(self, coord_str):
        """ 将 'j10' 转为 (row, col) """
        if not coord_str: return None
        c_char = coord_str[0]
        r_str = coord_str[1:]
        col = ord(c_char) - ord('a')
        row = int(r_str) - 1
        return row, col

    def __getitem__(self, idx):
        """
        随机从一局游戏中采样一个状态。
        返回:
            input_feature: (17, 19, 19)
            target_policy: (361,) - Multi-hot encoding (两个 1，其余 0)
            target_value: (1,) - 1.0 (赢), -1.0 (输), 0.0 (平)
        """
        row = self.data[idx]
        moves_str = row['moves']
        winner_color = row['winner'] # 'black', 'white', 'draw'
        
        moves = [m.strip() for m in moves_str.split(',') if m.strip()]
        total_moves = len(moves)
        
        if total_moves < 2:
            # 异常保护，递归取下一个
            return self.__getitem__((idx + 1) % len(self))
        
        # 随机选择一个切片点 step_idx
        # step_idx 指的是 move 列表的索引。
        # Move 0: 黑1子
        # Move 1,2: 白2子
        # Move 3,4: 黑2子
        # ...
        
        # 我们需要预测的是“这一手”。
        # 如果选 step_idx = 0，模型输入空盘，预测 Move 0。
        # 如果选 step_idx = 1，模型输入 Move 0 后的盘面，预测 Move 1,2。
        
        # 必须确保选择的点是回合的开始。
        # 回合结构: [0] | [1, 2] | [3, 4] | [5, 6] ...
        # 也就是可选的 index 是 0, 1, 3, 5, ... (除了0，都是奇数)
        
        possible_starts = [0] + list(range(1, total_moves, 2))
        # 去掉最后一步（因为最后一步是作为 Target 的，如果正好最后一步下完了，那是结束状态，无法预测下一步）
        # 实际上，如果 total_moves 是偶数，最后一次是 [N-2, N-1]，我们可以预测它。
        # 如果 total_moves 是奇数，最后一次是 [N-1]，只有1子？(根据清洗逻辑，不完整的会被修补或丢弃，假设完整)
        
        if not possible_starts:
             return self.__getitem__((idx + 1) % len(self))
             
        # 随机采样一个时刻 T
        chosen_start_idx = np.random.choice(possible_starts)
        
        # 确定当前玩家
        # Move 0 (idx=0): Black
        # Move 1 (idx=1): White
        # Move 3 (idx=3): Black
        # 规律: if idx == 0: Black; elif (idx // 2) % 2 == 0: White; else: Black
        # 不对，更简单的：
        # Turn 0: Black (1 stone) -> idx 0
        # Turn 1: White (2 stones) -> idx 1, 2
        # Turn 2: Black (2 stones) -> idx 3, 4
        # ...
        # Turn T: 
        #   if T == 0: Black
        #   else: if T % 2 != 0: White else Black
        
        # 让我们用 turn_count 来算
        if chosen_start_idx == 0:
            current_player_color = 'black'
            num_stones_to_predict = 1
        else:
            # 计算这是第几个 "2子回合"
            # idx 1 -> Turn 1 (White)
            # idx 3 -> Turn 2 (Black)
            # idx 5 -> Turn 3 (White)
            turn_num = (chosen_start_idx + 1) // 2
            if turn_num % 2 != 0:
                current_player_color = 'white'
            else:
                current_player_color = 'black'
            num_stones_to_predict = 2

        # 构建 Target Value
        if winner_color == 'draw':
            value_target = 0.0
        elif winner_color == current_player_color:
            value_target = 1.0
        else:
            value_target = -1.0
            
        # 构建 Input Feature
        # 我们需要重演到 chosen_start_idx 之前
        input_tensor = self.build_features(moves, chosen_start_idx, current_player_color)
        
        # 构建 Policy Target (Indices)
        # 361 作为 Padding Index (表示没有第二步)
        target_move1 = 361
        target_move2 = 361
        
        # 获取实际走的子
        # moves list: [..., move1, move2, ...]
        if num_stones_to_predict >= 1 and chosen_start_idx < total_moves:
            coord = self._coord_to_index(moves[chosen_start_idx])
            if coord: target_move1 = coord[0] * 19 + coord[1]
            
        if num_stones_to_predict >= 2 and chosen_start_idx + 1 < total_moves:
            coord = self._coord_to_index(moves[chosen_start_idx + 1])
            if coord: target_move2 = coord[0] * 19 + coord[1]
        
        # 如果第一步就是空的，说明数据有问题，跳过
        if target_move1 == 361:
             return self.__getitem__((idx + 1) % len(self))
             
        return torch.from_numpy(input_tensor).float(), torch.tensor(target_move1).long(), torch.tensor(target_move2).long(), torch.tensor([value_target]).float()

    def build_features(self, moves, current_idx, current_color):
        """
        构建 17 个特征平面
        0: Self stones
        1: Opponent stones
        2-9: History (Last 1, 2, 3, 4 moves)
        16: Color (0 for Black, 1 for White) - No, AlphaZero uses All-1 for Black, All-0 for White usually or vice versa.
             Let's use: All 1 if current player is Black, All 0 if White.
        """
        features = np.zeros((17, 19, 19), dtype=np.float32)
        
        # 填充棋盘历史
        # 我们需要快速知道哪些子是黑，哪些是白，哪些是最近下的
        # 为了速度，我们倒序遍历 moves[:current_idx]
        
        # 初始化棋盘状态
        # 0: empty, 1: black, 2: white
        board = np.zeros((19, 19), dtype=int)
        
        # 简单的重演一遍拿到最终状态 (Self/Opponent)
        # 优化：其实不需要重演整个过程来分黑白，可以直接根据 index 奇偶性判断
        # Move 0: Black
        # Move 1,2: White
        # Move 3,4: Black ...
        
        for i in range(current_idx):
            coord = self._coord_to_index(moves[i])
            if coord is None: continue
            r, c = coord
            
            # Determine color of this move
            if i == 0:
                color = 1 # Black
            else:
                turn_n = (i + 1) // 2
                if turn_n % 2 != 0:
                    color = 2 # White
                else:
                    color = 1 # Black
            
            board[r, c] = color
            
        # Plane 0 & 1: Self & Opponent
        self_color = 1 if current_color == 'black' else 2
        opp_color = 2 if current_color == 'black' else 1
        
        features[0] = (board == self_color).astype(float)
        features[1] = (board == opp_color).astype(float)
        
        # Plane 2-9: History
        # 倒推 4 个“动作组”。一个动作组可能是1颗子(开局)或2颗子。
        # Recent 1: moves[current_idx-1] (and -2 if same turn)
        # ...
        
        # 实际上，History Plane 最好是记录“那个回合新增的子”。
        # 我们倒序回顾
        plane_idx = 2
        history_ptr = current_idx - 1
        
        # 记录过去 8 步（steps），而不是 turns，这样更细粒度
        steps_recorded = 0
        while history_ptr >= 0 and steps_recorded < 8:
            coord = self._coord_to_index(moves[history_ptr])
            if coord:
                r, c = coord
                # Plane 2 is T-1, Plane 3 is T-2 ...
                features[plane_idx + steps_recorded][r, c] = 1.0
            
            steps_recorded += 1
            history_ptr -= 1
            
        # Plane 16: Color
        if current_color == 'black':
            features[16] = 1.0 # Black player
        else:
            features[16] = 0.0 # White player
            
        return features

if __name__ == "__main__":
    # Test
    ds = Connect6Dataset('../processed/connect6_cleaned.csv')
    print(f"Dataset size: {len(ds)}")
    
    inp, pol, val = ds[0]
    print(f"Input shape: {inp.shape}")
    print(f"Policy shape: {pol.shape}, Sum: {pol.sum()}") # Sum should be 1 or 2
    print(f"Value: {val}")
