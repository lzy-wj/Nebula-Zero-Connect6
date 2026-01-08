import numpy as np
import pandas as pd
import os

class DataRefiner:
    def __init__(self):
        # 棋盘大小
        self.size = 19
        
    def check_line(self, board, r, c, dr, dc, color):
        """
        检查某个方向的连子数和两端的空位数
        返回: (count, open_ends)
        """
        count = 1
        open_ends = 0
        
        # 正向
        nr, nc = r + dr, c + dc
        while 0 <= nr < self.size and 0 <= nc < self.size and board[nr][nc] == color:
            count += 1
            nr += dr
            nc += dc
        # 检查正向端点
        if 0 <= nr < self.size and 0 <= nc < self.size and board[nr][nc] == 0:
            open_ends += 1
            
        # 反向
        nr, nc = r - dr, c - dc
        while 0 <= nr < self.size and 0 <= nc < self.size and board[nr][nc] == color:
            count += 1
            nr -= dr
            nc -= dc
        # 检查反向端点
        if 0 <= nr < self.size and 0 <= nc < self.size and board[nr][nc] == 0:
            open_ends += 1
            
        return count, open_ends

    def analyze_move_quality(self, board_state, move_r, move_c, player):
        """
        分析这一步棋的质量
        返回: reward_bonus (float)
        """
        bonus = 0.0
        
        # 1. 检查威胁数量 (Double Threats)
        # 仅仅奖励 "双活四" 或 "双威胁"，不奖励单个活四/活五
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        max_line = 0
        threat_count = 0 # Count STRONG threats
        
        for dr, dc in directions:
            line_len, open_ends = self.check_line(board_state, move_r, move_c, dr, dc, player)
            max_line = max(max_line, line_len)
            
            # 定义更严格的 "威胁":
            # 1. 连五 (Line=5): 只要有一头能下就是巨大威胁 (open_ends >= 1，甚至 0 如果刚好凑成？不，凑成是6)
            #    注意：这里 max_line >= 6 会直接返回 1.0，所以这里只处理 <6 的情况。
            #    连五只要存在，就是威胁。
            # 2. 活四 (Line=4, Open=2): 两头都空，必须堵一头。
            # 
            # 排除:
            # - 死四 (Line=4, Open=1): 只有一头空，对手有2子，随便堵死。不威胁。
            
            if line_len == 5:
                threat_count += 1
            elif line_len == 4 and open_ends == 2:
                threat_count += 1
            
        if max_line >= 6:
            return 1.0 # 绝杀，直接给满分
            
        # 核心逻辑修改：只奖励双杀 (Double Threat)
        # 必须是有效的威胁
        if threat_count >= 2:
            bonus += 0.4 # 双杀奖励！非常高
            
        # 2. 检查是否是废棋 (孤子)
        # 如果周围 2 格内没有任何棋子，视为废棋倾向
        is_isolated = True
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0: continue
                nr, nc = move_r + dr, move_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if board_state[nr][nc] != 0:
                        is_isolated = False
                        break
            if not is_isolated: break
            
        if is_isolated:
            bonus -= 0.2 # 惩罚孤子/废棋
            
        return bonus

    def refine_csv(self, input_path, output_path):
        """
        读取 CSV，重新计算 Value，并保存
        """
        try:
            df = pd.read_csv(input_path, header=None)
            # 假设 CSV 格式: 
            # state_str, policy_dist, value, moves_history...
            # 这里我们需要解析 state 还原棋盘，这比较慢。
            # 
            # 快速方案：
            # 直接根据 winner (最后的值) 和步数来调整？
            # 或者，我们在 generate.py 里生成的时候就调用 refine 更好？
            #
            # 为了不破坏 CSV 结构，我们这里只做简单的 Value 修正：
            # 如果是短局数获胜 (<30步)，增加 Value 权重 (1.0 -> 1.1)
            # 如果是长局数平局，降低 Value
            
            # 由于在 CSV 里解析棋盘太慢，我们采用 "轻量级清洗"
            # 重点：根据游戏长度调整 Value
            
            new_rows = []
            for idx, row in df.iterrows():
                # 最后一列通常是 Value (-1, 0, 1)
                # 倒数第二列可能是 moves_count?
                # 让我们假设这是标准的 AlphaZero 格式
                # 实际上，最好是在 generate.py 刚下完棋的时候，内存里有 board 对象时做这个分析。
                # 
                # 这里我们做一个占位，真正的逻辑建议移到 generate.py 的 Game Loop 中。
                pass
                
            # 既然你是要修改 run_loop，不如我们在 run_loop 里调用一个处理函数
            # 但处理 CSV 真的很慢。
            #
            # 最优解：修改 generate.py，在生成数据落盘前，就加上 Bonus。
            pass
            
        except Exception as e:
            print(f"Refine Error: {e}")

# 既然要深度结合，我建议修改 generate.py
# 但这里为了不破坏现有结构，我提供一个独立的 Refiner 类供 run_loop 调用
# 它可以只做简单的 "胜率放大"
