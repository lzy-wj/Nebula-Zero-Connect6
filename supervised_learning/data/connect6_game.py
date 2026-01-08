import numpy as np

class Connect6Game:
    def __init__(self):
        self.board_size = 19
        # 0: 空, 1: 黑棋, -1: 白棋
        # 使用 int8 节省内存
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.moves = []
        self.current_player = 1  # 黑棋先手
        self.winner = 0
        self.move_count = 0
        
        # 预计算四个方向的增量: 横, 竖, 正斜, 反斜
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def reset(self):
        self.board.fill(0)
        self.moves = []
        self.current_player = 1
        self.winner = 0
        self.move_count = 0

    def _parse_coord(self, coord_str):
        """将 'j10' 转换为 (row_idx, col_idx)"""
        try:
            # 简单的手动解析通常比正则快
            coord_str = coord_str.lower().strip()
            if not coord_str: return None
            col_char = coord_str[0]
            
            if not ('a' <= col_char <= 's'): return None
            
            row_str = coord_str[1:]
            if not row_str.isdigit(): return None
            
            col = ord(col_char) - ord('a')
            row = int(row_str) - 1
            
            if not (0 <= col < self.board_size and 0 <= row < self.board_size):
                return None
            return row, col
        except:
            return None

    def _to_coord(self, row, col):
        """将 (row, col) 转换为 'j10'"""
        col_char = chr(col + ord('a'))
        row_str = str(row + 1)
        return f"{col_char}{row_str}"

    def check_win_at(self, r, c, player):
        """
        高效检查：只检查以 (r, c) 为中心，四个方向是否有6连。
        """
        # 边界检查优化：直接在循环内判断
        bs = self.board_size
        b = self.board
        
        for dr, dc in self.directions:
            count = 1 # 当前这颗子
            
            # 向正方向延伸
            r_curr, c_curr = r + dr, c + dc
            while 0 <= r_curr < bs and 0 <= c_curr < bs and b[r_curr, c_curr] == player:
                count += 1
                r_curr += dr
                c_curr += dc
                if count >= 6: return True

            # 向反方向延伸
            r_curr, c_curr = r - dr, c - dc
            while 0 <= r_curr < bs and 0 <= c_curr < bs and b[r_curr, c_curr] == player:
                count += 1
                r_curr -= dr
                c_curr -= dc
                if count >= 6: return True
            
            if count >= 6:
                return True
        return False

    def solve_immediate_win(self, player, stones_left):
        """
        检测是否能通过接下来的 stones_left 步直接获胜。
        优化策略：只检查现有棋子周围的空位。
        只做1步搜索（即使stones_left=2），因为2步搜索太慢且对于数据清洗不是必须的。
        我们假设如果能赢，第一步就能形成威胁或者直接连成。
        """
        if stones_left <= 0:
            return None

        # 获取当前玩家所有棋子的位置
        # np.argwhere 比较慢，如果维护一个列表会更快，但重置麻烦。
        # 考虑到这是批处理，用 numpy 还可以。
        player_stones_indices = np.argwhere(self.board == player)
        if len(player_stones_indices) == 0:
            return None

        # 使用集合去重候选点
        candidates = set()
        bs = self.board_size
        
        # 只检查已有棋子周围半径为5的区域（因为连6最远延伸5格）
        # 但简单点，只检查已有棋子紧邻的空位，或者延伸线上的空位。
        # 为了性能平衡，我们检查已有棋子周围 4 格内的空位（暴力一点但范围小）
        
        # 更高效的方法：
        # 只有当某个方向上已经有了至少1个子，才值得去尝试填空。
        # 我们遍历所有己方棋子，沿着4个方向检查空位。
        
        for r, c in player_stones_indices:
            for dr, dc in self.directions:
                # 往前查5格
                for k in range(1, 6):
                    nr, nc = r + k*dr, c + k*dc
                    if 0 <= nr < bs and 0 <= nc < bs:
                        if self.board[nr, nc] == 0:
                            candidates.add((nr, nc))
                        elif self.board[nr, nc] != player:
                            break # 被对方挡住，这方向没戏了（对于这颗子来说）
                    else:
                        break
                # 往后查5格
                for k in range(1, 6):
                    nr, nc = r - k*dr, c - k*dc
                    if 0 <= nr < bs and 0 <= nc < bs:
                        if self.board[nr, nc] == 0:
                            candidates.add((nr, nc))
                        elif self.board[nr, nc] != player:
                            break 
                    else:
                        break

        # 检查候选点
        for r, c in candidates:
            self.board[r, c] = player
            if self.check_win_at(r, c, player):
                self.board[r, c] = 0 # 回溯
                return [(r, c)]
            self.board[r, c] = 0 # 回溯
            
        return None

    def apply_move_sequence(self, move_sequence_str):
        """
        解析CSV中的移动字符串并重演棋局。
        返回:
            valid (bool): 棋局逻辑是否合法
            cleaned_moves (list): 清洗后的移动列表
            result_winner (str): 'black', 'white', 'draw', 或 None
        """
        self.reset()
        if not isinstance(move_sequence_str, str):
            return False, [], None

        # 预处理：直接分割，去空
        moves = [m.strip() for m in move_sequence_str.split(',') if m.strip()]
        
        if not moves:
            return False, [], None

        idx = 0
        total_moves = len(moves)
        cleaned_moves = []
        
        while idx < total_moves:
            is_first_turn = (self.move_count == 0)
            stones_count = 1 if is_first_turn else 2
            
            stones_played = 0
            while stones_played < stones_count:
                if idx >= total_moves:
                    break
                
                coord_str = moves[idx]
                coord = self._parse_coord(coord_str)
                
                if coord is None:
                    # 格式错误，停止解析，但保留之前的
                    break
                
                r, c = coord
                if self.board[r, c] != 0:
                    # 位置被占，停止解析
                    break
                
                self.board[r, c] = self.current_player
                cleaned_moves.append(coord_str)
                stones_played += 1
                idx += 1
                
                # 增量检查胜负
                if self.check_win_at(r, c, self.current_player):
                    return True, cleaned_moves, ('black' if self.current_player == 1 else 'white')
            
            # 检查这一手是否完整
            if stones_played < stones_count:
                # 残局，尝试修复
                needed = stones_count - stones_played
                win_moves = self.solve_immediate_win(self.current_player, needed)
                if win_moves:
                    for r, c in win_moves:
                        self.board[r, c] = self.current_player
                        cleaned_moves.append(self._to_coord(r, c))
                    return True, cleaned_moves, ('black' if self.current_player == 1 else 'white')
                else:
                    # 无法修复，作为未完成局返回
                    return True, cleaned_moves, None
            
            # 换手
            self.current_player = -self.current_player
            self.move_count += 1
        
        # 棋谱走完了，还没分胜负
        # 看看能不能帮下一手赢（修复只差一步的情况）
        stones_to_place = 1 if self.move_count == 0 else 2
        win_moves = self.solve_immediate_win(self.current_player, stones_to_place)
        if win_moves:
             for r, c in win_moves:
                 self.board[r, c] = self.current_player
                 cleaned_moves.append(self._to_coord(r, c))
             return True, cleaned_moves, ('black' if self.current_player == 1 else 'white')
        
        # 检查平局
        if len(cleaned_moves) == 361:
            return True, cleaned_moves, 'draw'

        return True, cleaned_moves, None
