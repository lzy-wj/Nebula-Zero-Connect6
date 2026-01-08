import re
import datetime

BOARD_SIZE = 19

class C6SGFHandler:
    """
    Handles the specific Connect6 competition format:
    {{C6}[Black][White][Result][Date Place][Event];B(J,10);W(I,11)...}
    
    坐标系统：
    - 内部: row=0 是顶部, row=18 是底部
    - 显示: 行号从下往上 1-19 (row=0 -> 19, row=18 -> 1)
    - 列: A-S (col=0 -> A, col=18 -> S)
    """
    def __init__(self):
        pass

    @staticmethod
    def format_move(move_idx, row, col, player, mark=None):
        """
        Convert (row, col) to format like B(J,19) or B(J,19)MARK[1].
        内部 row=0 是顶部 -> 显示为 19
        内部 row=18 是底部 -> 显示为 1
        Col: 0-18 -> A-S
        mark: Optional MARK value (-2 to 2), None means no annotation
        """
        col_char = chr(ord('A') + col)
        row_num = BOARD_SIZE - row  # row=0 -> 19, row=18 -> 1
        p_char = 'B' if player == 1 else 'W'
        move_str = f"{p_char}({col_char},{row_num})"
        
        # Add MARK annotation if provided and non-zero
        if mark is not None and mark != 0:
            move_str += f"MARK[{mark}]"
        
        return move_str

    @staticmethod
    def parse_move_string(move_str):
        """
        Parse "B(J,19)" -> (row, col, player)
        显示行号 19 -> 内部 row=0
        显示行号 1 -> 内部 row=18
        """
        match = re.match(r"([BW])\(([A-S]),(\d+)\)", move_str)
        if not match:
            return None
        
        p_char, col_char, row_str = match.groups()
        player = 1 if p_char == 'B' else -1
        col = ord(col_char) - ord('A')
        display_row = int(row_str)
        row = BOARD_SIZE - display_row  # 显示 19 -> row=0, 显示 1 -> row=18
        return (row, col, player)

    def save_game_with_info(self, game_info, moves, move_evaluations=None, save_dir="assets"):
        """
        按照比赛标准格式保存棋谱。
        
        文件名格式: C6-先手参赛队 B vs 后手参赛队 W-先（后）手胜-比赛时间地点-赛事名称.txt
        文件内容格式: {[C6][先手参赛队 B][后手参赛队 W][先手胜][2017.07.29 14:00 重庆][2017 CCGC];B(J,10);W(I,11)MARK[1]...}
        
        :param game_info: dict with keys: black_team, white_team, location, event, winner, time
        :param moves: List of (row, col, player)
        :param move_evaluations: Optional dict mapping move_idx -> mark value (-2 to 2)
        :param save_dir: 保存目录，相对于 local/
        :return: 保存的文件路径，失败返回 None
        """
        import os
        
        black_team = game_info['black_team']
        white_team = game_info['white_team']
        location = game_info['location']
        event_name = game_info['event']
        winner = game_info['winner']  # 1=黑胜, -1=白胜, 0=平局
        game_time = game_info['time']
        
        # 结果字符串
        if winner == 1:
            result_str = "先手胜"
        elif winner == -1:
            result_str = "后手胜"
        elif winner == 2:
            result_str = "流局"
        else:
            result_str = "平局"
        
        # 时间格式化
        time_str = game_time.strftime("%Y.%m.%d %H:%M")
        
        # 文件名格式: C6-先手参赛队 B vs 后手参赛队 W-先（后）手胜-比赛时间地点-赛事名称.txt
        # 清理文件名中的非法字符
        def clean_filename(s):
            return s.replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
        
        # 文件名中时间格式用 HH.MM (xx.xx 形式)
        filename_time = game_time.strftime('%Y.%m.%d %H.%M')
        filename = f"C6-{clean_filename(black_team)} B vs {clean_filename(white_team)} W-{result_str}-{filename_time} {clean_filename(location)}-{clean_filename(event_name)}.txt"
        
        # 构建文件内容
        # 格式: {[C6][先手参赛队 B][后手参赛队 W][先手胜][2017.07.29 14:00 重庆][2017 CCGC];B(J,10);W(I,11)...}
        # 队名后面要加 B 和 W 标识
        header = f"{{[C6][{black_team} B][{white_team} W][{result_str}][{time_str} {location}][{event_name}]"
        
        # 着法
        move_strs = []
        for idx, (r, c, p) in enumerate(moves):
            # Get MARK value for this move if available
            mark = None
            if move_evaluations and idx in move_evaluations:
                mark = move_evaluations[idx]
            move_strs.append(self.format_move(idx, r, c, p, mark))
        
        content = header + ";" + ";".join(move_strs) + "}"
        
        # 确保保存目录存在
        # 获取当前文件所在目录的父目录 (local/)
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_save_dir = os.path.join(current_dir, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)
        
        filepath = os.path.join(full_save_dir, filename)
        
        try:
            # 使用 GB2312 编码保存
            with open(filepath, 'w', encoding='gb2312') as f:
                f.write(content)
            print(f"棋谱已保存: {filepath}")
            return filepath
        except UnicodeEncodeError:
            # 如果 GB2312 编码失败，尝试 GBK（GB2312 的超集）
            try:
                with open(filepath, 'w', encoding='gbk') as f:
                    f.write(content)
                print(f"棋谱已保存 (GBK): {filepath}")
                return filepath
            except Exception as e:
                print(f"Save Error (GBK): {e}")
                return None
        except Exception as e:
            print(f"Save Error: {e}")
            return None

    def save_game(self, filepath, black_name, white_name, winner, moves, date_place="Unknown", event="Nebula Cup"):
        """
        旧版保存方法，保持向后兼容。
        moves: List of (row, col, player)
        winner: 'Black', 'White', or 'Draw'
        """
        # Result code
        if winner == 'Black':
            res = '先手胜'
        elif winner == 'White':
            res = '后手胜'
        else:
            res = '平局'
            
        current_time = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")
        full_date = f"{current_time} {date_place}"
        
        header = f"{{[C6][{black_name}][{white_name}][{res}][{full_date}][{event}]"
        
        # Moves
        move_strs = []
        for idx, (r, c, p) in enumerate(moves):
            move_strs.append(self.format_move(idx, r, c, p))
            
        content = header + ";" + ";".join(move_strs) + "}"
        
        try:
            with open(filepath, 'w', encoding='gb2312') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Save Error: {e}")
            return False

    def load_game(self, filepath):
        """
        加载棋谱文件，支持 GB2312、GBK 和 UTF-8 编码。
        """
        content = None
        
        # 尝试不同编码
        for encoding in ['gb2312', 'gbk', 'utf-8']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if content is None:
            print(f"无法读取文件: {filepath}")
            return []
            
        # Very basic parser
        # Extract header info
        # Extract moves
        moves = []
        parts = content.split(';')
        for p in parts[1:]: # Skip header part before first ;
            p = p.strip().replace('}', '')
            m = self.parse_move_string(p)
            if m:
                moves.append(m)
        return moves
