"""
从训练数据中提取开局库
使用方法: 
    python extract_opening_book.py                                    # 使用默认路径
    python extract_opening_book.py --data path/to/games.csv          # 指定单个文件
    python extract_opening_book.py --data path/to/data_dir/          # 指定目录（读取所有CSV）
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from typing import List, Dict, Tuple

# 默认数据目录（支持多个）
DEFAULT_DATA_DIRS = [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "vit_resnet", "phrase4", "data", "raw"
    ),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "vit_resnet", "phrase4", "data", "buffer"
    )
]

# 默认输出路径
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core", "opening_book.json"
)

def coord_to_idx(coord: str) -> int:
    """
    将坐标字符串转换为索引 (0-360)
    例如: "j10" -> 180, "a1" -> 342, "s19" -> 18
    """
    coord = coord.strip().lower()
    col = ord(coord[0]) - ord('a')
    row = int(coord[1:]) - 1
    # 内部坐标: r=0 是顶部 (row=18), r=18 是底部 (row=0)
    internal_row = 18 - row
    return internal_row * 19 + col

def idx_to_coord(idx: int) -> str:
    """
    将索引转换为坐标字符串
    例如: 180 -> "J10"
    """
    internal_row = idx // 19
    col = idx % 19
    row = 18 - internal_row + 1
    return f"{chr(ord('A') + col)}{row}"

def parse_moves(moves_str: str) -> List[int]:
    """
    解析着法字符串为索引列表
    例如: "j10,i11,k10" -> [180, 161, 182]
    """
    if not moves_str or moves_str == "":
        return []
    
    moves = []
    for m in moves_str.split(','):
        m = m.strip()
        if m:
            try:
                moves.append(coord_to_idx(m))
            except:
                continue
    return moves

def get_board_hash(moves: List[int]) -> str:
    """
    获取局面的哈希字符串
    保持着法顺序，因为顺序决定了当前轮到谁下
    """
    return ",".join(map(str, moves))

def extract_opening_book(
    data_paths,  # 支持单个路径或路径列表
    max_moves: int = 10,
    min_games: int = 3,
    min_win_rate: float = 0.55
) -> Dict[str, int]:
    """
    从训练数据中提取开局库
    
    Args:
        data_paths: CSV 数据文件或目录路径（单个或列表）
        max_moves: 最多记录前多少步
        min_games: 某个走法至少出现多少次才记录
        min_win_rate: 最低胜率阈值
    
    Returns:
        开局库字典 {局面哈希: 最佳着法}
    """
    # 统计: position_hash -> {next_move: [wins, total]}
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
    # 统一为列表
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    # 获取所有 CSV 文件
    csv_files = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            found = glob.glob(os.path.join(data_path, "*.csv"))
            print(f"在目录 {data_path} 中找到 {len(found)} 个 CSV 文件")
            csv_files.extend(found)
        elif os.path.isfile(data_path):
            csv_files.append(data_path)
    
    if not csv_files:
        print("错误: 没有找到 CSV 文件")
        return {}
    
    total_games = 0
    
    for csv_file in csv_files:
        print(f"  处理: {os.path.basename(csv_file)}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            print(f"    读取失败: {e}")
            continue
        
        # 跳过标题行
        if rows and 'moves' in rows[0][0].lower():
            rows = rows[1:]
        
        game_count = 0
        for row in rows:
            if len(row) < 2:
                continue
            
            # CSV 模块已经正确处理了引号
            moves_str = row[0].strip()
            winner_str = row[1].strip().lower()
            
            moves = parse_moves(moves_str)
            if not moves:
                continue
            
            game_count += 1
            
            # 确定赢家
            is_black_win = 'black' in winner_str or winner_str == '1'
            
            # 记录每一步的统计
            for i in range(min(len(moves), max_moves)):
                # 当前局面 (前 i 步)
                current_position = moves[:i]
                position_hash = get_board_hash(current_position)
                
                # 下一步着法
                next_move = moves[i]
                
                # 确定这一步是谁下的
                # 规则: 第0步是黑, 第1-2步是白, 第3-4步是黑, ...
                if i == 0:
                    is_current_black = True
                else:
                    turn = (i + 1) // 2
                    is_current_black = (turn % 2 == 0)
                
                # 这一步的执行者是否赢了
                win = 1 if (is_current_black == is_black_win) else 0
                
                stats[position_hash][next_move][0] += win
                stats[position_hash][next_move][1] += 1
        
        print(f"    读取了 {game_count} 局对局")
        total_games += game_count
    
    print(f"\n总计读取了 {total_games} 局对局")
    print(f"统计了 {len(stats)} 个局面")
    
    # 生成开局库
    opening_book = {}
    top_n = 5  # 每个局面保存 top N 个着法
    confidence_threshold = 100  # 样本量达到此值时，置信度为1
    
    def calc_score(win_rate, games):
        """
        综合得分 = 胜率 × 置信度权重
        置信度权重 = min(1.0, games / confidence_threshold)
        
        例如:
        - M15: 100%胜率, 21局 → 1.0 × 0.21 = 0.21
        - L6:  87.6%胜率, 3245局 → 0.876 × 1.0 = 0.876
        这样 L6 会排在前面！
        """
        confidence = min(1.0, games / confidence_threshold)
        return win_rate * confidence
    
    for position_hash, move_stats in stats.items():
        # 收集所有满足条件的着法
        valid_moves = []
        for move, (wins, total) in move_stats.items():
            if total < min_games:
                continue
            
            win_rate = wins / total if total > 0 else 0
            
            if win_rate >= min_win_rate:
                score = calc_score(win_rate, total)
                valid_moves.append({
                    'move': move,
                    'move_coord': idx_to_coord(move),
                    'win_rate': round(win_rate, 3),
                    'games': total,
                    'score': round(score, 3)  # 用于排序
                })
        
        if valid_moves:
            # 按综合得分排序，取 top N
            valid_moves.sort(key=lambda x: x['score'], reverse=True)
            top_moves = valid_moves[:top_n]
            
            # 主着法是得分最高的
            best = top_moves[0]
            opening_book[position_hash] = {
                'move': best['move'],
                'move_coord': best['move_coord'],
                'win_rate': best['win_rate'],
                'games': best['games'],
                'alternatives': [
                    {k: v for k, v in m.items() if k != 'score'}  # 移除 score 字段
                    for m in top_moves[1:]
                ] if len(top_moves) > 1 else []
            }
    
    print(f"生成了 {len(opening_book)} 条开局记录")
    return opening_book

def main():
    parser = argparse.ArgumentParser(description='从训练数据中提取开局库')
    parser.add_argument('--data', type=str, default=None, 
                        help='训练数据路径 (文件或目录)，可多次指定')
    parser.add_argument('--output', type=str, default=None, 
                        help=f'输出文件路径，默认: {DEFAULT_OUTPUT}')
    parser.add_argument('--max_moves', type=int, default=10, help='最多记录前多少步')
    parser.add_argument('--min_games', type=int, default=20, help='某个走法至少出现多少次才记录')
    parser.add_argument('--min_win_rate', type=float, default=0.60, help='最低胜率阈值')
    
    args = parser.parse_args()
    
    # 使用默认路径或用户指定路径
    if args.data:
        data_paths = [args.data]
    else:
        # 使用所有存在的默认目录
        data_paths = [d for d in DEFAULT_DATA_DIRS if os.path.exists(d)]
    
    output_path = args.output or DEFAULT_OUTPUT
    
    if not data_paths:
        print(f"错误: 找不到任何数据路径")
        print(f"请将训练数据放到: {DEFAULT_DATA_DIRS}")
        return
    
    print(f"数据路径: {data_paths}")
    print(f"输出路径: {output_path}")
    print()
    
    # 直接传入所有路径，统一统计
    opening_book = extract_opening_book(
        data_paths,
        max_moves=args.max_moves,
        min_games=args.min_games,
        min_win_rate=args.min_win_rate
    )
    
    if not opening_book:
        print("警告: 没有生成任何开局记录")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(opening_book, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 开局库已保存到: {output_path}")
    
    # 显示一些示例
    print("\n示例开局记录:")
    for i, (pos, info) in enumerate(list(opening_book.items())[:5]):
        print(f"  局面: {pos or '空盘'}")
        print(f"    最佳着法: {info['move_coord']} (胜率 {info['win_rate']:.1%}, {info['games']} 局)")

if __name__ == "__main__":
    main()
