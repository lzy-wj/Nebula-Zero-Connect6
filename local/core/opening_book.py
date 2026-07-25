"""
开局库模块 - 用于查询和使用开局库
"""

import json
import os
from typing import Optional, List, Dict

class OpeningBook:
    """
    开局库类
    """
    
    def __init__(self, book_path: Optional[str] = None):
        """
        初始化开局库
        
        Args:
            book_path: 开局库 JSON 文件路径
        """
        self.book: Dict[str, dict] = {}
        self.enabled = True
        
        if book_path and os.path.exists(book_path):
            self.load(book_path)
    
    def load(self, book_path: str):
        """加载开局库"""
        try:
            with open(book_path, 'r', encoding='utf-8') as f:
                self.book = json.load(f)
            print(f"[OpeningBook] 加载了 {len(self.book)} 条开局记录")
        except Exception as e:
            print(f"[OpeningBook] 加载失败: {e}")
            self.book = {}
    
    def save(self, book_path: str):
        """保存开局库"""
        with open(book_path, 'w', encoding='utf-8') as f:
            json.dump(self.book, f, indent=2, ensure_ascii=False)
    
    def get_board_hash(self, moves: List[int]) -> str:
        """
        获取局面的哈希字符串
        保持着法顺序，因为顺序决定了当前轮到谁下
        """
        return ",".join(map(str, moves))
    
    def query(self, moves: List[int]) -> Optional[int]:
        """
        查询开局库
        
        Args:
            moves: 当前局面的着法历史
        
        Returns:
            最佳着法索引，如果没有找到返回 None
        """
        if not self.enabled or not self.book:
            return None
        
        position_hash = self.get_board_hash(moves)
        
        if position_hash in self.book:
            info = self.book[position_hash]
            move = info['move'] if isinstance(info, dict) else info
            return move
        
        return None
    
    def query_random(self, moves: List[int]) -> Optional[dict]:
        """
        从开局库中随机选择一个着法
        
        Args:
            moves: 当前局面的着法历史
        
        Returns:
            随机选中的着法信息 {'move': int, 'win_rate': float, 'games': int}
        """
        import random
        
        if not self.enabled or not self.book:
            return None
        
        position_hash = self.get_board_hash(moves)
        print(f"[OpeningBook] Querying hash: '{position_hash}'")
        
        if position_hash not in self.book:
            print(f"[OpeningBook] Miss: '{position_hash}'")
            return None
        
        info = self.book[position_hash]
        
        # 收集所有候选着法
        candidates = [{
            'move': info['move'],
            'move_coord': info.get('move_coord', ''),
            'win_rate': info.get('win_rate', 0.5),
            'games': info.get('games', 0)
        }]
        
        # 添加备选着法
        alternatives = info.get('alternatives', [])
        for alt in alternatives:
            candidates.append({
                'move': alt['move'],
                'move_coord': alt.get('move_coord', ''),
                'win_rate': alt.get('win_rate', 0.5),
                'games': alt.get('games', 0)
            })
        
        # 按胜率加权随机选择
        # 使用胜率作为权重
        weights = [c['win_rate'] for c in candidates]
        selected = random.choices(candidates, weights=weights, k=1)[0]
        
        return selected
    
    def get_info(self, moves: List[int]) -> Optional[dict]:
        """
        获取开局库中的详细信息
        
        Args:
            moves: 当前局面的着法历史
        
        Returns:
            包含 move, win_rate, games 的字典
        """
        if not self.enabled or not self.book:
            return None
        
        position_hash = self.get_board_hash(moves)
        return self.book.get(position_hash)
    
    def add_record(self, moves: List[int], best_move: int, win_rate: float, games: int):
        """
        添加开局记录（用于在线学习）
        
        Args:
            moves: 当前局面
            best_move: 最佳着法
            win_rate: 胜率
            games: 对局数
        """
        position_hash = self.get_board_hash(moves)
        
        # 如果已有记录，检查是否需要更新
        if position_hash in self.book:
            existing = self.book[position_hash]
            existing_games = existing.get('games', 0)
            
            # 只有新数据更多时才更新
            if games <= existing_games:
                return
        
        self.book[position_hash] = {
            'move': best_move,
            'move_coord': self._idx_to_coord(best_move),
            'win_rate': round(win_rate, 3),
            'games': games
        }
    
    def _idx_to_coord(self, idx: int) -> str:
        """将索引转换为坐标字符串"""
        internal_row = idx // 19
        col = idx % 19
        row = 18 - internal_row + 1
        return f"{chr(ord('A') + col)}{row}"
    
    def set_enabled(self, enabled: bool):
        """启用/禁用开局库"""
        self.enabled = enabled
    
    def __len__(self):
        return len(self.book)
    
    def __bool__(self):
        return bool(self.book)


# 全局开局库实例（可选）
_global_opening_book: Optional[OpeningBook] = None

def get_opening_book() -> OpeningBook:
    """获取全局开局库实例"""
    global _global_opening_book
    if _global_opening_book is None:
        _global_opening_book = OpeningBook()
    return _global_opening_book

def init_opening_book(book_path: str) -> OpeningBook:
    """初始化全局开局库"""
    global _global_opening_book
    _global_opening_book = OpeningBook(book_path)
    return _global_opening_book
