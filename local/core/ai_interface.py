from PyQt5.QtCore import QObject, pyqtSignal, QThread
import time
import numpy as np
import torch
import sys
import os
from queue import Queue, Empty

# Adapt path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_mcts import MCTSEngine, MultiMCTSEngine # Modified copy in local folder
from mcts_strategy import MCTSStrategy, MCTSConfig, SearchMode, SearchConfig
from opening_book import OpeningBook


BOARD_SIZE = 19

def format_move_coord(r, c, pad=False):
    """
    将内部坐标 (r, c) 转换为标准棋盘坐标字符串。
    内部: r=0 是顶部, r=18 是底部
    显示: 行号从下往上 1-19, 列字母 A-S
    例如: (0, 9) -> "J19", (18, 0) -> "A1"
    pad: 是否填充到固定长度（3字符），避免 UI 抖动
    """
    col_char = chr(ord('A') + c)
    row_num = BOARD_SIZE - r  # r=0 -> 19, r=18 -> 1
    coord = f"{col_char}{row_num}"
    if pad and len(coord) < 3:
        coord = coord + " "  # L9 -> "L9 "
    return coord

class AIWorker(QThread):
    """
    Background thread for AI thinking.
    Supports:
    - Normal Search (My Turn)
    - Pondering (Opponent Turn) with Tree Reuse
    - Stateful MCTS (Tree reuse)
    """
    update_stats = pyqtSignal(dict) # sim_count, win_rate, depth, pv

    decision_made = pyqtSignal(int, int, int) # r, c, game_id

    def __init__(self, engine_path, device_id=0):
        super().__init__()
        self.engine_path = engine_path
        self.device = f"cuda:{device_id}"
        self.mcts = None
        self.running = True
        
        # Command Queue: (Command, Args)
        # Commands: 'RESET', 'MOVE', 'THINK', 'PARAM', 'PONDER_TOGGLE', 'SET_AI_COLOR'
        self.queue = Queue()
        
        self.pondering_enabled = False
        self.simulations = 12000  # 默认 12000
        # Increased default batch size to 32 for RTX 4060
        # Force single thread for deterministic behavior (OpenMP scheduling noise avoidance)
        self.params = {'batch': 32, 'threads': 8}
        
        # AI 颜色: 1=黑, -1=白
        self.ai_color = -1  # 默认 AI 执白
        
        # Pondering 状态
        self.ponder_mode = False          # 是否正在 Ponder
        self.ponder_phase = 0             # Ponder 阶段: 0=未开始, 1=预测第一子, 2=预测第二子
        self.ponder_moves1 = []           # 第一子 Top-K 预测 [(move_idx, prob), ...]
        self.ponder_move = -1             # 兼容旧代码：当前预测的着法
        self.ponder_moves = []            # 兼容旧代码：当前预测列表
        self.opponent_turn = False        # 是否轮到对手（只有对手回合才 Ponder）
        self.opponent_stones_in_turn = 0  # 对手当前回合已下的子数
        self.ponder_actual_move1 = -1     # 对手实际下的第一子（用于最终输出）
        self.opponent_turn_total = 0      # 对手本回合需要下的子数（黑第一手=1，其他=2）
        
        # === 5 路 Ponder 状态 ===
        # 每路: move1_topN + move2_top1 (基于该第一子的局面)
        # 算力分配: 35%, 25%, 18%, 13%, 9%
        self.ponder_paths = []  # 3 条路径，每条 {'move1': -1, 'move2': -1, 'sims': 0, 'moves2_list': []}
        self.ponder_path_ratios = [0.50, 0.30, 0.20]  # 算力分配比例（集中到前3路）
        self.ponder_num_paths = 3         # 路径数量（减少分散）
        self.ponder_move2_top_k = 5       # 每路第二子预测数量
        self.ponder_current_path = 0      # 当前搜索的路径索引 (0-4)
        self.ponder_total_sims = 0        # 总模拟次数
        self.ponder_batch_sims = 500      # 批量搜索次数（减少切换开销）
        self.ponder_path_sims_in_batch = 0  # 当前路径在本批次中已搜索的次数
        
        # 着法历史（用于 Ponder 未命中时恢复状态）
        self.move_history = []
        
        # Strategy Config
        self.search_mode = SearchMode.DYNAMIC
        self.strategy = None
        self.dynamic_thinking = False
        self.deep_thinking_enabled = True  # 默认开启深度思考
        

        
        # 必杀第二子（两子必杀时存储第二子）
        self.pending_kill_move = None
        
        # Ponder 统计（每局重置）
        self.ponder_stats = {'hit': 0, 'miss': 0, 'partial': 0}
        
        # === 多实例 Ponder 状态 (新增) ===
        self.multi_mcts = None  # MultiMCTSEngine 实例
        self.ponder_active_instance = None  # 当前活跃的 Ponder 实例
        self.ponder_use_multi_instance = True  # 是否使用多实例 Ponder
        
        # === 开局库 ===
        self.opening_book = None  # OpeningBook 实例
        self.opening_book_enabled = False  # 是否启用开局库（由 UI checkbox 控制）
        
    def init_engine(self):
        try:
            self.mcts = MCTSEngine(self.engine_path, device=self.device)
            self.mcts.set_params(self.params['batch'], self.params['threads'])
            
            # 初始化多实例引擎 (用于 Ponder)
            try:
                self.multi_mcts = MultiMCTSEngine(
                    num_instances=3,
                    batch_size=self.params['batch'],
                    num_threads=self.params['threads']
                )
                if self.multi_mcts.is_supported():
                    print("✅ [AI] Multi-instance MCTS initialized for pondering")
                    # Use same pruning as main search (K=30 fixed)
                    self.multi_mcts.set_pruning_k(30)
                    
                    # === 关键修复：预热所有 Ponder 实例 ===
                    # 确保它们处于活跃状态，避免第一次被征用时出现冷启动问题
                    print("🔥 [AI] Warming up Ponder instances...")
                    for i in range(self.multi_mcts.num_instances):
                        inst = self.multi_mcts.get_instance(i)
                        if inst:
                            # 必须设置随机种子以确保行为可控
                            try:
                                inst.set_random_seed(12345 + i)
                            except Exception:
                                pass
                            
                            try:
                                inst.run_simulations(10) # 预热
                                inst.reset()             # 重置回初始状态
                            except Exception as e:
                                print(f"⚠️ [AI] Ponder warmup failed for inst {i}: {e}")
                    print("🔥 [AI] Ponder instances warmed up.")
                else:
                    print("⚠️ [AI] Multi-instance not supported, using single-instance ponder")
                    self.ponder_use_multi_instance = False
            except Exception as e:
                print(f"⚠️ [AI] Failed to init multi-instance MCTS: {e}")
                self.ponder_use_multi_instance = False
            
            # 初始化后立即设置随机种子，确保首次和后续状态一致
            import time as time_module
            
            # 3. 关键修复：执行一次真实的极小规模搜索
            # 这会打通 Python -> C++ -> Python Callback -> C++ 的完整数据通路
            # 并消耗掉第一次调用可能出现的 "Zero Policy" 问题
            self.mcts.set_random_seed(12345)
            self.mcts.run_simulations(10)
            self.mcts.reset()

            # 初始化后立即设置随机种子
            import time as time_module
            self.mcts.set_random_seed(int(time_module.time() * 1000) % 10000000)
            
            # === 初始化开局库 ===
            try:
                book_path = os.path.join(os.path.dirname(__file__), 'opening_book.json')
                if os.path.exists(book_path):
                    self.opening_book = OpeningBook(book_path)
                    print(f"📖 [AI] 开局库加载完成: {len(self.opening_book)} 条记录")
                else:
                    print("⚠️ [AI] 开局库文件不存在，跳过加载")
            except Exception as e:
                print(f"⚠️ [AI] 开局库加载失败: {e}")
            
            print("🔥 [AI] 引擎初始化完成")
            
            return True
        except Exception as e:
            print(f"AI Init Failed: {e}")
            return False

    def reset_game(self, ai_color=None):
        # 先清空队列中的旧命令，避免旧的 THINK 请求导致状态混乱
        self.flush_commands()
        self.queue.put(('RESET', ai_color))  # 传入 ai_color 确保重置时同步设置

    def flush_commands(self):
        # Drain pending commands to avoid stale THINK requests after undo.
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass

    def notify_move(self, move_idx, is_same_turn_second=False):
        """
        Notify AI of a move (player or AI) to update state.
        move_idx: 0-360
        is_same_turn_second: True if this is the second stone of the same turn (for tree reuse)
        """
        self.queue.put(('MOVE', (move_idx, is_same_turn_second)))

    def request_move(self, moves_history, player_to_move, game_id, time_limit=None):
        # We assume state is synced via notify_move, but for safety/robustness
        # we can also support full replay if needed. 
        # For now, we trust the command stream.
        self.queue.put(('THINK', game_id))

    def finish_thinking(self):
        """Force AI to stop thinking and play the current best move."""
        self.queue.put(('FINISH_THINKING', None))

    def update_params(self, batch, threads, sims, dynamic=False, temperature=0.0):
        self.queue.put(('PARAM', (batch, threads, sims, dynamic, temperature)))
        
    def set_ponder(self, enabled):
        self.queue.put(('PONDER_TOGGLE', enabled))

    def set_ai_color(self, color):
        """设置 AI 的颜色: 1=黑, -1=白"""
        self.queue.put(('SET_AI_COLOR', color))

    def set_opponent_turn(self, is_opponent_turn):
        """设置是否轮到对手（用于控制 Ponder）"""
        self.queue.put(('SET_OPPONENT_TURN', is_opponent_turn))

    def set_deep_thinking(self, enabled):
        """设置是否启用深度思考"""
        self.queue.put(('SET_DEEP_THINKING', enabled))

    def set_opening_book_enabled(self, enabled):
        """设置是否启用开局库"""
        self.opening_book_enabled = enabled

    def load_game_state(self, moves_list):
        """线程安全地加载游戏状态"""
        self.queue.put(('LOAD_GAME_STATE', moves_list))

    def request_state_verify(self, ui_board_flat, ui_current_player):
        """
        请求状态校验。
        :param ui_board_flat: UI 棋盘的扁平化列表 (361 元素)
        :param ui_current_player: UI 当前玩家 (1=黑, -1=白)
        """
        self.queue.put(('VERIFY_STATE', (ui_board_flat, ui_current_player)))

    def _reset_ponder_state(self):
        """重置 Ponder 相关状态"""
        self.ponder_mode = False
        self.ponder_phase = 0
        self.ponder_move = -1
        self.ponder_moves = []
        self.ponder_moves1 = []
        # 3 路 Ponder 状态
        self.ponder_paths = []
        self.ponder_current_path = 0
        self.ponder_total_sims = 0
        self.ponder_path_sims_in_batch = 0  # 重置批量计数器
        # 阶段 3 状态
        if hasattr(self, 'ponder_hit_path'):
            delattr(self, 'ponder_hit_path')
        if hasattr(self, 'ponder_move2_sims'):
            self.ponder_move2_sims = {}
        if hasattr(self, 'ponder_move2_current'):
            delattr(self, 'ponder_move2_current')

    def stop(self):
        self.running = False
        self.queue.put(('STOP', None))
        self.wait()

    def run(self):
        if not self.mcts:
            if not self.init_engine():
                return

        current_tree_sims = 0 # Track sims done on current root

        while self.running:
            # Pondering Loop
            try:
                # If pondering, don't block. If not, block until command.
                if self.pondering_enabled:
                    cmd_data = self.queue.get_nowait()
                else:
                    cmd_data = self.queue.get()
            except Empty:
                # No command, time to PONDER!
                # 只有在对手回合且启用 Ponder 时才开始
                if self.pondering_enabled and self.opponent_turn and not self.ponder_mode:
                    # 判断对手本回合需要下几子
                    is_black_first = (len(self.move_history) == 0)
                    self.opponent_turn_total = 1 if is_black_first else 2
                    
                    # === 真正的多树 Ponder：用 MultiMCTSEngine ===
                    # 步骤 1: 用主引擎获取 Top-3 第一子预测
                    self.mcts.run_simulations(simulations=300)
                    policy = self.mcts.get_policy()
                    current_wr = self.mcts.get_win_rate(ai_color=self.ai_color)

                    top_indices = np.argsort(policy)[::-1][:3]
                    self.ponder_moves1 = [(int(idx), float(policy[idx])) for idx in top_indices]
                    self.ponder_moves = self.ponder_moves1
                    
                    if len(self.ponder_moves1) >= 3 and self.multi_mcts and self.multi_mcts.is_supported():
                        # === 多实例 Ponder 初始化 ===
                        
                        # 初始化 ponder_paths 用于兼容 MOVE 命令处理
                        self.ponder_paths = []
                        for i in range(3):
                            move1 = self.ponder_moves1[i][0]
                            move1_prob = self.ponder_moves1[i][1]
                            self.ponder_paths.append({
                                'move1': move1,
                                'move1_prob': move1_prob,
                                'move2': -1,
                                'sims': 0,
                                'moves2_list': []
                            })
                        
                        if self.opponent_turn_total == 2:
                            # 为 3 个实例设置不同的第一子并预测第二子
                            for i in range(3):
                                inst = self.multi_mcts.get_instance(i)
                                move1 = self.ponder_moves1[i][0]
                                inst.sync_from_moves(self.move_history)
                                inst.play_move(move1)  # 实例 i 预测第一子为 move1[i]
                                
                                # 搜索 200 次预测第二子 Top-3
                                inst.run_simulations(200)
                                policy2 = inst.get_policy()
                                top3_move2 = np.argsort(policy2)[::-1][:3]
                                moves2_list = [(int(idx), float(policy2[idx])) for idx in top3_move2]
                                move2_top1 = moves2_list[0][0] if moves2_list else -1
                                
                                # 更新 ponder_paths
                                self.ponder_paths[i]['move2'] = move2_top1
                                self.ponder_paths[i]['moves2_list'] = moves2_list
                            
                            self.ponder_phase = 1  # 阶段 1：搜索第一子
                            self.ponder_current_instance = 0
                            
                            # 输出预测信息
                            pred_strs = []
                            for i, p in enumerate(self.ponder_paths):
                                m1 = format_move_coord(p['move1'] // 19, p['move1'] % 19)
                                m2 = format_move_coord(p['move2'] // 19, p['move2'] % 19) if p['move2'] >= 0 else "?"
                                prob = int(p['move1_prob'] * 100)
                                pred_strs.append(f"{m1}+{m2}({prob}%)")
                            print(f"🔮 [MultiPonder] 预测: {', '.join(pred_strs)}")
                        else:
                            # 黑棋第一手，只预测 1 子
                            inst = self.multi_mcts.get_instance(0)
                            move1 = self.ponder_moves1[0][0]
                            inst.sync_from_moves(self.move_history)
                            inst.play_move(move1)
                            
                            self.ponder_phase = 1
                            self.ponder_current_instance = 0
                            
                            pred_strs = [f"{format_move_coord(m // 19, m % 19)}({int(p*100)}%)" 
                                        for m, p in self.ponder_moves1]
                            print(f"🔮 [MultiPonder] 预测: {', '.join(pred_strs)}")
                        
                        self.ponder_mode = True
                        self.ponder_total_sims = 0
                        self.ponder_move = self.ponder_moves1[0][0]
                        
                        # 发送预测到 UI (convert to r,c,prob format)
                        policy_data = [(m // 19, m % 19, p) for m, p in self.ponder_moves1]
                        self.update_stats.emit({
                            'sims': 0,
                            'win_rate': current_wr,
                            'time': 0,
                            'policy': policy_data,
                            'pruning_k': 0,
                            'is_ponder_prediction': True,
                        })
                
                if self.pondering_enabled and self.opponent_turn and self.ponder_mode:
                    # === 多实例 Ponder 搜索循环 ===
                    chunk = 100  # 每次搜索的模拟次数
                    
                    if self.multi_mcts and self.multi_mcts.is_supported():
                        if self.ponder_phase == 1:
                            # 阶段 1：轮转搜索 3 个第一子实例
                            inst = self.multi_mcts.get_instance(self.ponder_current_instance)
                            inst.run_simulations(chunk)
                            self.ponder_total_sims += chunk
                            
                            # 每 300 次切换到下一个实例（按概率分配）
                            if self.ponder_total_sims % 300 < chunk:
                                self.ponder_current_instance = (self.ponder_current_instance + 1) % 3
                        
                        elif self.ponder_phase == 2:
                            # 阶段 2：轮转搜索 3 个第二子实例
                            inst = self.multi_mcts.get_instance(self.ponder_current_instance)
                            inst.run_simulations(chunk)
                            self.ponder_total_sims += chunk
                            
                            # 每 300 次切换到下一个实例
                            if self.ponder_total_sims % 300 < chunk:
                                self.ponder_current_instance = (self.ponder_current_instance + 1) % 3
                    
                    self.ponder_total_sims = sum(p['sims'] for p in self.ponder_paths) if self.ponder_paths else 0
                    current_tree_sims = self.ponder_total_sims
                    
                    # Ponder 时不发送任何统计信息，减少 UI 更新
                
                time.sleep(0.001)
                continue

            cmd, args = cmd_data
            
            if cmd == 'STOP':
                break
                
            elif cmd == 'RESET':
                # 设定固定随机种子，确保每局游戏起始状态的一致性
                # 消除 Dirichlet Noise 带来的随机差异
                self.mcts.set_random_seed(12345)
                
                # Ponder 实例也需要重置种子（如果支持）
                if self.multi_mcts:
                    for i in range(self.multi_mcts.num_instances):
                        inst = self.multi_mcts.get_instance(i)
                        if inst:
                            try:
                                inst.set_random_seed(12345 + i)
                            except:
                                pass

                self.mcts.reset()
                self.mcts.run_simulations(10) # 少量预热（在 Reset 后预热新状态）
                
                # 再次重置随机种子！
                # 因为上面的 run_simulations(10) 是多线程的，消耗的 RNG 次数不确定
                # 必须在这里重置，确保正式游戏开始时的 RNG 状态是固定的
                self.mcts.set_random_seed(12345)
                
                current_tree_sims = 0
                # 重置 Ponder 状态
                self._reset_ponder_state()
                self.opponent_turn = False
                self.opponent_stones_in_turn = 0
                self.opponent_turn_total = 0
                self.ponder_actual_move1 = -1
                # 清空着法历史
                self.move_history = []
                # 清除必杀第二子
                self.pending_kill_move = None
                # 重置 Ponder 统计（每局重置）
                self.ponder_stats = {'hit': 0, 'miss': 0, 'partial': 0}
                
                # 直接设置 AI 颜色（避免时序问题）
                if args is not None:
                    self.ai_color = args
                    print(f"🎯 [RESET] ai_color set to {self.ai_color} ({'黑' if self.ai_color == 1 else '白'})")
                
                # Set fixed random seed (Deterministic)
                self.mcts.set_random_seed(12345)

            elif cmd == 'LOAD_GAME_STATE':
                # Load entire game history: includes Reset + Replay
                # This is thread-safe as all operations are serialized within the AI thread
                moves_list = args
                
                # 1. Reset
                self.mcts.set_random_seed(12345)
                if self.multi_mcts:
                    for i in range(self.multi_mcts.num_instances):
                        inst = self.multi_mcts.get_instance(i)
                        if inst:
                             try: inst.set_random_seed(12345 + i)
                             except: pass

                self.mcts.reset()
                self.mcts.run_simulations(10) # Warmup
                self.mcts.set_random_seed(12345)
                
                self._reset_ponder_state()
                self.opponent_turn = False
                self.opponent_stones_in_turn = 0
                self.opponent_turn_total = 0
                self.move_history = []
                
                # 2. Replay moves
                if moves_list:
                    print(f"🔄 [AI] Thread-safe loading {len(moves_list)} moves...")
                    self.mcts.sync_state_from_moves(moves_list)
                    self.move_history = list(moves_list) # Copy
                
                current_tree_sims = 0
            
            elif cmd == 'PARAM':
                batch, threads, sims, dynamic, temperature = args
                
                # Update class attributes directly (Critical for search loop)
                self.simulations = sims
                self.dynamic_thinking = dynamic
                
                if dynamic: self.params['dynamic_simulations'] = True
                else: 
                     self.params['simulations'] = sims
                     self.params['dynamic_simulations'] = False
                
                self.params['batch_size'] = batch
                self.params['num_threads'] = threads
                self.params['temperature'] = temperature
                
                self.mcts.set_params(batch, threads)
                
            elif cmd == 'MOVE':
                move_idx, is_same_turn_second = args
                move_str = format_move_coord(move_idx // 19, move_idx % 19)
                
                # === Ponder Hit 检测 ===
                ponder_hit = False
                ponder_hit_rank = -1
                
                if self.ponder_mode and self.ponder_phase == 1:
                    # === 阶段 1：对手下的第一子（或黑棋唯一一子）===
                    actual_str = format_move_coord(move_idx // 19, move_idx % 19)
                    
                    # 使用多实例查找命中
                    hit_idx = -1
                    hit_sims = 0
                    if self.multi_mcts and self.multi_mcts.is_supported():
                        hit_idx, hit_inst = self.multi_mcts.find_matching_instance(move_idx)
                        if hit_idx >= 0:
                            hit_sims = hit_inst.get_visit_count()
                    else:
                        # 回退到旧逻辑
                        for i, p in enumerate(self.ponder_paths):
                            if move_idx == p['move1']:
                                hit_idx = i
                                hit_sims = p['sims']
                                break
                    
                    self.move_history.append(move_idx)
                    
                    # 判断是单子回合还是两子回合
                    if self.opponent_turn_total == 1:
                        # 黑棋第一手（单子）→ 直接复用第一子搜索树
                        if hit_idx >= 0 and hit_inst and hit_inst.copy_to_default():
                            # 同步 py_board 状态
                            self.mcts.py_board.reset()
                            for move in self.move_history:
                                self.mcts.py_board.make_move(move)
                            
                            self.ponder_stats['hit'] += 1
                            print(f"✅ [MultiPonder] 命中实例 {hit_idx}: {actual_str} +{hit_sims} sims (真实复用)")
                            current_tree_sims = hit_sims
                            ponder_hit = True
                        else:
                            # 未命中或复制失败
                            self.mcts.sync_state_from_moves(self.move_history)
                            if hit_idx >= 0:
                                self.ponder_stats['hit'] += 1
                                print(f"✅ [MultiPonder] 命中实例 {hit_idx}: {actual_str} (需重建树)")
                            else:
                                self.ponder_stats['miss'] += 1
                                print(f"❌ [MultiPonder] 未命中: {actual_str}")
                            current_tree_sims = 0
                        
                        self.opponent_stones_in_turn = 0
                        self._reset_ponder_state()
                    else:
                        # 两子回合，设置第二子搜索
                        self.opponent_stones_in_turn = 1
                        self.ponder_actual_move1 = move_idx
                        
                        if hit_idx >= 0 and hit_inst:
                            # 第一子命中！克隆命中实例到所有实例
                            self.multi_mcts.clone_all_from(hit_idx)
                            
                            # 获取 Top-3 第二子预测
                            policy2 = hit_inst.get_policy()
                            top3_move2 = np.argsort(policy2)[::-1][:3]
                            self.ponder_moves2 = [(int(idx), float(policy2[idx])) for idx in top3_move2]
                            
                            # 更新 ponder_paths[0] 的第二子列表
                            self.ponder_paths[0]['moves2_list'] = self.ponder_moves2
                            self.ponder_paths[0]['move2'] = self.ponder_moves2[0][0]
                            
                            # 为 3 个实例设置不同的第二子
                            for i in range(3):
                                inst = self.multi_mcts.get_instance(i)
                                move2 = self.ponder_moves2[i][0]
                                inst.play_move(move2)  # 在克隆的树上执行第二子
                            
                            # 发送第二子预测热力图
                            pred_strs = [f"{format_move_coord(m // 19, m % 19)}({int(p*100)}%)" 
                                        for m, p in self.ponder_moves2]
                            print(f"🔮 [MultiPonder] 第二子 Top-3: {', '.join(pred_strs)}")
                            
                            # Convert to (r,c,prob) format
                            policy_data = [(m // 19, m % 19, p) for m, p in self.ponder_moves2]
                            self.update_stats.emit({
                                'sims': hit_sims,
                                'win_rate': self.mcts.get_win_rate(ai_color=self.ai_color),
                                'time': 0,
                                'policy': policy_data,
                                'is_ponder_move2_prediction': True,
                            })
                            
                            self.ponder_phase = 2  # 阶段 2：搜索第二子
                            self.ponder_current_instance = 0
                            self.ponder_total_sims = 0
                        else:
                            # 第一子未命中，用主引擎快速预测第二子
                            self.mcts.run_simulations(simulations=200)
                            policy2 = self.mcts.get_policy()
                            top3_move2 = np.argsort(policy2)[::-1][:3]
                            self.ponder_moves2 = [(int(idx), float(policy2[idx])) for idx in top3_move2]
                            
                            # 设置 3 个实例搜索第二子
                            for i in range(3):
                                inst = self.multi_mcts.get_instance(i)
                                move2 = self.ponder_moves2[i][0]
                                inst.sync_from_moves(self.move_history)  # move_history 已包含第一子
                                inst.play_move(move2)
                            
                            self.ponder_phase = 2
                            self.ponder_current_instance = 0
                            self.ponder_total_sims = 0
                        
                        print(f"⏳ [MultiPonder] 等待对手第二子...")
                    
                elif self.ponder_mode and self.ponder_phase == 2:
                    # === 阶段 2：对手下的第二子 ===
                    actual1 = self.ponder_actual_move1
                    actual2 = move_idx
                    actual1_str = format_move_coord(actual1 // 19, actual1 % 19)
                    actual2_str = format_move_coord(actual2 // 19, actual2 % 19)
                    
                    # 使用多实例查找命中
                    hit_idx = -1
                    hit_sims = 0
                    hit_inst = None
                    if self.multi_mcts and self.multi_mcts.is_supported():
                        hit_idx, hit_inst = self.multi_mcts.find_matching_instance(actual2)
                        if hit_idx >= 0:
                            hit_sims = hit_inst.get_visit_count()
                    
                    if hit_idx >= 0 and hit_inst:
                        # 命中！复制实例树到主引擎实现真正复用
                        if hit_inst.copy_to_default():
                            # 同步 py_board 状态
                            self.mcts.py_board.reset()
                            for move in self.move_history:
                                self.mcts.py_board.make_move(move)
                            self.mcts.py_board.make_move(actual2)
                            
                            self.ponder_stats['hit'] += 1
                            print(f"✅ [MultiPonder] 命中 {actual1_str}+{actual2_str} (实例{hit_idx}) +{hit_sims} sims (真实复用)")
                            current_tree_sims = hit_sims
                            ponder_hit = True
                        else:
                            # 回退
                            self.mcts.sync_state_from_moves(self.move_history + [actual2])
                            self.ponder_stats['hit'] += 1
                            print(f"✅ Hit {actual1_str}+{actual2_str} (需重建树)")
                            current_tree_sims = 0
                            ponder_hit = True
                    else:
                        # 未命中
                        self.mcts.sync_state_from_moves(self.move_history + [actual2])
                        self.ponder_stats['miss'] += 1
                        print(f"❌ Miss {actual1_str}+{actual2_str}")
                        current_tree_sims = 0
                    
                    # 对手回合结束
                    self.move_history.append(actual2)
                    self.opponent_stones_in_turn = 0
                    self.ponder_actual_move1 = -1
                    self._reset_ponder_state()
                        
                else:
                    # 没有在 Ponder，正常更新状态
                    self.mcts.update_state(move_idx)
                    current_tree_sims = 0
                    self.move_history.append(move_idx)
                    # 验证状态同步
                    if len(self.move_history) != len(self.mcts.py_board.board) - self.mcts.py_board.board.count(0):
                        print(f"⚠️ [State] 历史长度不匹配: history={len(self.move_history)}, board_stones={361 - self.mcts.py_board.board.count(0)}")
                
                # 如果是同回合的第二子，强制重置搜索状态（不复用 ponder 树）
                if is_same_turn_second:
                    current_tree_sims = 0  # 强制第二子重新搜索
                    try:
                        self.mcts.reexpand_root()
                    except Exception:
                        pass
                # Ponder 主预测命中时，也重新展开根节点（但保留搜索结果）
                elif ponder_hit and ponder_hit_rank == 0:
                    try:
                        self.mcts.reexpand_root()
                    except Exception:
                        pass
                
            elif cmd == 'PARAM':
                b, t, s, d, temp = args
                self.params['batch'] = b
                self.params['threads'] = t
                self.simulations = s
                self.dynamic_thinking = d
                self.params['temperature'] = temp
                self.mcts.set_params(b, t)
                # 剪枝 K 值在 THINK 循环里动态调整，这里不需要设置
                
            elif cmd == 'PONDER_TOGGLE':
                self.pondering_enabled = args
                # 如果关闭后台思考，重置 Ponder 状态
                if not args:
                    self._reset_ponder_state()

            elif cmd == 'SET_AI_COLOR':
                self.ai_color = args

            elif cmd == 'SET_OPPONENT_TURN':
                self.opponent_turn = args
                if not args:
                    # 轮到 AI 了，停止 Ponder 并重置状态
                    self._reset_ponder_state()

            elif cmd == 'SET_DEEP_THINKING':
                self.deep_thinking_enabled = args

            elif cmd == 'VERIFY_STATE':
                ui_board_flat, ui_current_player = args
                
                # 如果在阶段 3 Ponder（有模拟的第二子），先同步回正确状态
                if self.ponder_mode and self.ponder_phase == 3:
                    self.mcts.sync_state_from_moves(self.move_history)
                
                is_match, mismatch_info = self.mcts.verify_state(ui_board_flat, ui_current_player)
                
                if not is_match:
                    print(f"⚠️ [State] 状态不匹配!")
                    print(f"   AI Player: {mismatch_info['ai_player']}, UI Player: {mismatch_info['ui_player']}")
                    if mismatch_info['mismatch_positions']:
                        for r, c, ai_val, ui_val in mismatch_info['mismatch_positions']:
                            print(f"   位置 ({r},{c}): AI={ai_val}, UI={ui_val}")
                    
                    # 强制重新同步
                    print(f"🔄 [State] 强制从 move_history 重新同步...")
                    self.mcts.sync_state_from_moves(self.move_history)
                    print(f"✅ [State] 同步完成")
                
            elif cmd == 'THINK':
                # ═══════════════════════════════════════════════════════════════
                #                    三阶段思考系统
                # ═══════════════════════════════════════════════════════════════
                
                # === 必杀第二子（两子必杀的后续） ===
                if self.pending_kill_move is not None:
                    move = self.pending_kill_move
                    self.pending_kill_move = None  # 清除
                    coord = format_move_coord(move // 19, move % 19)
                    print(f"⚔️ [必杀] 第二子: {coord}")
                    r, c = move // 19, move % 19
                    self.decision_made.emit(r, c, args) # args is game_id
                    continue
                

                
                # === 状态校验 (优先于必杀检测) ===
                board_stones = 361 - self.mcts.py_board.board.count(0)
                if len(self.move_history) != board_stones:
                    print(f"⚠️ [State] 思考前状态不一致: history={len(self.move_history)}, board={board_stones}")
                    print(f"🔄 [State] 强制重新同步...")
                    self.mcts.sync_state_from_moves(self.move_history)
                    print(f"✅ [State] 同步完成")

                # === 必杀检测 (回合开始时) ===
                # 只在回合第一子之前检测，检测 5+1 和 4+2 两种模式
                is_first_stone_of_turn = self.mcts.py_board.stones_in_turn == 0
                
                if is_first_stone_of_turn:
                    winning_pairs = self.mcts.py_board.get_winning_pairs()
                    if winning_pairs:
                        move1, move2 = winning_pairs[0]
                        coord1 = format_move_coord(move1 // 19, move1 % 19)
                        
                        if move2 is None:
                            # 5+1 模式：单子必杀
                            print(f"⚔️ [必杀] 单子必杀: {coord1}")
                        else:
                            # 4+2 模式：两子必杀，存储第二子
                            coord2 = format_move_coord(move2 // 19, move2 % 19)
                            print(f"⚔️ [必杀] 两子必杀: {coord1} + {coord2}")
                            self.pending_kill_move = move2  # 存储第二子
                        
                        r, c = move1 // 19, move1 % 19
                        self.decision_made.emit(r, c, args) # args is game_id
                        continue

                # === 开局库查询 ===
                # 启用开局库时，前6手优先查询开局库（前3回合）
                if self.opening_book and self.opening_book_enabled and len(self.move_history) < 6:
                    book_result = self.opening_book.query_random(self.move_history)
                    if book_result:
                        book_move = book_result['move']
                        book_wr = book_result['win_rate']
                        book_coord = format_move_coord(book_move // 19, book_move % 19)
                        print(f"📖 [开局库] 命中: {book_coord} (胜率 {book_wr:.1%})")
                        
                        # 发送统计信息到 UI
                        self.update_stats.emit({
                            'sims': 0,
                            'win_rate': book_wr,
                            'time': 0,
                            'policy': [(book_move // 19, book_move % 19, 1.0)],
                            'pruning_k': 0,
                            'is_opening_book': True,
                        })
                        
                        # 注意：不需要在这里更新 MCTS 状态
                        # UI 会通过 notify_move 发送 MOVE 命令自动更新
                        
                        r, c = book_move // 19, book_move % 19
                        self.decision_made.emit(r, c, args)  # args is game_id
                        continue
                
                # === 初始化搜索配置 ===
                search_config = SearchConfig(
                    max_simulations=self.simulations,
                    enable_dynamic=self.dynamic_thinking,
                    enable_deep=self.deep_thinking_enabled
                )
                
                # 确定搜索模式
                mode = SearchMode.FIXED
                if self.dynamic_thinking:
                    mode = SearchMode.DYNAMIC
                
                config = MCTSConfig(mode, self.simulations)
                strategy = MCTSStrategy(config, self.mcts, search_config)
                
                target = self.simulations
                chunk = 100
                start_time = time.time()
                k = 0  # 剪枝 K 值
                
                # ═══════════════════════════════════════════════════════════════
                #              阶段 1: 后台思考结果处理 (Background)
                # ═══════════════════════════════════════════════════════════════
                
                ponder_hit = current_tree_sims > 0
                ponder_base_sims = current_tree_sims
                
                # 设置 Ponder 命中状态，影响动态思考参数
                strategy.set_ponder_hit(ponder_hit)
                
                # 后台思考参数 (百分比)
                ponder_min_ratio = search_config.ponder_min_ratio      # 60%
                ponder_extra_ratio = search_config.ponder_extra_ratio  # 25%
                
                min_ponder_sims = int(self.simulations * ponder_min_ratio)
                min_extra_sims_base = int(self.simulations * ponder_extra_ratio)
                
                if ponder_hit:
                    # === 命中: 复用搜索树 ===
                    extra_needed = max(min_extra_sims_base, min_ponder_sims - current_tree_sims)
                    # 确保目标至少达到 min_ponder_sims
                    if current_tree_sims + extra_needed > target:
                        target = current_tree_sims + extra_needed
                else:
                    # === 未命中: 重置搜索树 ===
                    extra_needed = 0
                
                # 重置 Dynamic 状态
                strategy.last_check_sims = 0
                strategy.last_dist = None
                strategy.top1_q_history = []
                dynamic_state_reset = False
                
                # 重置 Q 值监控状态
                strategy.reset_q_monitor()
                
                # ═══════════════════════════════════════════════════════════════
                #              阶段 2: 动态思考 (Dynamic Thinking)
                # ═══════════════════════════════════════════════════════════════
                
                # === 初始设置 K=30 (确保第一次展开就使用正确的候选数量) ===
                try:
                    self.mcts.set_pruning_k(30)
                except Exception:
                    pass
                
                dynamic_fused = False  # 是否触发熔断
                deep_triggered = False  # 是否触发深度思考
                
                # 统计信息发送：使用 500 为间隔，更频繁更新
                last_stats_milestone = (current_tree_sims // 500) * 500
                
                # Ponder 命中时，立即发送一次初始状态
                if ponder_hit and current_tree_sims > 0:
                    wr = self.mcts.get_win_rate(ai_color=self.ai_color)
                    full_policy = self.mcts.get_policy()
                    
                    # 应用 Top-K 过滤，避免显示过多节点
                    policy_items = []
                    for idx in range(361):
                        if full_policy[idx] > 1e-6:
                            r, c = divmod(idx, 19)
                            policy_items.append((r, c, float(full_policy[idx])))
                    
                    # 按概率排序，取前 K 个（此时 K 应该是 50）
                    policy_items.sort(key=lambda x: x[2], reverse=True)
                    if len(policy_items) > 50:  # 限制最多 50 个
                        policy_data = policy_items[:50]
                    else:
                        policy_data = policy_items
                    
                    self.update_stats.emit({
                        'sims': current_tree_sims,
                        'win_rate': wr,
                        'time': 0,
                        'policy': policy_data,
                        'pruning_k': 30,  # 固定 K=30
                    })
                
                # === 胜率监控初始化 ===
                last_wr = self.mcts.get_win_rate(ai_color=self.ai_color) if current_tree_sims > 0 else None
                wr_drop_warned = False  # 防止重复警告
                
                while current_tree_sims < target:
                    # 检查中断
                    if not self.queue.empty():
                        next_cmd = self.queue.queue[0][0]
                        if next_cmd in ['RESET', 'STOP', 'MOVE']:
                            break
                        if next_cmd == 'FINISH_THINKING':
                            self.queue.get()
                            break
                        if next_cmd == 'PARAM':
                            _, p_args = self.queue.get()
                            b, t, s, d, temp = p_args
                            self.params['batch'] = b
                            self.params['threads'] = t
                            self.simulations = s
                            self.dynamic_thinking = d
                            self.params['temperature'] = temp
                            self.mcts.set_params(b, t)
                            
                            new_mode = SearchMode.FIXED
                            if d:
                                new_mode = SearchMode.DYNAMIC
                            strategy.config.mode = new_mode
                            strategy.update_budget(s)
                            target = s

                    # 执行搜索
                    self.mcts.get_mcts_move(simulations=chunk, temperature=0.0)
                    current_tree_sims += chunk

                    # === 固定剪枝 K=30 (标准视野) ===
                    k = 30
                    try:
                        self.mcts.set_pruning_k(k)
                    except Exception:
                        pass
                    
                    # === Ponder 复用时的额外搜索处理 ===
                    extra_done = current_tree_sims - ponder_base_sims
                    ponder_extra_complete = (not ponder_hit) or (extra_done >= extra_needed)
                    
                    # 额外搜索完成时，重置 Dynamic 状态
                    if ponder_hit and ponder_extra_complete and not dynamic_state_reset:
                        strategy.last_check_sims = current_tree_sims
                        strategy.last_dist = None
                        strategy.top1_q_history = []
                        dynamic_state_reset = True
                    
                    # === 动态思考检测 ===
                    if strategy.config.mode == SearchMode.DYNAMIC and ponder_extra_complete:
                        # 记录 Q 值历史
                        best_val = self.mcts.get_root_value()
                        strategy.top1_q_history.append(best_val)
                        
                        # 检查熔断条件
                        if strategy._check_dynamic_termination(current_tree_sims):
                            dynamic_fused = True
                            break
                    
                    # === Q 值监控触发额外搜索（不中断主循环）===
                    if strategy.check_q_change_trigger(current_tree_sims):
                        q_extra = int(self.simulations * search_config.q_extra_ratio)
                        print(f"⚡ [Q-Monitor] Q 值剧变 @ {current_tree_sims}, 额外搜索 {q_extra}")
                        
                        # 额外搜索 5%，不中断主循环
                        q_sims_done = 0
                        while q_sims_done < q_extra:
                            self.mcts.get_mcts_move(simulations=chunk, temperature=0.0)
                            q_sims_done += chunk
                            current_tree_sims += chunk
                        
                        # 扩展目标（如果需要）
                        if current_tree_sims > target:
                            target = current_tree_sims + int(self.simulations * 0.10)
                    
                    # === 检查深度思考触发条件 ===
                    if strategy.check_deep_trigger(current_tree_sims) and not deep_triggered:
                        deep_triggered = True
                        break
                         
                     # 发送统计信息 (每 500 次更新，让搜索有时间集中)
                    current_milestone = (current_tree_sims // 500) * 500
                    if current_milestone > last_stats_milestone:
                        last_stats_milestone = current_milestone
                        
                        wr = self.mcts.get_win_rate(ai_color=self.ai_color)
                        elapsed = time.time() - start_time
                        full_policy = self.mcts.get_policy()
                        
                        # 只显示当前剪枝范围内的 Top K 节点
                        # 这样用户可以看到剪枝后保留的节点（50→30→20→10）
                        policy_items = []
                        for idx in range(361):
                            if full_policy[idx] > 1e-9:  # 过滤掉完全没访问的节点
                                r, c = divmod(idx, 19)
                                policy_items.append((r, c, float(full_policy[idx])))
                        
                        # 按访问概率排序，取前 K 个
                        policy_items.sort(key=lambda x: x[2], reverse=True)
                        if k > 0 and len(policy_items) > k:
                            policy_data = policy_items[:k]  # 只显示 Top K
                        else:
                            policy_data = policy_items  # 如果不足 K 个，全部显示
                        
                        self.update_stats.emit({
                            'sims': current_tree_sims,
                            'win_rate': wr,
                            'time': elapsed,
                            'policy': policy_data,
                            'pruning_k': k,
                        })
                        
                        # === 胜率突降监控 ===
                        if last_wr is not None and not wr_drop_warned:
                            wr_drop = last_wr - wr
                            if wr_drop >= 0.20:  # 下降超过20%
                                print(f"⚠️ [胜率] 胜率下降 {wr_drop*100:.0f}% ({last_wr*100:.0f}% -> {wr*100:.0f}%) @ {current_tree_sims} sims")
                                wr_drop_warned = True
                            last_wr = wr  # 更新基准值

                # ═══════════════════════════════════════════════════════════════
                #              阶段 3: 深度思考 (Deep Thinking) - 循环验证
                # ═══════════════════════════════════════════════════════════════
                
                # 检查是否需要深度思考（熔断或90%触发）
                should_deep_think = (dynamic_fused or deep_triggered) and search_config.enable_deep
                
                if should_deep_think:
                    # 检查是否有中断
                    has_interrupt = False
                    if not self.queue.empty():
                        next_cmd = self.queue.queue[0][0]
                        if next_cmd in ['RESET', 'STOP', 'MOVE']:
                            has_interrupt = True
                    
                    if not has_interrupt:
                        # 循环验证：不稳定就继续搜索 10%，再验证，最多到 150%
                        max_sims = int(self.simulations * search_config.deep_max_ratio)
                        step_sims = int(self.simulations * search_config.deep_loop_step_ratio)
                        deep_chunk = 100
                        loop_count = 0
                        max_loops = 3  # 减少循环次数
                        
                        while current_tree_sims < max_sims and loop_count < max_loops:
                            loop_count += 1
                            
                            # 检查中断
                            if not self.queue.empty():
                                next_cmd = self.queue.queue[0][0]
                                if next_cmd in ['RESET', 'STOP', 'MOVE', 'FINISH_THINKING']:
                                    break
                            
                            # 记录验证前的 Value
                            value_before = self.mcts.get_root_value()
                            
                            # 执行额外模拟 (10%)
                            sims_done = 0
                            while sims_done < step_sims:
                                if not self.queue.empty():
                                    next_cmd = self.queue.queue[0][0]
                                    if next_cmd in ['RESET', 'STOP', 'MOVE', 'FINISH_THINKING']:
                                        break
                                
                                self.mcts.get_mcts_move(simulations=deep_chunk, temperature=0.0)
                                sims_done += deep_chunk
                                current_tree_sims += deep_chunk
                                
                                # === 深度思考剪枝策略 ===
                                # Q 值震荡触发：保持 K=30（继续探索）
                                # 90% 阶段或熔断触发：收缩到 K=15（深度验证，原 K=10 太窄）
                                if deep_triggered or dynamic_fused:  # 最终验证阶段
                                    k = 15  # 聚焦 Top 15 (平衡点)
                                else:
                                    k = 30  # Q 值震荡，保持探索
                                try:
                                    self.mcts.set_pruning_k(k)
                                except Exception:
                                    pass
                                
                                # 更新 UI 统计信息 (每 500 次)
                                current_milestone = (current_tree_sims // 500) * 500
                                if current_milestone > last_stats_milestone:
                                    last_stats_milestone = current_milestone
                                    wr = self.mcts.get_win_rate(ai_color=self.ai_color)
                                    elapsed = time.time() - start_time
                                    full_policy = self.mcts.get_policy()
                                    
                                    # 应用 Top-K 过滤
                                    policy_items = []
                                    for idx in range(361):
                                        if full_policy[idx] > 0.001:
                                            r, c = divmod(idx, 19)
                                            policy_items.append((r, c, float(full_policy[idx])))
                                    
                                    policy_items.sort(key=lambda x: x[2], reverse=True)
                                    if k > 0 and len(policy_items) > k:
                                        policy_data = policy_items[:k]
                                    else:
                                        policy_data = policy_items
                                    
                                    self.update_stats.emit({
                                        'sims': current_tree_sims,
                                        'win_rate': wr,
                                        'time': elapsed,
                                        'policy': policy_data,
                                        'pruning_k': k,
                                    })
                                    
                                    # === 胜率突降监控 ===
                                    if last_wr is not None and not wr_drop_warned:
                                        wr_drop = last_wr - wr
                                        if wr_drop >= 0.20:  # 下降超过20%
                                            print(f"⚠️ [胜率] 胜率下降 {wr_drop*100:.0f}% ({last_wr*100:.0f}% -> {wr*100:.0f}%) @ {current_tree_sims} sims")
                                            wr_drop_warned = True
                                        last_wr = wr  # 更新基准值
                            
                            # 检查 Value 稳定性
                            value_after = self.mcts.get_root_value()
                            value_change = abs(value_after - value_before)
                            
                            is_stable = value_change < search_config.deep_stable_threshold
                            
                            if is_stable:
                                break  # 稳定了，退出循环

                # ═══════════════════════════════════════════════════════════════
                #                         落子决策
                # ═══════════════════════════════════════════════════════════════
                
                # 检查中断
                has_interrupt = False
                if not self.queue.empty():
                    next_cmd = self.queue.queue[0][0]
                    if next_cmd in ['RESET', 'STOP', 'MOVE']:
                        has_interrupt = True
                
                if not has_interrupt:
                    elapsed = time.time() - start_time
                    print(f"✅ Done {current_tree_sims} sims {elapsed:.1f}s")
                    
                    # Opening randomness adjustment (DISABLED):
                    # 原因：temperature=0.5会导致AI在开局时随机选择，可能选到边缘位置
                    # 解决方案：开局也使用temperature=0.0，确保选择最佳着法
                    # Black first 3 stones (idxs 0, 3, 4) & White first 2 stones (idxs 1, 2)
                    # This corresponds to the first 5 moves in history.
                    move_temp = self.params.get('temperature', 0.0)  # 统一使用0.0，禁用开局随机性
                    
                    best_move = self.mcts.get_mcts_move(simulations=0, temperature=move_temp)
                    r, c = best_move // 19, best_move % 19
                    self.decision_made.emit(r, c, args) # args is game_id
