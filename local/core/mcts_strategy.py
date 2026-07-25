import time
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass


class SearchMode(Enum):
    FIXED = 1
    DYNAMIC = 2


@dataclass
class SearchConfig:
    """
    三阶段思考系统配置（所有阈值使用百分比，提高鲁棒性）
    
    阶段 1: 后台思考 (Background Pondering)
    阶段 2: 动态思考 (Dynamic Thinking) 
    阶段 3: 深度思考 (Deep Thinking)
    """
    # === 开关 ===
    enable_background: bool = True   # 后台思考
    enable_dynamic: bool = True      # 动态思考（熔断）
    enable_deep: bool = True         # 深度思考
    
    # === 基础参数 ===
    max_simulations: int = 12000     # 最大模拟次数
    
    # === 阶段 1: 后台思考参数 ===
    ponder_min_ratio: float = 0.50       # 复用最低阈值 50%（降低）
    ponder_extra_ratio: float = 0.15     # 额外搜索 15%（降低，命中说明预测准）
    ponder_batch_sims: int = 500         # Ponder 批量搜索次数（减少切换开销）
    
    # === 阶段 2: 动态思考参数 (命中) ===
    dynamic_min_ratio_hit: float = 0.20      # 命中: 20% 才开始检查熔断（提高）
    dynamic_fuse_ratio_hit: float = 4.0      # 命中: 熔断阈值 (Top1/Top2)（提高，更难熔断）
    dynamic_stable_start_hit: float = 0.50   # 命中: 稳定性检查起点 50%
    
    # === 阶段 2: 动态思考参数 (未命中) ===
    dynamic_min_ratio_miss: float = 0.30     # 未命中: 30% 就开始检查
    dynamic_fuse_ratio_miss: float = 5.0     # 未命中: 熔断阈值 (Top1/Top2)（提高，更难熔断）
    dynamic_stable_start_miss: float = 0.50  # 未命中: 稳定性检查起点 50%
    
    # === 阶段 3: 深度思考参数 ===
    deep_trigger_ratio: float = 0.90         # 90% 时触发深度思考
    deep_extra_ratio: float = 0.10           # 额外 10% 模拟
    deep_stable_window: float = 0.05         # 5% 窗口内检查稳定性
    deep_stable_threshold: float = 0.04      # Value 变化 < 0.04 视为稳定 (放宽到 4%)
    
    # === Q 值监控触发额外搜索 ===
    q_change_threshold: float = 0.15         # Q 值变化 > 0.15 触发额外搜索 (原 0.12)
    q_check_interval: int = 500              # Q 值检查间隔（模拟次数）
    q_monitor_start_ratio: float = 0.70      # 70% 后才开始 Q 值监控（原 0.60）
    q_extra_ratio: float = 0.03              # Q 值触发后额外搜索 3%（减少）
    
    # === 深度思考循环验证 ===
    deep_max_ratio: float = 1.30             # 最多搜索到 130%
    deep_loop_step_ratio: float = 0.15       # 每次循环额外搜索 15% (加快验证)


class MCTSConfig:
    def __init__(self, mode: SearchMode, max_budget: int = 2000):
        self.mode = mode
        self.max_budget = max_budget
        self.original_budget = max_budget

    def update_budget(self, new_budget):
        self.max_budget = new_budget
        self.original_budget = new_budget


class MCTSStrategy:
    """
    MCTS 搜索策略管理器
    
    支持三阶段思考系统:
    1. 后台思考 (Background): 对手回合预测搜索
    2. 动态思考 (Dynamic): 基于概率比的熔断机制
    3. 深度思考 (Deep): Top1 路径的深度验证
    """
    
    def __init__(self, config: MCTSConfig, mcts_engine, search_config: SearchConfig = None, callbacks=None):
        """
        :param config: MCTSConfig object (兼容旧代码)
        :param mcts_engine: MCTS 引擎
        :param search_config: SearchConfig 三阶段配置
        :param callbacks: 回调函数
        """
        self.config = config
        self.mcts = mcts_engine
        self.search_config = search_config or SearchConfig()
        self.callbacks = callbacks or {}
        self.stop_flag = False
        
        # State for Dynamic Logic
        self.last_dist = None
        self.last_check_sims = 0
        self.top1_q_history = []
        
        # 命中状态（由外部设置）
        self.ponder_hit = False
        
        # Q 值监控状态
        self.last_q_value = None
        self.last_q_check_sims = 0
        self.q_triggered_deep = False  # Q 值变化是否触发了深度思考

    def set_ponder_hit(self, hit: bool):
        """设置 Ponder 命中状态，影响动态思考参数"""
        self.ponder_hit = hit

    def get_dynamic_params(self):
        """
        根据 Ponder 命中状态返回动态思考参数
        
        Returns:
            (min_ratio, fuse_ratio, stable_start)
        """
        if self.ponder_hit:
            return (
                self.search_config.dynamic_min_ratio_hit,
                self.search_config.dynamic_fuse_ratio_hit,
                self.search_config.dynamic_stable_start_hit
            )
        else:
            return (
                self.search_config.dynamic_min_ratio_miss,
                self.search_config.dynamic_fuse_ratio_miss,
                self.search_config.dynamic_stable_start_miss
            )

    def update_budget(self, new_budget):
        if self.config.mode in [SearchMode.FIXED, SearchMode.DYNAMIC]:
            self.config.update_budget(new_budget)

    def stop(self):
        self.stop_flag = True

    def _get_top_moves_stats(self, current_sims):
        """
        获取 Top1 和 Top2 的访问次数和概率
        """
        policy = self.mcts.get_policy()
        top_indices = np.argsort(policy)[-2:][::-1]
        
        v1_idx = top_indices[0]
        p1 = float(policy[v1_idx])
        n1 = p1 * current_sims
        
        if len(top_indices) > 1:
            v2_idx = top_indices[1]
            p2 = float(policy[v2_idx])
            n2 = p2 * current_sims
        else:
            n2 = 0
            p2 = 0.0
            
        return n1, n2, p1, p2

    def _check_dynamic_termination(self, current_sims):
        """
        阶段 2: 动态思考 - 检查是否触发熔断
        
        根据 Ponder 命中状态使用不同参数:
        - 命中: 激进参数 (min=10%, fuse=3.0)
        - 未命中: 保守参数 (min=35%, fuse=5.0)
        """
        min_ratio, fuse_ratio, stable_start = self.get_dynamic_params()
        
        # 最低模拟次数检查
        if current_sims < self.config.max_budget * min_ratio:
            return False

        n1, n2, p1, p2 = self._get_top_moves_stats(current_sims)
        
        # Phase 1: Flash 熔断 (Top1 > 15× Top2)
        if n2 > 0 and n1 > 15 * n2:
            return True
        
        # Phase 2: Confident 熔断 (Top1 > fuse_ratio × Top2)
        if n2 > 0 and n1 > fuse_ratio * n2:
            return True
        
        # p1 > 70%
        if p1 > 0.70 and (n2 == 0 or n1 > fuse_ratio * n2):
            return True
                
        # Phase 3: Stability 稳定性检查
        if current_sims >= self.config.max_budget * stable_start:
            if current_sims - self.last_check_sims >= 1000:
                if self.last_dist is not None:
                    change = abs(p1 - self.last_dist)
                    if change < 0.02 and (n2 == 0 or n1 > 1.3 * n2):
                        return True
                
                self.last_dist = p1
                self.last_check_sims = current_sims
        
        # Phase 4: Panic 检查 (100% 预算时)
        if current_sims >= self.config.max_budget:
            if len(self.top1_q_history) >= 5:
                old_q = self.top1_q_history[-5]
                curr_q = self.mcts.get_root_value()
                
                if old_q - curr_q > 0.05:
                    ext = int(self.config.original_budget * 0.3)
                    self.config.update_budget(self.config.max_budget + ext)
                    return False
            
            return True
            
        return False

    def check_deep_trigger(self, current_sims):
        """
        阶段 3: 深度思考 - 检查是否触发深度验证
        
        触发条件:
        1. 动态熔断触发后
        2. 搜索达到 90% 时
        """
        if not self.search_config.enable_deep:
            return False
        
        trigger_sims = self.config.max_budget * self.search_config.deep_trigger_ratio
        return current_sims >= trigger_sims

    def check_q_change_trigger(self, current_sims):
        """
        Q 值监控触发深度思考
        
        当 Q 值在短时间内剧烈变化时，触发深度验证
        这可以捕捉到搜索过程中发现的重要变化
        
        注意：只在搜索中后期（60%后）才开始监控，前期 Q 值本来就不稳定
        可多次触发，每次触发后重置状态
        
        Returns:
            bool: 是否触发深度思考
        """
        if not self.search_config.enable_deep:
            return False
        
        # 前期不监控（Q 值本来就不稳定）
        if current_sims < self.config.max_budget * self.search_config.q_monitor_start_ratio:
            return False
        
        # 检查间隔
        if current_sims - self.last_q_check_sims < self.search_config.q_check_interval:
            return False
        
        current_q = self.mcts.get_root_value()
        self.last_q_check_sims = current_sims
        
        if self.last_q_value is not None:
            q_change = abs(current_q - self.last_q_value)
            if q_change > self.search_config.q_change_threshold:
                # 可多次触发，不设置 q_triggered_deep = True
                self.last_q_value = current_q  # 更新基准值
                return True
        
        self.last_q_value = current_q
        return False
    
    def reset_q_monitor(self):
        """重置 Q 值监控状态"""
        self.last_q_value = None
        self.last_q_check_sims = 0
        self.q_triggered_deep = False

    def run_deep_thinking(self, current_sims):
        """
        阶段 3: 深度思考 - 对 Top1 进行深度验证（单次）
        
        流程:
        1. 记录验证前的 Value
        2. 执行额外模拟 (10%)
        3. 检查 Value 稳定性
        4. 返回是否稳定
        """
        if not self.search_config.enable_deep:
            return True, 0  # 不启用深度思考，直接返回稳定
        
        extra_sims = int(self.config.max_budget * self.search_config.deep_extra_ratio)
        
        # 记录验证前的 Value
        value_before = self.mcts.get_root_value()
        
        # 执行额外模拟
        chunk = 100
        sims_done = 0
        
        while sims_done < extra_sims:
            self.mcts.get_mcts_move(simulations=chunk, temperature=0.0)
            sims_done += chunk
        
        # 检查 Value 稳定性
        value_after = self.mcts.get_root_value()
        value_change = abs(value_after - value_before)
        
        is_stable = value_change < self.search_config.deep_stable_threshold
        return is_stable, sims_done
    
    def run_deep_thinking_loop(self, current_sims, start_time, ai_color, update_callback=None):
        """
        阶段 3: 深度思考循环验证
        
        不稳定就继续搜索 10%，再验证，最多到 150%
        
        Args:
            current_sims: 当前模拟次数
            start_time: 搜索开始时间
            ai_color: AI 颜色
            update_callback: UI 更新回调函数
            
        Returns:
            (final_sims, is_stable): 最终模拟次数和是否稳定
        """
        if not self.search_config.enable_deep:
            return current_sims, True
        
        max_sims = int(self.config.max_budget * self.search_config.deep_max_ratio)
        step_sims = int(self.config.max_budget * self.search_config.deep_loop_step_ratio)
        chunk = 100
        
        total_sims = current_sims
        loop_count = 0
        max_loops = 5  # 最多循环 5 次，防止无限循环
        
        while total_sims < max_sims and loop_count < max_loops:
            loop_count += 1
            
            # 记录验证前的 Value
            value_before = self.mcts.get_root_value()
            
            # 执行额外模拟 (10%)
            sims_done = 0
            while sims_done < step_sims:
                self.mcts.get_mcts_move(simulations=chunk, temperature=0.0)
                sims_done += chunk
                total_sims += chunk
                
                # UI 更新回调
                if update_callback and total_sims % 1000 < chunk:
                    update_callback(total_sims)
            
            # 检查 Value 稳定性
            value_after = self.mcts.get_root_value()
            value_change = abs(value_after - value_before)
            
            is_stable = value_change < self.search_config.deep_stable_threshold
            
            if is_stable:
                return total_sims, True
        
        # 达到上限或循环次数用尽
        return total_sims, False

    def _report_progress(self, sims, final=False):
        if 'progress' in self.callbacks:
            wr = self.mcts.get_root_value()
            pol = self.mcts.get_policy()
            self.callbacks['progress'](sims, wr, pol, 0)
