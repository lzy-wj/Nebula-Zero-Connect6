import sys
import os
import argparse
import glob
import re
import random
import json
import time
import shutil
import numpy as np
import torch
from datetime import datetime

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../reinforcement_learning')))

from core.connect6_game import Connect6Game
from core.mcts import MCTSEngine
import config

# --- Configuration ---
FINAL_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(FINAL_DIR, "tournament_state.json")
LOGS_DIR = os.path.join(FINAL_DIR, "logs")
RESULTS_DIR = os.path.join(FINAL_DIR, "results")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Helpers ---
def run_command(cmd):
    print(f"Running: {cmd}")
    os.system(cmd)

class TournamentEngine:
    def __init__(self, use_gpu='0'):
        self.device = torch.device(f'cuda:{use_gpu}' if torch.cuda.is_available() else 'cpu')
        self.engines = {} # Cache loaded engines? No, too much memory. Load on demand.
        self.gpu_id = use_gpu

    def ensure_engine(self, pth_path):
        engine_path = pth_path.replace('.pth', '.engine')
        if not os.path.exists(engine_path):
            print(f"Building engine for {pth_path}...")
            # 1. Export ONNX
            onnx_path = engine_path.replace('.engine', '.onnx')
            script_export = os.path.join(config.BASE_DIR, 'pipeline/export_onnx.py')
            cmd_export = f"CUDA_VISIBLE_DEVICES={self.gpu_id} python3 {script_export} {pth_path} {onnx_path}"
            os.system(cmd_export)
            
            # 2. Build Engine
            script_build = os.path.join(config.BASE_DIR, 'pipeline/build_engine.py')
            cmd_build = f"CUDA_VISIBLE_DEVICES={self.gpu_id} python3 {script_build} {onnx_path} {engine_path}"
            os.system(cmd_build)
            
            # Cleanup
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        return engine_path

    def play_game(self, black_path, white_path, simulations, game_id, round_name):
        """
        Play a single game.
        Returns: (winner_color, moves_list)
        winner_color: 1 (Black), -1 (White), 2 (Draw)
        """
        black_engine_path = self.ensure_engine(black_path)
        white_engine_path = self.ensure_engine(white_path)
        
        import gc
        
        mcts_black = None
        mcts_white = None
        
        try:
            mcts_black = MCTSEngine(black_engine_path, device=self.device)
            mcts_white = MCTSEngine(white_engine_path, device=self.device)
            
            mcts_black.set_params(batch_size=32, num_threads=4)
            mcts_white.set_params(batch_size=32, num_threads=4)
        except Exception as e:
            print(f"初始化引擎错误: {e}")
            return 0, []

        game = Connect6Game()
        
        try:
            # Game Loop
            while game.winner == 0:
                if game.current_player == 1:
                    current_mcts = mcts_black
                else:
                    current_mcts = mcts_white
                    
                # Setup callback
                from core.mcts import mcts_lib
                mcts_lib.set_eval_callback(current_mcts.c_callback)
                
                # Replay history (Essential for correct state)
                mcts_lib.init_game()
                for m_str in game.moves:
                    r, c = game._parse_coord(m_str)
                    mcts_lib.play_move(r * 19 + c)
                
                # Think
                # Low temperature for tournament play
                temp = 0.5 if len(game.moves) < 6 else 0.0 
                move = current_mcts.get_mcts_move(simulations=simulations, temperature=temp)
                game.play(move)
                
                # Safety check for stuck games
                if len(game.moves) > 200:
                    print("对局过长，强制平局。")
                    return 2, game.moves

            result = game.winner
            
            # Save Log
            log_file = os.path.join(LOGS_DIR, f"{round_name}_{game_id}.csv")
            with open(log_file, 'w') as f:
                f.write(f"胜者:{result},黑方:{os.path.basename(black_path)},白方:{os.path.basename(white_path)}\n")
                f.write(",".join(game.moves))
                
            return result, game.moves
            
        finally:
            print(f"清理内存: {game_id}")
            if mcts_black: del mcts_black
            if mcts_white: del mcts_white
            gc.collect()
            torch.cuda.empty_cache()

class TournamentManager:
    def __init__(self):
        self.state = self.load_state()
        self.engine = TournamentEngine()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "round": 0, # 0:Init, 1:Qualifiers, 2:Semis, 3:Elite8, 4:Finals
            "models": [], # All models
            "groups": {}, # Current grouping
            "scores": {}, # Scores for current round
            "qualified": [], # Models qualified for next round
            "games_played": [] # History of game IDs to avoid replay
        }

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_models(self):
        # Scan for all available generations
        files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "model_gen_*.pth"))
        models = []
        for f in files:
            match = re.search(r"model_gen_(\d+).pth", f)
            if match:
                models.append(f)
        
        # Sort by gen
        models.sort(key=lambda x: int(re.search(r"model_gen_(\d+).pth", x).group(1)))
        
        # Limit to 100 if more
        return models[-100:]

    def run_round_robin(self, group_name, model_paths, games_per_pair, simulations):
        print(f"\n>>> 正在进行循环赛: 组别 {group_name} ({len(model_paths)} 个模型)")
        
        # Initialize round_scores structure strictly before use
        if "round_scores" not in self.state: self.state["round_scores"] = {}
        if group_name not in self.state["round_scores"]: self.state["round_scores"][group_name] = {}
        
        if not model_paths:
            print("没有模型，跳过此组。")
            return []
            
        # Generate Schedule
        schedule = []
        for i in range(len(model_paths)):
            for j in range(len(model_paths)):
                if i != j:
                    num_games = games_per_pair // 2
                    for g in range(num_games):
                        schedule.append((model_paths[i], model_paths[j]))

        print(f"计划总对局数: {len(schedule)}")
        
        # Play
        for idx, (black, white) in enumerate(schedule):
            game_id = f"{group_name}_{idx}"
            
            # Check if played
            if game_id in self.state["games_played"]:
                print(f"跳过 {game_id} (已完成)")
                continue

            b_name = os.path.basename(black)
            w_name = os.path.basename(white)
            print(f"[{idx+1}/{len(schedule)}] {b_name} (黑) vs {w_name} (白) ...", end="", flush=True)
            
            start_t = time.time()
            winner, _ = self.engine.play_game(black, white, simulations, game_id, group_name)
            dur = time.time() - start_t
            
            gs = self.state["round_scores"][group_name]
            if b_name not in gs: gs[b_name] = 0
            if w_name not in gs: gs[w_name] = 0
            
            res_str = ""
            if winner == 1: 
                gs[b_name] += 1
                gs[w_name] -= 1
                res_str = "黑胜"
            elif winner == -1: 
                gs[w_name] += 1
                gs[b_name] -= 1
                res_str = "白胜"
            else:
                res_str = "平局"
            
            print(f" {res_str} ({dur:.1f}s)")
            
            self.state["games_played"].append(game_id)
            self.save_state()

        # Generate Detailed Report
        summary_path = os.path.join(RESULTS_DIR, f"summary_{group_name}.txt")
        gs = self.state["round_scores"][group_name]
        sorted_scores = sorted(gs.items(), key=lambda x: x[1], reverse=True)
        
        with open(summary_path, 'w') as f:
            f.write(f"=== {group_name} 成绩单 ===\n\n")
            f.write("【排行榜】\n")
            f.write(f"{'模型':<20} {'积分':<10}\n")
            f.write("-" * 30 + "\n")
            for name, score in sorted_scores:
                f.write(f"{name:<20} {score:<10}\n")
            
            f.write("\n【详细战绩】\n")
            # We need to reconstruct match history from logs or state?
            # Simpler: just iterate logs for this group since we have game_ids
            for idx, (black, white) in enumerate(schedule):
                game_id = f"{group_name}_{idx}"
                log_file = os.path.join(LOGS_DIR, f"{group_name}_{game_id}.csv")
                if os.path.exists(log_file):
                    with open(log_file, 'r') as logf:
                        header = logf.readline().strip() 
                        # Header format: 胜者:1,黑方:...,白方:...
                        try:
                            parts = header.split(',')
                            winner_code = int(parts[0].split(':')[1])
                            b_name = os.path.basename(black)
                            w_name = os.path.basename(white)
                            
                            res_str = "平局"
                            if winner_code == 1: res_str = f"{b_name} 胜"
                            elif winner_code == -1: res_str = f"{w_name} 胜"
                            
                            f.write(f"对局 {idx+1}: {b_name} (黑) vs {w_name} (白) -> {res_str}\n")
                        except:
                            f.write(f"对局 {idx+1}: 记录解析失败\n")

        return sorted_scores

    def run(self):
        if self.state["round"] == 0:
            print("--- 初始化擂台赛 ---")
            models = self.get_models()
            random.shuffle(models)
            self.state["models"] = models
            self.state["round"] = 1
            self.state["groups"] = {}
            self.save_state()

        # --- Round 1: Qualifiers (100 -> 40) ---
        if self.state["round"] == 1:
            print("\n=== 第一轮: 海选小组赛 (100 模型) ===")
            
            if "Round1" not in self.state["groups"]:
                groups = np.array_split(self.state["models"], 10)
                self.state["groups"]["Round1"] = [list(g) for g in groups]
                self.save_state()
            
            qualifiers = []
            
            for i, group in enumerate(self.state["groups"]["Round1"]):
                group_name = f"R1_组{i+1}"
                print(f"\n处理小组 {group_name}")
                
                if "round_results" in self.state and group_name in self.state["round_results"]:
                    print(f"小组 {group_name} 已完成。")
                    pass
                else:
                    scores = self.run_round_robin(group_name, group, games_per_pair=2, simulations=1200)
                    
                    if "round_results" not in self.state: self.state["round_results"] = {}
                    self.state["round_results"][group_name] = scores
                    self.save_state()

                scores = self.state["round_results"][group_name]
                top4 = [x[0] for x in scores[:4]]
                
                for name in top4:
                    full_path = next((p for p in group if os.path.basename(p) == name), None)
                    if full_path: qualifiers.append(full_path)
            
            self.state["round2_qualifiers"] = qualifiers
            print(f"第一轮结束。 {len(qualifiers)} 个模型晋级。")
            
            with open(os.path.join(RESULTS_DIR, "round1_winners.txt"), "w") as f:
                f.write("\n".join(qualifiers))
                
            self.state["round"] = 2
            self.state["groups"] = {} 
            self.save_state()

        # --- Round 2: Semi-Finals (40 -> 8) ---
        if self.state["round"] == 2:
            print("\n=== 第二轮: 半决赛小组赛 (40 模型) ===")
            models = self.state["round2_qualifiers"]
            random.shuffle(models)
            
            if "Round2" not in self.state["groups"]:
                groups = np.array_split(models, 2) 
            # Note: The logic below continues as in existing file, but since I am writing the WHOLE file, I need the rest.
            # I will assume the previous 'view_file' output is robust and I'll include the rest of the logic.
                self.state["groups"]["Round2"] = [list(g) for g in groups]
                self.save_state()
            
            qualifiers = []
            for i, group in enumerate(self.state["groups"]["Round2"]):
                group_name = f"R2_组{i+1}"
                scores = self.run_round_robin(group_name, group, games_per_pair=6, simulations=1200)
                
                if "round_results" not in self.state: self.state["round_results"] = {}
                self.state["round_results"][group_name] = scores
                self.save_state()
                
                top4 = [x[0] for x in scores[:4]]
                for name in top4:
                    full_path = next((p for p in group if os.path.basename(p) == name), None)
                    if full_path: qualifiers.append(full_path)

            self.state["round3_qualifiers"] = qualifiers
            print(f"第二轮结束。 {len(qualifiers)} 个模型晋级。")
            with open(os.path.join(RESULTS_DIR, "round2_winners.txt"), "w") as f:
                f.write("\n".join(qualifiers))
                
            self.state["round"] = 3
            self.state["groups"] = {}
            self.save_state()

        # --- Round 3: Elite 8 (8 -> 3) ---
        if self.state["round"] == 3:
            print("\n=== 第三轮: 八强精英赛 (8 模型) ===")
            models = self.state["round3_qualifiers"]
            
            group_name = "R3_精英组"
            scores = self.run_round_robin(group_name, models, games_per_pair=6, simulations=1200)
            
            if "round_results" not in self.state: self.state["round_results"] = {}
            self.state["round_results"][group_name] = scores
            self.save_state()
            
            top3 = [x[0] for x in scores[:3]]
            qualifiers = []
            for name in top3:
                full_path = next((p for p in models if os.path.basename(p) == name), None)
                if full_path: qualifiers.append(full_path)

            self.state["round4_qualifiers"] = qualifiers
            print(f"第三轮结束。 {len(qualifiers)} 个模型晋级。")
            with open(os.path.join(RESULTS_DIR, "round3_winners.txt"), "w") as f:
                f.write("\n".join(qualifiers))

            self.state["round"] = 4
            self.save_state()

        # --- Round 4: Finals (3 -> 2) ---
        if self.state["round"] == 4:
            print("\n=== 第四轮: 最终决赛 (3 模型) ===")
            models = self.state["round4_qualifiers"]
            
            group_name = "R4_决赛组"
            scores = self.run_round_robin(group_name, models, games_per_pair=3, simulations=5000)
            
            if "round_results" not in self.state: self.state["round_results"] = {}
            self.state["round_results"][group_name] = scores
            self.save_state()
            
            print("\n🏆 擂台赛最终结果 🏆")
            for rank, (name, score) in enumerate(scores):
                print(f"第 {rank+1} 名: {name} (积分: {score})")
                
            with open(os.path.join(RESULTS_DIR, "final_results.txt"), "w") as f:
                for rank, (name, score) in enumerate(scores):
                    f.write(f"第 {rank+1} 名: {name} (积分: {score})\n")

            self.state["round"] = 5 
            self.save_state()
            
        print("\n擂台赛全部结束!")

if __name__ == "__main__":
    manager = TournamentManager()
    manager.run()
