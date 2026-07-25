"""Nebula Zero 六子棋 Web 对弈服务。"""

import glob
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template, request


# 兼容从 Competition/web 目录直接执行 python app.py 的启动方式。
COMPETITION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if COMPETITION_DIR not in sys.path:
    sys.path.insert(0, COMPETITION_DIR)

# C++ MCTS 动态库可能尚未编译。延迟报告错误，让页面和模型列表仍能正常访问。
MCTSEngine = None
ENGINE_IMPORT_ERROR = None
try:
    from core.mcts import MCTSEngine
except Exception as exc:  # pragma: no cover - 取决于本机 CUDA/动态库环境
    ENGINE_IMPORT_ERROR = str(exc)


app = Flask(__name__)

BOARD_SIZE = 19
MAX_SIMULATIONS = 100_000
MAX_BATCH_SIZE = 64
MAX_THREADS = 256
CHECKPOINT_DIR = os.path.abspath(
    os.environ.get(
        "NEBULA_WEB_CHECKPOINT_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "../../reinforcement_learning/checkpoints",
        ),
    )
)
DEFAULT_ENGINE = os.path.join(CHECKPOINT_DIR, "current_model.engine")
DEFAULT_ENGINE = os.path.abspath(
    os.environ.get("NEBULA_WEB_DEFAULT_ENGINE", DEFAULT_ENGINE)
)


def _clamp_number(value, default, minimum, maximum, cast):
    """解析并限制网页传入的数值，避免非法参数进入 C++ 层。"""
    try:
        parsed = cast(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _available_engines():
    """只暴露 checkpoints 目录中的 TensorRT 引擎。"""
    engines = glob.glob(os.path.join(CHECKPOINT_DIR, "*.engine"))
    return sorted(engines, key=os.path.getmtime, reverse=True)


def _resolve_engine_path(requested_path):
    """校验模型路径，防止客户端让服务读取任意文件。"""
    candidate = os.path.realpath(requested_path or DEFAULT_ENGINE)
    allowed = {os.path.realpath(path) for path in _available_engines()}
    return candidate if candidate in allowed else None


def _pair_heads_for_engine(engine_path):
    """按同名 sidecar 绑定棋对头，确保 Web 不会混用不同代的参数。"""

    stem, _ = os.path.splitext(engine_path)
    candidates = [f"{stem}.pair_heads.pt"]
    # 正式循环的原子晋升文件采用 current_pair.engine 与
    # current_pair_heads.pt 两个固定名字，Web 可直接跟随最新已晋升模型。
    if os.path.basename(engine_path) == "current_pair.engine":
        candidates.append(
            os.path.join(os.path.dirname(engine_path), "current_pair_heads.pt")
        )
    return next((path for path in candidates if os.path.isfile(path)), None)


class GameState:
    """单局对弈状态；所有棋盘与引擎操作都由同一把可重入锁保护。"""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1  # 1：黑方，-1：白方
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.engine = None
        self.current_engine_path = None
        self.current_pair_heads_path = None
        self.lock = threading.RLock()
        # 搜索状态使用独立锁，MCTS 占用棋局锁时，网页仍然可以轮询真实进度。
        self.search_lock = threading.Lock()
        self.search_info = {
            "status": "idle",
            "completed": 0,
            "total": 0,
            "started_at": None,
            "finished_at": None,
            "params": {},
            "error": None,
        }

    def init_engine(self, engine_path):
        if MCTSEngine is None:
            detail = ENGINE_IMPORT_ERROR or "未知错误"
            raise RuntimeError(f"MCTS 引擎不可用：{detail}")

        with self.lock:
            if self.current_engine_path == engine_path and self.engine is not None:
                return

            print(f"Loading Engine: {engine_path}")
            pair_heads_path = _pair_heads_for_engine(engine_path)
            engine = MCTSEngine(
                engine_path,
                device="cuda:0",
                pair_heads_path=pair_heads_path,
            )
            engine.reset()
            for row, col in self.move_history:
                engine.update_state(row * BOARD_SIZE + col)

            # 新引擎完全就绪后再替换旧实例，加载失败不会破坏当前对局。
            self.engine = engine
            self.current_engine_path = engine_path
            self.current_pair_heads_path = pair_heads_path

    def reset(self):
        with self.lock:
            self.board.fill(0)
            self.current_player = 1
            self.move_history = []
            self.game_over = False
            self.winner = None
            if self.engine:
                self.engine.reset()
            self.clear_search()

    def clear_search(self):
        with self.search_lock:
            self.search_info = {
                "status": "idle",
                "completed": 0,
                "total": 0,
                "started_at": None,
                "finished_at": None,
                "params": {},
                "error": None,
            }

    def begin_search(self, total, params):
        with self.search_lock:
            self.search_info = {
                "status": "running",
                "completed": 0,
                "total": total,
                "started_at": time.time(),
                "finished_at": None,
                "params": dict(params),
                "error": None,
            }

    def update_search(self, completed):
        with self.search_lock:
            self.search_info["completed"] = min(
                int(completed), int(self.search_info["total"])
            )

    def finish_search(self, error=None):
        with self.search_lock:
            self.search_info["status"] = "error" if error else "done"
            if not error:
                self.search_info["completed"] = self.search_info["total"]
            self.search_info["finished_at"] = time.time()
            self.search_info["error"] = error

    def search_snapshot(self):
        with self.search_lock:
            info = dict(self.search_info)
            info["params"] = dict(self.search_info["params"])
        if info["started_at"] is None:
            info["elapsed"] = 0.0
        else:
            end_time = info["finished_at"] or time.time()
            info["elapsed"] = max(0.0, end_time - info["started_at"])
        info["progress"] = (
            info["completed"] / info["total"] if info["total"] > 0 else 0.0
        )
        return info

    def snapshot(self):
        with self.lock:
            return {
                "board": self.board.tolist(),
                "current_player": self.current_player,
                "history": [list(move) for move in self.move_history],
                "game_over": self.game_over,
                "winner": self.winner,
                "engine_name": (
                    os.path.basename(self.current_engine_path)
                    if self.current_engine_path
                    else None
                ),
                "engine_available": MCTSEngine is not None,
                "pair_enhanced": self.current_pair_heads_path is not None,
            }

    def check_win(self, row, col, color):
        """检查最后一手是否形成六连或以上。"""
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for delta_row, delta_col in directions:
            count = 1

            next_row, next_col = row + delta_row, col + delta_col
            while (
                0 <= next_row < BOARD_SIZE
                and 0 <= next_col < BOARD_SIZE
                and self.board[next_row, next_col] == color
            ):
                count += 1
                next_row += delta_row
                next_col += delta_col

            next_row, next_col = row - delta_row, col - delta_col
            while (
                0 <= next_row < BOARD_SIZE
                and 0 <= next_col < BOARD_SIZE
                and self.board[next_row, next_col] == color
            ):
                count += 1
                next_row -= delta_row
                next_col -= delta_col

            if count >= 6:
                return True
        return False

    def play(self, row, col):
        with self.lock:
            if self.game_over:
                return "game_over"
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                return "invalid"
            if self.board[row, col] != 0:
                return "occupied"

            color = self.current_player
            self.board[row, col] = color
            self.move_history.append((row, col))

            if self.engine:
                self.engine.update_state(row * BOARD_SIZE + col)

            if self.check_win(row, col, color):
                self.game_over = True
                self.winner = color
                return "win"

            if len(self.move_history) == BOARD_SIZE * BOARD_SIZE:
                self.game_over = True
                self.winner = 0
                return "draw"

            # 六子棋：黑方首手一子，之后双方每回合连续落两子。
            total_moves = len(self.move_history)
            if total_moves == 1:
                self.current_player = -1
            elif (total_moves - 1) % 2 == 0:
                self.current_player *= -1

            return "continue"

    def _rebuild(self, moves):
        """从着法序列重建棋盘和 MCTS 状态；调用方必须持有棋局锁。"""
        self.board.fill(0)
        self.current_player = 1
        self.move_history = []
        self.game_over = False
        self.winner = None
        if self.engine:
            self.engine.reset()

        for row, col in moves:
            result = self.play(row, col)
            if result not in {"continue", "win", "draw"}:
                raise ValueError(f"着法 {row},{col} 不合法：{result}")

    def load_moves(self, moves):
        """原子加载棋谱；新棋谱无效时恢复原有对局。"""
        with self.lock:
            previous_moves = list(self.move_history)
            try:
                self._rebuild(moves)
            except Exception:
                self._rebuild(previous_moves)
                raise
            self.clear_search()

    def undo(self, steps=1):
        with self.lock:
            if not self.move_history:
                return 0
            actual_steps = min(max(1, int(steps)), len(self.move_history))
            remaining_moves = self.move_history[:-actual_steps]
            self._rebuild(remaining_moves)
            self.clear_search()
            return actual_steps


game = GameState()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    game.reset()
    return jsonify({"status": "ok", **game.snapshot()})


@app.route("/state")
def get_state():
    return jsonify(game.snapshot())


@app.route("/search_status")
def get_search_status():
    return jsonify({"status": "ok", **game.search_snapshot()})


@app.route("/models")
def list_models():
    models = [
        {
            "name": os.path.basename(path),
            "path": path,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
            "modified_at": int(os.path.getmtime(path)),
            "pair_enhanced": _pair_heads_for_engine(path) is not None,
        }
        for path in _available_engines()
    ]
    return jsonify(
        {
            "models": models,
            "engine_available": MCTSEngine is not None,
            "engine_error": None if MCTSEngine is not None else ENGINE_IMPORT_ERROR,
            "default_path": (
                DEFAULT_ENGINE if os.path.realpath(DEFAULT_ENGINE) in {
                    os.path.realpath(path) for path in _available_engines()
                } else None
            ),
        }
    )


@app.route("/move", methods=["POST"])
def human_move():
    data = request.get_json(silent=True) or {}
    try:
        row, col = int(data["r"]), int(data["c"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "msg": "落子坐标格式不正确"}), 400

    with game.lock:
        result = game.play(row, col)
        snapshot = game.snapshot()
    if result in {"continue", "win", "draw"}:
        return jsonify(
            {
                "status": "ok",
                **snapshot,
                "game_over": snapshot["game_over"],
                "winner": snapshot["winner"],
            }
        )

    messages = {
        "occupied": "这个位置已经有棋子了",
        "game_over": "本局已经结束，请重新开局",
        "invalid": "落子位置超出棋盘范围",
    }
    return jsonify({"status": "error", "msg": messages.get(result, "无效落子")}), 400


@app.route("/undo", methods=["POST"])
def undo_move():
    data = request.get_json(silent=True) or {}
    steps = _clamp_number(data.get("steps"), 1, 1, 8, int)
    undone = game.undo(steps)
    if undone == 0:
        return jsonify({"status": "error", "msg": "当前没有可以撤销的着法"}), 409
    return jsonify({"status": "ok", "undone": undone, **game.snapshot()})


@app.route("/load", methods=["POST"])
def load_record():
    data = request.get_json(silent=True) or {}
    raw_moves = data.get("moves")
    if not isinstance(raw_moves, list) or len(raw_moves) > BOARD_SIZE * BOARD_SIZE:
        return jsonify({"status": "error", "msg": "棋谱 moves 必须是长度不超过 361 的数组"}), 400

    moves = []
    try:
        for item in raw_moves:
            if isinstance(item, dict):
                row = int(item.get("row", item.get("r")))
                col = int(item.get("col", item.get("c")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                row, col = int(item[0]), int(item[1])
            else:
                raise ValueError("不支持的着法格式")
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                raise ValueError("着法坐标超出棋盘范围")
            moves.append((row, col))
        game.load_moves(moves)
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "msg": f"棋谱加载失败：{exc}"}), 400

    return jsonify({"status": "ok", **game.snapshot()})


@app.route("/ai_move", methods=["POST"])
def ai_move():
    data = request.get_json(silent=True) or {}
    simulations = _clamp_number(
        data.get("sims"), 800, 1, MAX_SIMULATIONS, int
    )
    temperature = _clamp_number(data.get("temp"), 0.1, 0.0, 2.0, float)
    batch_size = _clamp_number(
        data.get("batch_size"), 32, 1, MAX_BATCH_SIZE, int
    )
    num_threads = _clamp_number(
        data.get("num_threads"), 4, 1, MAX_THREADS, int
    )

    engine_path = _resolve_engine_path(data.get("model_path"))
    if engine_path is None:
        return jsonify({"status": "error", "msg": "所选模型不存在或不在 checkpoints 目录中"}), 400

    try:
        game.init_engine(engine_path)
    except Exception as exc:  # TensorRT 错误需要明确反馈到页面
        app.logger.exception("加载 MCTS 引擎失败")
        return jsonify({"status": "error", "msg": f"模型加载失败：{exc}"}), 503

    # 主线 MCTS 支持尾批，严格执行用户填写的模拟次数，不再向下取整。
    effective_simulations = simulations
    search_params = {
        "simulations": effective_simulations,
        "requested_simulations": simulations,
        "batch_size": batch_size,
        "num_threads": num_threads,
        "temperature": temperature,
    }
    game.begin_search(effective_simulations, search_params)

    try:
        with game.lock:
            if game.game_over:
                game.finish_search("本局已经结束")
                return jsonify({"status": "error", "msg": "本局已经结束，请重新开局"}), 409

            game.engine.set_params(batch_size=batch_size, num_threads=num_threads)
            started_at = time.perf_counter()

            # 分块调用不改变搜索总量，使前端能看到真实完成进度。
            # 每块约 256 次模拟，兼顾进度刷新频率与 C++ 缓冲区分配开销。
            simulations_per_chunk = max(
                batch_size, (256 // batch_size) * batch_size
            )
            completed = 0
            while completed < effective_simulations:
                chunk = min(
                    simulations_per_chunk, effective_simulations - completed
                )
                game.engine.run_simulations(chunk)
                completed += chunk
                game.update_search(completed)

            best_move = int(
                game.engine.get_mcts_move(
                    simulations=0, temperature=temperature
                )
            )
            duration = time.perf_counter() - started_at
            win_rate = float(game.engine.get_win_rate())
            policy_array = np.asarray(game.engine.get_policy(), dtype=np.float32)
            legal_mask = game.board.reshape(-1) == 0

            row, col = divmod(best_move, BOARD_SIZE)
            if (
                not (0 <= best_move < BOARD_SIZE * BOARD_SIZE)
                or game.board[row, col] != 0
            ):
                raise RuntimeError("MCTS 没有返回有效落点")

            result = game.play(row, col)
            snapshot = game.snapshot()
        game.finish_search()
    except Exception as exc:
        game.finish_search(str(exc))
        app.logger.exception("MCTS 搜索失败")
        return jsonify({"status": "error", "msg": f"搜索失败：{exc}"}), 500

    # 搜索概率只用于展示，不参与落子；过滤掉已占据位置并取前五。
    ranked_indices = np.argsort(np.where(legal_mask, policy_array, -1.0))[::-1][:5]
    debug_moves = []
    for index in ranked_indices:
        probability = float(policy_array[index])
        if probability < 0.001:
            continue
        move_row, move_col = divmod(int(index), BOARD_SIZE)
        debug_moves.append(
            {
                "coord": f"{chr(ord('A') + move_col)}{BOARD_SIZE - move_row}",
                "prob": probability,
            }
        )

    return jsonify(
        {
            "status": "ok",
            **snapshot,
            "move": [row, col],
            "win_rate": win_rate,
            "duration": duration,
            "debug_moves": debug_moves,
            "game_over": snapshot["game_over"],
            "winner": snapshot["winner"],
            "search": {
                **search_params,
                "batch_size": batch_size,
                "num_threads": num_threads,
            },
        }
    )


if __name__ == "__main__":
    # 预加载失败不会阻止页面启动，用户点击 AI 落子时会看到具体原因。
    if os.path.exists(DEFAULT_ENGINE) and MCTSEngine is not None:
        try:
            game.init_engine(DEFAULT_ENGINE)
        except Exception:
            app.logger.exception("默认模型预加载失败")

    # 真实进度接口需要另一个请求线程在 MCTS 搜索期间提供状态。
    app.run(
        host=os.environ.get("NEBULA_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("NEBULA_WEB_PORT", "5000")),
        debug=False,
        threaded=True,
    )
