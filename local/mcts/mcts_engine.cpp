#include "c6_logic.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <omp.h>
#include <random>
#include <set>
#include <vector>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {

// Constants for Scaled Atomic Float
const long long VALUE_SCALE = 1000000LL;

struct MCTSNode {
  int move;
  std::atomic<int> visit_count;
  std::atomic<int> virtual_loss;
  std::atomic<long long> value_sum_scaled;
  float prior_prob;
  int player;
  int next_player;

  MCTSNode *parent;
  std::vector<std::unique_ptr<MCTSNode>> children;

  MCTSNode(int m, float p, int ply, int next_ply, MCTSNode *par)
      : move(m), visit_count(0), virtual_loss(0), value_sum_scaled(0),
        prior_prob(p), player(ply), next_player(next_ply), parent(par) {}

  bool is_leaf() const { return children.empty(); }

  float get_value_sum() const {
    return (float)value_sum_scaled.load() / VALUE_SCALE;
  }

  void add_value(float v) {
    long long scaled = (long long)(v * VALUE_SCALE);
    value_sum_scaled.fetch_add(scaled);
  }

  float q_value() const {
    int vc = visit_count.load();
    int vl = virtual_loss.load();
    int total_visits = vc + vl;
    if (total_visits == 0)
      return 0.0f;

    float v_sum = get_value_sum();
    float adjusted_value = v_sum - (float)vl * 1.0f;
    return adjusted_value / total_visits;
  }
};

// ============================================================
// Multi-Instance MCTS: Instance Structure
// ============================================================

struct MCTSInstance {
  std::unique_ptr<MCTSNode> root;
  Connect6Board game_board;
  std::mt19937 rng;
  int batch_size;
  int num_threads;
  int pruning_k;

  // Persistent buffers
  std::vector<int> batch_boards;
  std::vector<float> batch_policies;
  std::vector<float> batch_values;
  std::vector<MCTSNode *> batch_nodes;
  std::vector<Connect6Board> batch_scratch_boards;
  std::vector<bool> batch_valid;

  MCTSInstance() : rng(42), batch_size(32), num_threads(4), pruning_k(0) {
    game_board.reset();
    root = std::make_unique<MCTSNode>(-1, 1.0f, 0, BLACK, nullptr);
    resize_buffers();
  }

  void resize_buffers() {
    int total_size = batch_size * 361;
    if (batch_boards.size() != total_size) {
      batch_boards.resize(total_size);
      batch_policies.resize(total_size);
      batch_values.resize(batch_size);
      batch_nodes.resize(batch_size);
      batch_scratch_boards.resize(batch_size);
      batch_valid.resize(batch_size);
    }
  }

  ~MCTSInstance() = default;
};

// ============================================================
// Global Callback (shared by all instances)
// ============================================================

typedef void (*EvalCallback)(int batch_size, const int *boards,
                             float *policy_output, float *value_output);
static EvalCallback py_eval_callback = nullptr;

// Default instance for backward compatibility
static MCTSInstance *g_default_instance = nullptr;

static MCTSInstance *get_default_instance() {
  if (!g_default_instance) {
    g_default_instance = new MCTSInstance();
  }
  return g_default_instance;
}

// ============================================================
// Multi-Instance API (new, with _ex suffix)
// ============================================================

DLL_EXPORT void *create_instance() { return new MCTSInstance(); }

DLL_EXPORT void destroy_instance(void *instance) {
  if (instance) {
    delete static_cast<MCTSInstance *>(instance);
  }
}

// Helper function: Deep copy a node and all its children
static std::unique_ptr<MCTSNode> clone_node_recursive(MCTSNode *src,
                                                      MCTSNode *new_parent) {
  if (!src)
    return nullptr;

  auto node = std::make_unique<MCTSNode>(
      src->move, src->prior_prob, src->player, src->next_player, new_parent);

  // Copy atomic values
  node->visit_count.store(src->visit_count.load());
  node->virtual_loss.store(src->virtual_loss.load());
  node->value_sum_scaled.store(src->value_sum_scaled.load());

  // Recursively clone children
  for (auto &child : src->children) {
    node->children.push_back(clone_node_recursive(child.get(), node.get()));
  }

  return node;
}

DLL_EXPORT void clone_instance(void *src_inst, void *dst_inst) {
  MCTSInstance *src = static_cast<MCTSInstance *>(src_inst);
  MCTSInstance *dst = static_cast<MCTSInstance *>(dst_inst);
  if (!src || !dst)
    return;

  // Copy game board state
  std::memcpy(dst->game_board.board, src->game_board.board,
              sizeof(src->game_board.board));
  dst->game_board.current_player = src->game_board.current_player;
  dst->game_board.stones_in_turn = src->game_board.stones_in_turn;
  dst->game_board.total_stones = src->game_board.total_stones;

  // Copy MCTS tree (deep copy)
  dst->root = clone_node_recursive(src->root.get(), nullptr);

  // Copy parameters
  dst->batch_size = src->batch_size;
  dst->num_threads = src->num_threads;
  dst->pruning_k = src->pruning_k;

  // RNG is not copied (each instance has independent randomness)
}

// Copy a multi-instance tree to the default (main) instance
DLL_EXPORT void copy_instance_to_default(void *src_inst) {
  MCTSInstance *src = static_cast<MCTSInstance *>(src_inst);
  MCTSInstance *dst = get_default_instance();
  if (!src || !dst)
    return;

  // Copy game board state
  std::memcpy(dst->game_board.board, src->game_board.board,
              sizeof(src->game_board.board));
  dst->game_board.current_player = src->game_board.current_player;
  dst->game_board.stones_in_turn = src->game_board.stones_in_turn;
  dst->game_board.total_stones = src->game_board.total_stones;

  // Copy MCTS tree (deep copy)
  dst->root = clone_node_recursive(src->root.get(), nullptr);
}

DLL_EXPORT void init_game_ex(void *instance) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst)
    return;

  inst->game_board.reset();
  inst->root = std::make_unique<MCTSNode>(-1, 1.0f, 0, BLACK, nullptr);
}

DLL_EXPORT void set_random_seed_ex(void *instance, int seed) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (inst)
    inst->rng.seed(seed);
}

DLL_EXPORT void set_mcts_params_ex(void *instance, int batch_size,
                                   int num_threads) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst)
    return;

  if (batch_size > 0)
    inst->batch_size = batch_size;
  if (num_threads > 0)
    inst->num_threads = num_threads;
  inst->resize_buffers();
  omp_set_num_threads(inst->num_threads);
}

DLL_EXPORT void set_pruning_k_ex(void *instance, int k) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (inst)
    inst->pruning_k = k;
}

DLL_EXPORT void play_move_ex(void *instance, int move) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst)
    return;

  inst->game_board.make_move(move);

  bool found = false;
  if (inst->root && !inst->root->children.empty()) {
    for (auto &child : inst->root->children) {
      if (child->move == move) {
        inst->root = std::move(child);
        inst->root->parent = nullptr;
        found = true;
        break;
      }
    }
  }

  if (!found) {
    inst->root = std::make_unique<MCTSNode>(
        move, 1.0f, 0, inst->game_board.current_player, nullptr);
  }
}

DLL_EXPORT void run_mcts_simulations_ex(void *instance, int num_simulations) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst || py_eval_callback == nullptr)
    return;

  int loops = num_simulations / inst->batch_size;
  if (loops == 0)
    loops = 1;

  // Using persistent buffers from inst
  std::vector<int> &batch_boards = inst->batch_boards;
  std::vector<float> &batch_policies = inst->batch_policies;
  std::vector<float> &batch_values = inst->batch_values;
  std::vector<MCTSNode *> &batch_nodes = inst->batch_nodes;
  std::vector<Connect6Board> &batch_scratch_boards = inst->batch_scratch_boards;
  std::vector<bool> &batch_valid = inst->batch_valid;

  for (int l = 0; l < loops; ++l) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < inst->batch_size; ++b) {
      Connect6Board scratch_board = inst->game_board;
      MCTSNode *node = inst->root.get();

      while (!node->is_leaf()) {
        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode *best_child = nullptr;

        for (auto &child : node->children) {
          float score = child->q_value();
          if (node->next_player == WHITE)
            score = -score;

          int p_visits = node->visit_count.load() + node->virtual_loss.load();
          float prior = child->prior_prob;
          int c_visits = child->visit_count.load() + child->virtual_loss.load();
          float u_val =
              1.5f * prior * std::sqrt((float)p_visits) / (1.0f + c_visits);
          float total_score = score + u_val;

          if (total_score > best_score) {
            best_score = total_score;
            best_child = child.get();
          }
        }

        if (best_child) {
          best_child->virtual_loss++;
          node = best_child;
          scratch_board.make_move(node->move);
        } else
          break;
      }

      batch_nodes[b] = node;
      batch_scratch_boards[b] = scratch_board;

      int win = scratch_board.check_win(node->move);
      if (win != 0) {
        float value = (win == BLACK) ? 1.0f : -1.0f;
        MCTSNode *curr = node;
        while (curr != nullptr) {
          if (curr != inst->root.get() && curr->parent) {
            curr->virtual_loss--;
            curr->visit_count++;
            curr->add_value(value);
          } else if (curr == inst->root.get()) {
            curr->visit_count++;
            curr->add_value(value);
          }
          curr = curr->parent;
        }
        batch_valid[b] = false;
      } else {
        int offset = b * 361;
        const int *b_ptr = scratch_board.board;
        for (int i = 0; i < 361; ++i)
          batch_boards[offset + i] = b_ptr[i];
        batch_valid[b] = true;
      }
    }

    int valid_count = 0;
    for (bool v : batch_valid)
      if (v)
        valid_count++;
    if (valid_count == 0)
      continue;

    if (py_eval_callback) {
      py_eval_callback(inst->batch_size, batch_boards.data(),
                       batch_policies.data(), batch_values.data());
    }

    for (int i = 0; i < inst->batch_size; ++i) {
      if (!batch_valid[i])
        continue;

      MCTSNode *node = batch_nodes[i];
      bool already_expanded = !node->is_leaf();

      float val_ret = batch_values[i];
      const Connect6Board &s_board = batch_scratch_boards[i];
      float value = (s_board.current_player == BLACK) ? val_ret : -val_ret;

      // 修复竞态条件：使用锁保护整个扩展过程
      // 防止多个线程同时检测到 is_leaf 并尝试写入 node->children
      static std::mutex expand_mutex;
      std::lock_guard<std::mutex> lock(expand_mutex);

      if (node->is_leaf()) {
        float *policy = &batch_policies[i * 361];
        std::vector<int> legal_moves = s_board.get_legal_moves();

        std::vector<int> winning_moves = s_board.get_winning_moves();
        std::vector<int> forced_moves = s_board.get_forced_moves();
        std::set<int> protected_moves;
        for (int m : winning_moves)
          protected_moves.insert(m);
        for (int m : forced_moves)
          protected_moves.insert(m);

        struct MoveProb {
          int move;
          float prob;
          bool is_protected;
        };
        std::vector<MoveProb> candidates;
        candidates.reserve(legal_moves.size());

        float policy_sum = 0.0f;
        for (int m : legal_moves) {
          policy_sum += policy[m];
          candidates.push_back({m, policy[m], protected_moves.count(m) > 0});
        }

        if (policy_sum > 1e-8) {
          for (auto &item : candidates)
            item.prob /= policy_sum;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const MoveProb &a, const MoveProb &b) {
                    return a.prob > b.prob;
                  });

        // 直接使用 pruning_k 控制候选数量（不使用累积概率）
        const int MAX_MOVES = (inst->pruning_k > 0) ? inst->pruning_k : 30;

        std::vector<MoveProb> filtered;
        filtered.reserve(candidates.size());

        // 先添加必须保护的着法
        for (const auto &item : candidates) {
          if (item.is_protected) {
            filtered.push_back(item);
          }
        }

        // 再添加其他高概率着法，直到达到 MAX_MOVES
        for (const auto &item : candidates) {
          if (item.is_protected)
            continue;
          if (filtered.size() >= (size_t)MAX_MOVES)
            break;
          filtered.push_back(item);
        }

        float subset_sum = 0.0f;
        for (const auto &item : filtered)
          subset_sum += item.prob;

        for (const auto &item : filtered) {
          float prior = item.prob / (subset_sum + 1e-8);
          if (item.is_protected && prior < 0.05f)
            prior = 0.05f; // 提高到 5%，确保防守点获得足够探索

          int next_ply = s_board.current_player;
          int stones = s_board.stones_in_turn + 1;
          int total = s_board.total_stones + 1;
          if (total == 1 || stones >= 2)
            next_ply = -next_ply;

          node->children.push_back(std::make_unique<MCTSNode>(
              item.move, prior, s_board.current_player, next_ply, node));
        }
      }

      MCTSNode *curr = node;
      while (curr != nullptr) {
        if (curr != inst->root.get() && curr->parent) {
          curr->virtual_loss--;
          curr->visit_count++;
          curr->add_value(value);
        } else if (curr == inst->root.get()) {
          curr->visit_count++;
          curr->add_value(value);
        }
        curr = curr->parent;
      }
    }
  }
}

DLL_EXPORT int get_best_move_ex(void *instance, float temperature) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst || !inst->root || inst->root->children.empty())
    return -1;

  if (temperature == 0.0f) {
    int best_move = -1;
    int max_visits = -1;
    for (auto &child : inst->root->children) {
      int vc = child->visit_count.load();
      if (vc > max_visits) {
        max_visits = vc;
        best_move = child->move;
      }
    }
    return best_move;
  } else {
    std::vector<float> probs;
    std::vector<int> moves;
    float sum = 0.0f;

    for (auto &child : inst->root->children) {
      int vc = child->visit_count.load();
      if (vc > 0) {
        float p = std::pow((float)vc, 1.0f / temperature);
        probs.push_back(p);
        moves.push_back(child->move);
        sum += p;
      }
    }

    if (sum == 0.0f)
      return -1;

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(inst->rng);
    float running_sum = 0.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
      running_sum += probs[i];
      if (r <= running_sum)
        return moves[i];
    }
    return moves.back();
  }
}

DLL_EXPORT float get_root_value_ex(void *instance) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst || !inst->root)
    return 0.0f;
  return inst->root->q_value();
}

DLL_EXPORT void get_policy_ex(void *instance, float *output) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  std::fill(output, output + 361, 0.0f);
  if (!inst || !inst->root || inst->root->children.empty())
    return;

  float sum_visits = 0.0f;
  for (auto &child : inst->root->children) {
    sum_visits += (float)child->visit_count.load();
  }

  for (auto &child : inst->root->children) {
    if (child->move >= 0 && child->move < 361) {
      float p = (sum_visits > 0.0f)
                    ? (float)child->visit_count.load() / sum_visits
                    : 0.0f;
      if (p == 0.0f)
        p = 1e-9f;
      output[child->move] = p;
    }
  }
}

DLL_EXPORT int get_visit_count_ex(void *instance) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst || !inst->root)
    return 0;
  return inst->root->visit_count.load();
}

DLL_EXPORT void reexpand_root_ex(void *instance, const float *new_policy) {
  MCTSInstance *inst = static_cast<MCTSInstance *>(instance);
  if (!inst || !inst->root || !new_policy)
    return;

  std::set<int> existing_moves;
  for (auto &child : inst->root->children)
    existing_moves.insert(child->move);

  std::vector<int> legal_moves = inst->game_board.get_legal_moves();
  std::vector<int> winning_moves = inst->game_board.get_winning_moves();
  std::vector<int> forced_moves = inst->game_board.get_forced_moves();

  std::set<int> protected_moves;
  for (int m : winning_moves)
    protected_moves.insert(m);
  for (int m : forced_moves)
    protected_moves.insert(m);

  struct MoveProb {
    int move;
    float prob;
    bool is_protected;
    bool exists;
  };
  std::vector<MoveProb> candidates;

  float policy_sum = 0.0f;
  for (int m : legal_moves) {
    policy_sum += new_policy[m];
    candidates.push_back({m, new_policy[m], protected_moves.count(m) > 0,
                          existing_moves.count(m) > 0});
  }

  if (policy_sum > 1e-8) {
    for (auto &item : candidates)
      item.prob /= policy_sum;
  }

  for (auto &child : inst->root->children) {
    for (const auto &item : candidates) {
      if (item.move == child->move) {
        child->prior_prob = item.prob;
        break;
      }
    }
  }

  std::sort(
      candidates.begin(), candidates.end(),
      [](const MoveProb &a, const MoveProb &b) { return a.prob > b.prob; });

  // 直接使用 pruning_k 控制候选数量（不使用累积概率）
  const int MAX_MOVES = (inst->pruning_k > 0) ? inst->pruning_k : 30;

  // 先添加必须保护的着法
  for (const auto &item : candidates) {
    if (!item.exists && item.is_protected) {
      int next_ply = inst->game_board.current_player;
      int stones = inst->game_board.stones_in_turn + 1;
      int total = inst->game_board.total_stones + 1;
      if (total == 1 || stones >= 2)
        next_ply = -next_ply;

      float prior = std::max(item.prob, 0.01f);
      inst->root->children.push_back(std::make_unique<MCTSNode>(
          item.move, prior, inst->game_board.current_player, next_ply,
          inst->root.get()));
    }
  }

  // 再添加其他高概率着法，直到达到 MAX_MOVES
  for (const auto &item : candidates) {
    if (item.exists || item.is_protected)
      continue;
    if (inst->root->children.size() >= (size_t)MAX_MOVES)
      break;

    int next_ply = inst->game_board.current_player;
    int stones = inst->game_board.stones_in_turn + 1;
    int total = inst->game_board.total_stones + 1;
    if (total == 1 || stones >= 2)
      next_ply = -next_ply;

    inst->root->children.push_back(std::make_unique<MCTSNode>(
        item.move, item.prob, inst->game_board.current_player, next_ply,
        inst->root.get()));
  }

  float total_prior = 0.0f;
  for (auto &child : inst->root->children)
    total_prior += child->prior_prob;
  if (total_prior > 1e-8) {
    for (auto &child : inst->root->children)
      child->prior_prob /= total_prior;
  }
}

// ============================================================
// Legacy API (backward compatibility - uses default instance)
// ============================================================

DLL_EXPORT void set_eval_callback(EvalCallback cb) { py_eval_callback = cb; }

DLL_EXPORT void set_random_seed(int seed) {
  set_random_seed_ex(get_default_instance(), seed);
}

DLL_EXPORT void set_mcts_params(int batch_size, int num_threads) {
  set_mcts_params_ex(get_default_instance(), batch_size, num_threads);
}

DLL_EXPORT void set_pruning_k(int k) {
  set_pruning_k_ex(get_default_instance(), k);
}

DLL_EXPORT void init_game() { init_game_ex(get_default_instance()); }

DLL_EXPORT void play_move(int move) {
  play_move_ex(get_default_instance(), move);
}

DLL_EXPORT void run_mcts_simulations(int num_simulations) {
  run_mcts_simulations_ex(get_default_instance(), num_simulations);
}

DLL_EXPORT int get_best_move(float temperature) {
  return get_best_move_ex(get_default_instance(), temperature);
}

DLL_EXPORT float get_root_value() {
  return get_root_value_ex(get_default_instance());
}

DLL_EXPORT void get_policy(float *output) {
  get_policy_ex(get_default_instance(), output);
}

DLL_EXPORT void reexpand_root(const float *new_policy) {
  reexpand_root_ex(get_default_instance(), new_policy);
}

DLL_EXPORT void print_top_moves() {
  // Silent
}
}
