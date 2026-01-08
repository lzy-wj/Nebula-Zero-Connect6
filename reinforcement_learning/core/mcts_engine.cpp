#include "c6_logic.h"
#include <cmath>
#include <memory>
#include <map>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <omp.h>

// Cross-platform DLL export macro
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// Export C Interface
extern "C" {

// Constants for Scaled Atomic Float
const long long VALUE_SCALE = 1000000LL;

struct MCTSNode {
    int move;
    std::atomic<int> visit_count;
    std::atomic<int> virtual_loss; // Virtual Loss to penalize nodes currently being explored
    std::atomic<long long> value_sum_scaled; // Scaled value sum for atomic updates
    float prior_prob;
    int player; // Player who made the move leading to this node
    int next_player; // Player to move next
    
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    
    MCTSNode(int m, float p, int ply, int next_ply, MCTSNode* par) 
        : move(m), visit_count(0), virtual_loss(0), value_sum_scaled(0), prior_prob(p), 
          player(ply), next_player(next_ply), parent(par) {}

    bool is_leaf() const {
        return children.empty();
    }
    
    float get_value_sum() const {
        return (float)value_sum_scaled.load() / VALUE_SCALE;
    }
    
    void add_value(float v) {
        long long scaled = (long long)(v * VALUE_SCALE);
        value_sum_scaled.fetch_add(scaled);
    }
    
    float q_value() const {
        // visit_count contains completed visits.
        // virtual_loss contains pending visits.
        
        int vc = visit_count.load();
        int vl = virtual_loss.load();
        int total_visits = vc + vl;
        if (total_visits == 0) return 0.0f;
        
        // AlphaZero Standard: Q = (W - VL) / (N + VL)
        float v_sum = get_value_sum();
        float adjusted_value = v_sum - (float)vl * 1.0f;
        return adjusted_value / total_visits;
    }
    
    // UCB
    float ucb_score(float cpuct) const {
        float q = q_value();
        
        if (parent == nullptr) return 0.0f;
        
        // Parent total visits (Real + Virtual)
        int p_vc = parent->visit_count.load();
        int p_vl = parent->virtual_loss.load();
        int parent_visits = p_vc + p_vl;
        
        int my_vc = visit_count.load();
        int my_vl = virtual_loss.load();
        
        if (parent_visits == 0) return 0.0f;
        
        float u = prior_prob * std::sqrt((float)parent_visits) / (1.0f + my_vc + my_vl);
        return q + cpuct * u;
    }
};

std::unique_ptr<MCTSNode> root;
Connect6Board game_board;
std::mt19937 rng(42); 

// Params
int G_BATCH_SIZE = 32;
int G_NUM_THREADS = 4;

// Callback
typedef void (*EvalCallback)(int batch_size, const int* boards, float* policy_output, float* value_output);
EvalCallback py_eval_callback = nullptr;

EXPORT void set_eval_callback(EvalCallback cb) {
    py_eval_callback = cb;
}

EXPORT void set_random_seed(int seed) {
    rng.seed(seed);
}

EXPORT void set_mcts_params(int batch_size, int num_threads) {
    if (batch_size > 0) G_BATCH_SIZE = batch_size;
    if (num_threads > 0) G_NUM_THREADS = num_threads;
    omp_set_num_threads(G_NUM_THREADS);
}

EXPORT void init_game() {
    game_board.reset();
    root = std::make_unique<MCTSNode>(-1, 1.0f, 0, BLACK, nullptr);
}

EXPORT void play_move(int move) {
    game_board.make_move(move);
    
    bool found = false;
    if (root && !root->children.empty()) {
        for (auto& child : root->children) {
            if (child->move == move) {
                root = std::move(child);
                root->parent = nullptr;
                found = true;
                break;
            }
        }
    }
    
    if (!found) {
        root = std::make_unique<MCTSNode>(move, 1.0f, 0, game_board.current_player, nullptr);
    }
}

EXPORT void run_mcts_simulations(int num_simulations) {
    if (py_eval_callback == nullptr) return;

    int loops = num_simulations / G_BATCH_SIZE;
    if (loops == 0) loops = 1; 
    
    // Buffers for batch evaluation
    // Note: With OpenMP, we need to be careful about pushing to vectors.
    // We will use pre-allocated arrays for the batch.
    
    std::vector<int> batch_boards(G_BATCH_SIZE * 361);
    std::vector<float> batch_policies(G_BATCH_SIZE * 361);
    std::vector<float> batch_values(G_BATCH_SIZE);
    
    // Pointers to nodes corresponding to batch
    std::vector<MCTSNode*> batch_nodes(G_BATCH_SIZE);
    std::vector<Connect6Board> batch_scratch_boards(G_BATCH_SIZE);
    std::vector<bool> batch_valid(G_BATCH_SIZE);

    for (int l = 0; l < loops; ++l) {
        // 1. Selection Phase (Parallel)
        
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < G_BATCH_SIZE; ++b) {
            Connect6Board scratch_board = game_board; 
            MCTSNode* node = root.get();
            
            // Selection
            while (!node->is_leaf()) {
                float best_score = -std::numeric_limits<float>::infinity();
                MCTSNode* best_child = nullptr;
                
                // Iterate children (Read-only structure, safe)
                for (auto& child : node->children) {
                    float score = child->q_value();
                    if (node->next_player == WHITE) {
                        score = -score; 
                    }
                    
                    float u = 1.5f * child->ucb_score(0.0f); // Simplified call, ucb_score computes full PUCT
                    
                    // Re-implement UCB logic to match original exactly
                    int p_visits = node->visit_count.load() + node->virtual_loss.load();
                    float prior = child->prior_prob;
                    int c_visits = child->visit_count.load() + child->virtual_loss.load();
                    
                    float u_val = 1.5f * prior * std::sqrt((float)p_visits) / (1.0f + c_visits);
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
                } else {
                    break;
                }
            }
            
            batch_nodes[b] = node;
            batch_scratch_boards[b] = scratch_board;
            
            // Check Terminal
            int win = scratch_board.check_win(node->move);
            if (win != 0) {
                // Terminal
                float value = (win == BLACK) ? 1.0f : -1.0f;
                
                // Backprop immediately (thread-safe due to atomics)
                MCTSNode* curr = node;
                while(curr != nullptr) {
                     if (curr != root.get() && curr->parent) {
                        curr->virtual_loss--;
                        curr->visit_count++;
                        curr->add_value(value);
                     } else if (curr == root.get()) {
                         curr->visit_count++;
                         curr->add_value(value);
                     }
                     curr = curr->parent;
                }
                batch_valid[b] = false; // Do not eval
            } else {
                // Flatten board to buffer
                int offset = b * 361;
                const int* b_ptr = scratch_board.board;
                for(int i=0; i<361; ++i) batch_boards[offset + i] = b_ptr[i];
                
                batch_valid[b] = true;
            }
        }
        
        // Compact the batch for Python?
        // Python expects contiguous batch.
        
        int valid_count = 0;
        for(bool v : batch_valid) if(v) valid_count++;
        
        if (valid_count == 0) continue;

        // 2. Batch Evaluation (Sequential / GPU)
        py_eval_callback(G_BATCH_SIZE, batch_boards.data(), batch_policies.data(), batch_values.data());
        
        // 3. Expansion & Backpropagation (Sequential for now, or Parallel?)
        
        for (int i = 0; i < G_BATCH_SIZE; ++i) {
            if (!batch_valid[i]) continue;

            MCTSNode* node = batch_nodes[i];
            
            bool already_expanded = !node->is_leaf();
            
            float val_ret = batch_values[i];
            const Connect6Board& s_board = batch_scratch_boards[i];
            float value = (s_board.current_player == BLACK) ? val_ret : -val_ret;
            
            if (!already_expanded) {
                // Expand logic
                float* policy = &batch_policies[i * 361];
                std::vector<int> legal_moves = s_board.get_legal_moves();
                
                struct MoveProb { int move; float prob; };
                std::vector<MoveProb> candidates;
                candidates.reserve(legal_moves.size());
                
                float policy_sum = 0.0f;
                for (int m : legal_moves) {
                    policy_sum += policy[m];
                    candidates.push_back({m, policy[m]});
                }
                
                // Normalize
                if (policy_sum > 1e-8) {
                    for (auto& item : candidates) item.prob /= policy_sum;
                }

                // Top K
                const int TOP_K = 20;
                if (candidates.size() > TOP_K) {
                    std::partial_sort(candidates.begin(), candidates.begin() + TOP_K, candidates.end(),
                        [](const MoveProb& a, const MoveProb& b) { return a.prob > b.prob; });
                    candidates.resize(TOP_K);
                }
                
                // Re-normalize subset
                float subset_sum = 0.0f;
                for (const auto& item : candidates) subset_sum += item.prob;
                
                for (const auto& item : candidates) {
                    float prior = item.prob / (subset_sum + 1e-8);
                    
                    int next_ply = s_board.current_player;
                    int stones = s_board.stones_in_turn + 1;
                    int total = s_board.total_stones + 1;
                    if (total == 1 || stones >= 2) next_ply = -next_ply;
                    
                    node->children.push_back(std::make_unique<MCTSNode>(
                        item.move, prior, s_board.current_player, next_ply, node
                    ));
                }
            }
            
            // Backprop
            MCTSNode* curr = node;
            while (curr != nullptr) {
                if (curr != root.get() && curr->parent) {
                    curr->virtual_loss--;
                    curr->visit_count++;
                    curr->add_value(value);
                } else if (curr == root.get()) {
                    curr->visit_count++;
                    curr->add_value(value);
                }
                curr = curr->parent;
            }
        }
    }
}

EXPORT int get_best_move(float temperature) {
    if (!root || root->children.empty()) return -1;
    
    if (temperature == 0.0f) {
        // Argmax (Deterministic)
        int best_move = -1;
        int max_visits = -1;
        
        for (auto& child : root->children) {
            int vc = child->visit_count.load();
            if (vc > max_visits) {
                max_visits = vc;
                best_move = child->move;
            }
        }
        return best_move;
    } else {
        // Softmax Sampling (Probabilistic)
        std::vector<float> probs;
        std::vector<int> moves;
        float sum = 0.0f;
        
        for (auto& child : root->children) {
            // N^(1/T)
            if (child->visit_count > 0) {
                float p = std::pow((float)child->visit_count, 1.0f / temperature);
                probs.push_back(p);
                moves.push_back(child->move);
                sum += p;
            }
        }
        
        if (sum == 0.0f) return -1;
        
        // Sample
        std::uniform_real_distribution<float> dist(0.0f, sum);
        float r = dist(rng);
        float running_sum = 0.0f;
        
        for (size_t i = 0; i < probs.size(); ++i) {
            running_sum += probs[i];
            if (r <= running_sum) {
                return moves[i];
            }
        }
        return moves.back();
    }
}

// Get Root Win Rate (Black Perspective)
EXPORT float get_root_value() {
    if (!root) return 0.0f;
    return root->q_value(); 
}

// Get Policy Distribution (361 dims)
// output must be float[361]
EXPORT void get_policy(float* output) {
    // Clear
    std::fill(output, output + 361, 0.0f);
    
    if (!root || root->children.empty()) return;
    
    float sum_visits = 0.0f;
    for (auto& child : root->children) {
        sum_visits += (float)child->visit_count.load();
    }
    
    if (sum_visits > 0.0f) {
        for (auto& child : root->children) {
            if (child->move >= 0 && child->move < 361) {
                output[child->move] = (float)child->visit_count.load() / sum_visits;
            }
        }
    }
}

EXPORT void print_top_moves() {
    if (!root) return;
    
    std::vector<MCTSNode*> children;
    for (auto& child : root->children) {
        children.push_back(child.get());
    }
    
    // Sort by visit count (descending)
    std::sort(children.begin(), children.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->visit_count.load() > b->visit_count.load();
    });
    
    int count = 0;
    for (auto* child : children) {
        if (count >= 5) break;
        
        int r = child->move / BOARD_SIZE;
        int c = child->move % BOARD_SIZE;
        char col_char = 'A' + c;
        // Q value is Black perspective
        float q = child->q_value();
        // Convert to human readable win rate %
        float wr = (q + 1.0f) / 2.0f * 100.0f;
        
        std::cout << "  " << col_char << (r + 1) 
                  << " | Visits: " << child->visit_count.load()
                  << " | WinRate(Black): " << wr << "%"
                  << " | Prior: " << child->prior_prob 
                  << std::endl;
        count++;
    }
}

}
