#include "c6_logic.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include <omp.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

namespace {

constexpr long long VALUE_SCALE = 1000000LL;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int PRIOR_SORT_PREFIX = 64;
constexpr int PAIR_RANK = 16;
constexpr int PAIR_FACTOR_COUNT = BOARD_CELLS * PAIR_RANK;

struct PairData {
    std::array<uint16_t, PAIR_FACTOR_COUNT> candidate_factors{};
    std::array<uint16_t, PAIR_FACTOR_COUNT> first_factors{};
    std::array<uint16_t, BOARD_CELLS> values{};
};

struct MCTSNode;

struct MCTSEdge {
    int move;
    std::atomic<int> visit_count;
    std::atomic<int> virtual_loss;
    std::atomic<long long> value_sum_scaled;
    float prior_prob;
    std::unique_ptr<MCTSNode> child;

    MCTSEdge(int move_value, float prior)
        : move(move_value),
          visit_count(0),
          virtual_loss(0),
          value_sum_scaled(0),
          prior_prob(prior) {}

    MCTSEdge(const MCTSEdge&) = delete;
    MCTSEdge& operator=(const MCTSEdge&) = delete;
    MCTSEdge(MCTSEdge&& other) noexcept;
    MCTSEdge& operator=(MCTSEdge&& other) noexcept;
    ~MCTSEdge();

    float get_value_sum() const {
        return static_cast<float>(value_sum_scaled.load(std::memory_order_relaxed))
            / static_cast<float>(VALUE_SCALE);
    }

    void add_value(float value) {
        value_sum_scaled.fetch_add(
            static_cast<long long>(value * static_cast<float>(VALUE_SCALE)),
            std::memory_order_relaxed
        );
    }

    float q_value() const {
        const int completed = visit_count.load(std::memory_order_relaxed);
        const int pending = virtual_loss.load(std::memory_order_relaxed);
        const int total = completed + pending;
        if (total == 0) {
            return 0.0f;
        }
        return (get_value_sum() - static_cast<float>(pending)) / static_cast<float>(total);
    }
};

struct MCTSNode {
    int last_move;
    int next_player;
    MCTSEdge* incoming_edge;
    std::atomic<int> root_visit_count;
    std::atomic<long long> root_value_sum_scaled;
    bool expanded;
    bool provisional_pair;
    std::atomic<bool> pair_refresh_queued;
    std::shared_ptr<const PairData> pair_data;
    std::vector<MCTSEdge> children;

    MCTSNode(int move_value, int next_player_value, MCTSEdge* incoming = nullptr)
        : last_move(move_value),
          next_player(next_player_value),
          incoming_edge(incoming),
          root_visit_count(0),
          root_value_sum_scaled(0),
          expanded(false),
          provisional_pair(false),
          pair_refresh_queued(false) {}

    bool is_leaf() const {
        return !expanded;
    }

    int total_visits() const {
        if (incoming_edge != nullptr) {
            return incoming_edge->visit_count.load(std::memory_order_relaxed)
                + incoming_edge->virtual_loss.load(std::memory_order_relaxed);
        }
        return root_visit_count.load(std::memory_order_relaxed);
    }

    float root_value_sum() const {
        return static_cast<float>(root_value_sum_scaled.load(std::memory_order_relaxed))
            / static_cast<float>(VALUE_SCALE);
    }

    float root_q_value() const {
        const int visits = root_visit_count.load(std::memory_order_relaxed);
        if (visits == 0) {
            return 0.0f;
        }
        return root_value_sum() / static_cast<float>(visits);
    }
};

MCTSEdge::MCTSEdge(MCTSEdge&& other) noexcept
    : move(other.move),
      visit_count(other.visit_count.load(std::memory_order_relaxed)),
      virtual_loss(other.virtual_loss.load(std::memory_order_relaxed)),
      value_sum_scaled(other.value_sum_scaled.load(std::memory_order_relaxed)),
      prior_prob(other.prior_prob),
      child(std::move(other.child)) {
    if (child) {
        child->incoming_edge = this;
    }
}

MCTSEdge& MCTSEdge::operator=(MCTSEdge&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    move = other.move;
    visit_count.store(
        other.visit_count.load(std::memory_order_relaxed),
        std::memory_order_relaxed
    );
    virtual_loss.store(
        other.virtual_loss.load(std::memory_order_relaxed),
        std::memory_order_relaxed
    );
    value_sum_scaled.store(
        other.value_sum_scaled.load(std::memory_order_relaxed),
        std::memory_order_relaxed
    );
    prior_prob = other.prior_prob;
    child = std::move(other.child);
    if (child) {
        child->incoming_edge = this;
    }
    return *this;
}

MCTSEdge::~MCTSEdge() = default;

struct MCTSContext {
    Connect6Board board;
    std::unique_ptr<MCTSNode> root;
    std::mt19937 rng;

    explicit MCTSContext(int seed = 42) : rng(seed) {
        reset();
    }

    void reset() {
        board.reset();
        root = std::make_unique<MCTSNode>(-1, BLACK, nullptr);
    }
};

// 缓存键显式包含轮次状态。正常棋局中它可由棋子数推出，但这样做可以防止
// 调试或棋谱导入产生的非标准状态错误共享网络结果。
struct BoardKey {
    std::array<int8_t, BOARD_CELLS> cells{};
    int8_t current_player = BLACK;
    int8_t stones_in_turn = 0;
    int16_t total_stones = 0;
    uint64_t board_hash = 0;

    bool operator==(const BoardKey& other) const {
        return board_hash == other.board_hash
            && current_player == other.current_player
            && stones_in_turn == other.stones_in_turn
            && total_stones == other.total_stones
            && cells == other.cells;
    }
};

struct BoardKeyHash {
    std::size_t operator()(const BoardKey& key) const noexcept {
        // 棋子布局哈希由 make_move 增量维护；这里只混入轮次状态。最终仍会
        // 逐格比较 cells，因此 64 位碰撞不会错误复用网络结果。
        uint64_t value = key.board_hash;
        value ^= static_cast<uint64_t>(static_cast<int>(key.current_player) + 2)
            * 0x9e3779b97f4a7c15ULL;
        value ^= static_cast<uint64_t>(static_cast<int>(key.stones_in_turn) + 1)
            * 0xbf58476d1ce4e5b9ULL;
        value ^= static_cast<uint64_t>(key.total_stones) * 0x94d049bb133111ebULL;
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        return static_cast<std::size_t>(value);
    }
};

struct EvalCacheEntry {
    std::array<float, BOARD_CELLS> policy{};
    float value = 0.0f;
    std::shared_ptr<const PairData> pair_data;
};

using EvalCallback = void (*)(int, const int*, float*, float*);
using PairEvalCallback = void (*)(
    int,
    const int*,
    float*,
    float*,
    uint16_t*,
    uint16_t*,
    uint16_t*
);

MCTSContext G_DEFAULT_CONTEXT(42);
int G_BATCH_SIZE = 32;
int G_NUM_THREADS = 4;
float G_CPUCT = 1.5f;
int G_PROGRESSIVE_WIDENING_BASE = 20;
float G_PROGRESSIVE_WIDENING_SCALE = 0.25f;
bool G_DETERMINISTIC_TREE_SELECTION = false;
EvalCallback G_EVAL_CALLBACK = nullptr;
PairEvalCallback G_PAIR_EVAL_CALLBACK = nullptr;
float G_PAIR_BASE_SCALE = 1.0f;
float G_PAIR_VALUE_SCALE = 1.0f;
int G_PAIR_REFRESH_VISITS = 2;
int G_PAIR_REFRESH_VISITS_BLACK = 2;
int G_PAIR_REFRESH_VISITS_WHITE = 2;
std::array<float, BOARD_CELLS * BOARD_CELLS> G_PAIR_RELATIVE_LOOKUP{};
std::array<float, (BOARD_SIZE * 2 - 1) * (BOARD_SIZE * 2 - 1) * PAIR_RANK>
    G_PAIR_RELATIVE_GATE{};
bool G_PAIR_RELATIVE_GATE_READY = false;
bool G_PAIR_PARAMS_READY = false;
bool G_PAIR_DEFERRED_REFRESH = false;

// 这些统计针对一个 C++ 动态库（也就是一张 GPU 的推理服务），能直接反映
// 跨棋局合批和持久缓存对实际网络调用量的影响。
long long G_TOTAL_LEAF_REQUESTS = 0;
long long G_TOTAL_SEARCH_BATCHES = 0;
long long G_TOTAL_BATCH_UNIQUE_POSITIONS = 0;
long long G_TOTAL_NEURAL_EVALUATIONS = 0;
long long G_TOTAL_SECOND_STONE_EVALUATIONS = 0;
long long G_TOTAL_EVAL_BATCHES = 0;
long long G_TOTAL_CACHE_HITS = 0;
long long G_TOTAL_CACHE_MISSES = 0;
long long G_TOTAL_PAIR_PROVISIONAL_EVALUATIONS = 0;
long long G_TOTAL_PAIR_EXACT_REFRESHES = 0;
long long G_TOTAL_PAIR_EAGER_EXACT_EVALUATIONS = 0;

int pair_refresh_visits(int player) {
    return player == BLACK
        ? G_PAIR_REFRESH_VISITS_BLACK
        : G_PAIR_REFRESH_VISITS_WHITE;
}

std::size_t G_EVAL_CACHE_CAPACITY = 32768;
std::unordered_map<BoardKey, EvalCacheEntry, BoardKeyHash> G_EVAL_CACHE;
std::deque<BoardKey> G_EVAL_CACHE_FIFO;

struct PairRefreshJob {
    MCTSNode* node = nullptr;
    Connect6Board board;
    BoardKey key;
};

struct SearchWorkspace {
    std::vector<MCTSContext*> batch_contexts;
    std::vector<MCTSNode*> batch_leaf_nodes;
    std::vector<MCTSEdge*> batch_leaf_edges;
    std::vector<MCTSNode*> batch_leaf_parents;
    std::vector<MCTSEdge*> batch_paths;
    std::vector<int> batch_path_lengths;
    std::vector<Connect6Board> batch_boards;
    std::vector<uint8_t> batch_valid;
    std::vector<uint8_t> batch_force_exact;
    std::vector<uint8_t> batch_can_use_pair;
    std::vector<MCTSNode*> batch_refresh_nodes;
    std::vector<Connect6Board> batch_refresh_boards;
    std::vector<int> eval_slot;
    std::vector<BoardKey> unique_keys;
    std::vector<int> unique_boards;
    std::vector<float> unique_policies;
    std::vector<float> unique_values;
    std::vector<uint8_t> unique_use_pair;
    std::vector<uint8_t> unique_defer_refresh;
    std::vector<int> unique_request_counts;
    std::vector<int> unique_first_moves;
    std::vector<MCTSNode*> unique_pair_parents;
    std::vector<std::shared_ptr<const PairData>> unique_pair_data;
    std::vector<int> callback_boards;
    std::vector<float> callback_policies;
    std::vector<float> callback_values;
    std::vector<uint16_t> callback_candidate_factors;
    std::vector<uint16_t> callback_first_factors;
    std::vector<uint16_t> callback_pair_values;
    std::vector<int> miss_to_unique;
    std::vector<MCTSContext*> selection_contexts;
    std::vector<uintptr_t> leaf_hash_keys;
    std::vector<int> leaf_hash_values;
    std::vector<int> board_hash_values;
    int hash_capacity = 0;

    void ensure(int request_capacity, int gpu_batch_size, int context_count) {
        batch_contexts.resize(request_capacity);
        batch_leaf_nodes.resize(request_capacity);
        batch_leaf_edges.resize(request_capacity);
        batch_leaf_parents.resize(request_capacity);
        batch_paths.resize(request_capacity * BOARD_CELLS);
        batch_path_lengths.resize(request_capacity);
        batch_boards.resize(request_capacity);
        batch_valid.resize(request_capacity);
        batch_force_exact.resize(request_capacity);
        batch_can_use_pair.resize(request_capacity);
        batch_refresh_nodes.resize(request_capacity);
        batch_refresh_boards.resize(request_capacity);
        eval_slot.resize(request_capacity);
        unique_keys.resize(request_capacity);
        unique_boards.resize(request_capacity * BOARD_CELLS);
        unique_policies.resize(request_capacity * BOARD_CELLS);
        unique_values.resize(request_capacity);
        unique_use_pair.resize(request_capacity);
        unique_defer_refresh.resize(request_capacity);
        unique_request_counts.resize(request_capacity);
        unique_first_moves.resize(request_capacity);
        unique_pair_parents.resize(request_capacity);
        unique_pair_data.resize(request_capacity);
        callback_boards.resize(gpu_batch_size * BOARD_CELLS);
        callback_policies.resize(gpu_batch_size * BOARD_CELLS);
        callback_values.resize(gpu_batch_size);
        callback_candidate_factors.resize(gpu_batch_size * PAIR_FACTOR_COUNT);
        callback_first_factors.resize(gpu_batch_size * PAIR_FACTOR_COUNT);
        callback_pair_values.resize(gpu_batch_size * BOARD_CELLS);
        miss_to_unique.resize(request_capacity);
        selection_contexts.resize(context_count);
        int required_hash_capacity = 1;
        while (required_hash_capacity < request_capacity * 2) {
            required_hash_capacity <<= 1;
        }
        if (required_hash_capacity > hash_capacity) {
            hash_capacity = required_hash_capacity;
            leaf_hash_keys.resize(hash_capacity);
            leaf_hash_values.resize(hash_capacity);
            board_hash_values.resize(hash_capacity);
        }
    }
};

SearchWorkspace G_SEARCH_WORKSPACE;

BoardKey make_board_key(const Connect6Board& board) {
    BoardKey key;
    for (int index = 0; index < BOARD_CELLS; ++index) {
        key.cells[index] = static_cast<int8_t>(board.board[index]);
    }
    key.current_player = static_cast<int8_t>(board.current_player);
    key.stones_in_turn = static_cast<int8_t>(board.stones_in_turn);
    key.total_stones = static_cast<int16_t>(board.total_stones);
    key.board_hash = board.zobrist_hash;
    return key;
}

void clear_eval_cache_impl() {
    G_EVAL_CACHE.clear();
    G_EVAL_CACHE_FIFO.clear();
}

bool lookup_eval_cache(
    const BoardKey& key,
    float* policy,
    float& value,
    std::shared_ptr<const PairData>& pair_data
) {
    if (G_EVAL_CACHE_CAPACITY == 0) {
        return false;
    }
    const auto found = G_EVAL_CACHE.find(key);
    if (found == G_EVAL_CACHE.end()) {
        return false;
    }
    std::copy(found->second.policy.begin(), found->second.policy.end(), policy);
    value = found->second.value;
    pair_data = found->second.pair_data;
    return true;
}

void store_eval_cache(
    const BoardKey& key,
    const float* policy,
    float value,
    const std::shared_ptr<const PairData>& pair_data
) {
    if (G_EVAL_CACHE_CAPACITY == 0) {
        return;
    }

    const auto existing = G_EVAL_CACHE.find(key);
    if (existing != G_EVAL_CACHE.end()) {
        std::copy_n(policy, BOARD_CELLS, existing->second.policy.begin());
        existing->second.value = value;
        existing->second.pair_data = pair_data;
        return;
    }

    while (G_EVAL_CACHE.size() >= G_EVAL_CACHE_CAPACITY && !G_EVAL_CACHE_FIFO.empty()) {
        G_EVAL_CACHE.erase(G_EVAL_CACHE_FIFO.front());
        G_EVAL_CACHE_FIFO.pop_front();
    }

    EvalCacheEntry entry;
    std::copy_n(policy, BOARD_CELLS, entry.policy.begin());
    entry.value = value;
    entry.pair_data = pair_data;
    G_EVAL_CACHE.emplace(key, std::move(entry));
    G_EVAL_CACHE_FIFO.push_back(key);
}

void reset_statistics_impl() {
    G_TOTAL_LEAF_REQUESTS = 0;
    G_TOTAL_SEARCH_BATCHES = 0;
    G_TOTAL_BATCH_UNIQUE_POSITIONS = 0;
    G_TOTAL_NEURAL_EVALUATIONS = 0;
    G_TOTAL_SECOND_STONE_EVALUATIONS = 0;
    G_TOTAL_EVAL_BATCHES = 0;
    G_TOTAL_CACHE_HITS = 0;
    G_TOTAL_CACHE_MISSES = 0;
    G_TOTAL_PAIR_PROVISIONAL_EVALUATIONS = 0;
    G_TOTAL_PAIR_EXACT_REFRESHES = 0;
    G_TOTAL_PAIR_EAGER_EXACT_EVALUATIONS = 0;
}

void backpropagate(
    MCTSNode& root,
    MCTSEdge* const* path,
    int path_length,
    float black_value
) {
    for (int index = 0; index < path_length; ++index) {
        MCTSEdge* edge = path[index];
        edge->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
        edge->visit_count.fetch_add(1, std::memory_order_relaxed);
        edge->add_value(black_value);
    }
    root.root_visit_count.fetch_add(1, std::memory_order_relaxed);
    root.root_value_sum_scaled.fetch_add(
        static_cast<long long>(black_value * static_cast<float>(VALUE_SCALE)),
        std::memory_order_relaxed
    );
}

void expand_node(MCTSNode* node, const Connect6Board& board, const float* policy) {
    if (node->expanded) {
        return;
    }

    // 不再做 top-k 裁剪：每一个合法着法都进入搜索树。低概率动作仍可由
    // PUCT 自然控制访问量，因此不会丢掉网络排名靠后但战术上关键的应手。
    struct MovePrior {
        int move;
        float prior;
    };
    std::array<MovePrior, BOARD_CELLS> move_priors{};
    int legal_count = 0;
    float policy_sum = 0.0f;
    for (int move = 0; move < BOARD_CELLS; ++move) {
        if (board.board[move] != EMPTY) {
            continue;
        }
        const float raw_prior = policy[move];
        const float safe_prior = std::isfinite(raw_prior) ? std::max(raw_prior, 0.0f) : 0.0f;
        move_priors[legal_count++] = {move, safe_prior};
        policy_sum += safe_prior;
    }
    if (legal_count == 0) {
        node->expanded = true;
        return;
    }

    const bool use_uniform = policy_sum <= 1e-8f;
    const float uniform_prior = 1.0f / static_cast<float>(legal_count);
    for (int index = 0; index < legal_count; ++index) {
        MovePrior& item = move_priors[index];
        item.prior = use_uniform ? uniform_prior : item.prior / policy_sum;
    }
    // 400/1200 搜索下渐进激活远小于 64，只精排前 64 就与完整排序严格等价；
    // 其余合法动作仍永久保留，超大预算继续扩展时也可以参与 PUCT。
    const auto prior_order = [](const MovePrior& left, const MovePrior& right) {
        if (left.prior != right.prior) {
            return left.prior > right.prior;
        }
        return left.move < right.move;
    };
    const std::size_t sorted_prefix = std::min<std::size_t>(
        PRIOR_SORT_PREFIX,
        static_cast<std::size_t>(legal_count)
    );
    if (sorted_prefix < static_cast<std::size_t>(legal_count)) {
        std::partial_sort(
            move_priors.begin(),
            move_priors.begin() + sorted_prefix,
            move_priors.begin() + legal_count,
            prior_order
        );
    } else {
        std::sort(move_priors.begin(), move_priors.begin() + legal_count, prior_order);
    }
    node->children.reserve(legal_count);

    for (int index = 0; index < legal_count; ++index) {
        const MovePrior& item = move_priors[index];
        const int move = item.move;
        const float prior = item.prior;

        // 这里只建立紧凑边；真正选中该动作时才创建子节点，避免为几百个
        // 低概率动作执行独立堆分配。
        node->children.emplace_back(move, prior);
    }
    node->expanded = true;
}

float half_to_float(uint16_t value) {
#if defined(__F16C__)
    return _cvtsh_ss(value);
#else
    // 非 F16C 主机的精确 IEEE-754 half 回退路径。
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    int32_t exponent = static_cast<int32_t>((value >> 10) & 0x1fu);
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ffu;
            bits = sign
                | (static_cast<uint32_t>(exponent + 112) << 23)
                | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign
            | (static_cast<uint32_t>(exponent + 112) << 23)
            | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
#endif
}

void half_factors_to_float(const uint16_t* source, float* destination) {
#if defined(__F16C__) && defined(__AVX2__)
    for (int offset = 0; offset < PAIR_RANK; offset += 8) {
        const __m128i source_half = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source + offset)
        );
        _mm256_storeu_ps(
            destination + offset,
            _mm256_cvtph_ps(source_half)
        );
    }
#else
    for (int index = 0; index < PAIR_RANK; ++index) {
        destination[index] = half_to_float(source[index]);
    }
#endif
}

float pair_factor_dot(
    const uint16_t* candidate,
    const float* first,
    const float* relative_gate
) {
#if defined(__F16C__) && defined(__AVX2__)
    __m256 accumulator = _mm256_setzero_ps();
    for (int offset = 0; offset < PAIR_RANK; offset += 8) {
        const __m128i candidate_half = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(candidate + offset)
        );
        const __m256 candidate_float = _mm256_cvtph_ps(candidate_half);
        __m256 product = _mm256_mul_ps(
            candidate_float,
            _mm256_loadu_ps(first + offset)
        );
        if (relative_gate != nullptr) {
            product = _mm256_mul_ps(
                product,
                _mm256_loadu_ps(relative_gate + offset)
            );
        }
        accumulator = _mm256_add_ps(accumulator, product);
    }
    const __m128 low = _mm256_castps256_ps128(accumulator);
    const __m128 high = _mm256_extractf128_ps(accumulator, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
#else
    float result = 0.0f;
    for (int index = 0; index < PAIR_RANK; ++index) {
        const float gate = relative_gate != nullptr ? relative_gate[index] : 1.0f;
        result += half_to_float(candidate[index])
            * first[index]
            * gate;
    }
    return result;
#endif
}

void synthesize_pair_evaluation(
    const MCTSNode& parent,
    int first_move,
    float* policy,
    float& value
) {
    std::fill(policy, policy + BOARD_CELLS, 0.0f);
    if (!parent.pair_data || first_move < 0 || first_move >= BOARD_CELLS) {
        value = 0.0f;
        return;
    }

    const PairData& pair = *parent.pair_data;
    const uint16_t* first = pair.first_factors.data() + first_move * PAIR_RANK;
    // 第一子的低秩向量会与全部候选第二子复用，只转换一次 FP16，避免
    // 在整张棋盘的点积中重复解码同一组 16 个数。
    alignas(32) std::array<float, PAIR_RANK> first_float{};
    half_factors_to_float(first, first_float.data());
    std::array<float, BOARD_CELLS> logits{};
    float maximum = -std::numeric_limits<float>::infinity();
    const float rank_scale = 1.0f / std::sqrt(static_cast<float>(PAIR_RANK));
    const int first_row = first_move / BOARD_SIZE;
    const int first_column = first_move % BOARD_SIZE;
    for (const MCTSEdge& edge : parent.children) {
        if (edge.move == first_move) {
            continue;
        }
        const uint16_t* candidate = pair.candidate_factors.data()
            + edge.move * PAIR_RANK;
        const int relative_row = edge.move / BOARD_SIZE
            - first_row + BOARD_SIZE - 1;
        const int relative_column = edge.move % BOARD_SIZE
            - first_column + BOARD_SIZE - 1;
        const int relative_index = relative_row * (BOARD_SIZE * 2 - 1)
            + relative_column;
        const float* relative_gate = G_PAIR_RELATIVE_GATE_READY
            ? G_PAIR_RELATIVE_GATE.data() + relative_index * PAIR_RANK
            : nullptr;
        const float logit = G_PAIR_BASE_SCALE * std::log(
            std::max(edge.prior_prob, 1e-12f)
        ) + pair_factor_dot(candidate, first_float.data(), relative_gate) * rank_scale
            + G_PAIR_RELATIVE_LOOKUP[first_move * BOARD_CELLS + edge.move];
        logits[edge.move] = logit;
        maximum = std::max(maximum, logit);
    }

    value = G_PAIR_VALUE_SCALE * half_to_float(pair.values[first_move]);
    if (!std::isfinite(maximum)) {
        return;
    }
    float sum = 0.0f;
    for (const MCTSEdge& edge : parent.children) {
        if (edge.move == first_move) {
            continue;
        }
        const float probability = std::exp(logits[edge.move] - maximum);
        policy[edge.move] = probability;
        sum += probability;
    }
    if (sum > 0.0f) {
        const float inverse_sum = 1.0f / sum;
        for (int move = 0; move < BOARD_CELLS; ++move) {
            policy[move] *= inverse_sum;
        }
    }
}

void refresh_node_priors(
    MCTSNode* node,
    const Connect6Board& board,
    const float* policy
) {
    if (node == nullptr || !node->expanded) {
        return;
    }
    float policy_sum = 0.0f;
    for (MCTSEdge& edge : node->children) {
        const float raw_prior = policy[edge.move];
        edge.prior_prob = std::isfinite(raw_prior)
            ? std::max(raw_prior, 0.0f)
            : 0.0f;
        if (board.board[edge.move] == EMPTY) {
            policy_sum += edge.prior_prob;
        }
    }
    const bool use_uniform = policy_sum <= 1e-8f;
    const float uniform_prior = node->children.empty()
        ? 0.0f
        : 1.0f / static_cast<float>(node->children.size());
    for (MCTSEdge& edge : node->children) {
        edge.prior_prob = use_uniform
            ? uniform_prior
            : edge.prior_prob / policy_sum;
    }
    const auto prior_order = [](const MCTSEdge& left, const MCTSEdge& right) {
        if (left.prior_prob != right.prior_prob) {
            return left.prior_prob > right.prior_prob;
        }
        return left.move < right.move;
    };
    const std::size_t sorted_prefix = std::min<std::size_t>(
        PRIOR_SORT_PREFIX,
        node->children.size()
    );
    if (sorted_prefix < node->children.size()) {
        std::partial_sort(
            node->children.begin(),
            node->children.begin() + sorted_prefix,
            node->children.end(),
            prior_order
        );
    } else {
        std::sort(node->children.begin(), node->children.end(), prior_order);
    }
    node->provisional_pair = false;
}

int active_child_count(const MCTSNode& node) {
    const int visits = node.total_visits();
    const int progressively_active = G_PROGRESSIVE_WIDENING_BASE
        + static_cast<int>(G_PROGRESSIVE_WIDENING_SCALE * std::sqrt(static_cast<float>(visits)));
    return std::min<int>(
        node.children.size(),
        std::max(progressively_active, 1)
    );
}

void play_move_impl(MCTSContext& context, int move) {
    if (move < 0 || move >= BOARD_CELLS || context.board.board[move] != EMPTY) {
        return;
    }

    context.board.make_move(move);
    if (context.root) {
        for (auto& edge : context.root->children) {
            if (edge.move == move && edge.child) {
                const int visits = edge.visit_count.load(std::memory_order_relaxed);
                const long long value_sum = edge.value_sum_scaled.load(std::memory_order_relaxed);
                std::unique_ptr<MCTSNode> reused = std::move(edge.child);
                reused->incoming_edge = nullptr;
                reused->root_visit_count.store(visits, std::memory_order_relaxed);
                reused->root_value_sum_scaled.store(value_sum, std::memory_order_relaxed);
                context.root = std::move(reused);
                return;
            }
        }
    }

    context.root = std::make_unique<MCTSNode>(move, context.board.current_player, nullptr);
}

int best_move_impl(MCTSContext& context, float temperature) {
    if (!context.root || context.root->children.empty()) {
        return -1;
    }

    if (temperature <= 0.0f) {
        int best_move = -1;
        int best_visits = -1;
        for (const auto& edge : context.root->children) {
            const int visits = edge.visit_count.load(std::memory_order_relaxed);
            if (visits > best_visits) {
                best_visits = visits;
                best_move = edge.move;
            }
        }
        return best_move;
    }

    std::vector<float> weights;
    std::vector<int> moves;
    float weight_sum = 0.0f;
    for (const auto& edge : context.root->children) {
        const int visits = edge.visit_count.load(std::memory_order_relaxed);
        if (visits <= 0) {
            continue;
        }
        const float weight = std::pow(static_cast<float>(visits), 1.0f / temperature);
        weights.push_back(weight);
        moves.push_back(edge.move);
        weight_sum += weight;
    }
    if (weight_sum <= 0.0f) {
        // 极小模拟预算可能只够展开根节点，子节点还没有访问量。此时按网络
        // 先验采样，既不会返回非法着法，也比固定选择第一个空位更合理。
        weights.clear();
        moves.clear();
        for (const auto& edge : context.root->children) {
            const float weight = std::pow(
                std::max(edge.prior_prob, 0.0f),
                1.0f / temperature
            );
            weights.push_back(weight);
            moves.push_back(edge.move);
            weight_sum += weight;
        }
        if (weight_sum <= 0.0f) {
            std::uniform_int_distribution<int> uniform_move(
                0,
                static_cast<int>(moves.size()) - 1
            );
            return moves[uniform_move(context.rng)];
        }
    }

    std::uniform_real_distribution<float> distribution(0.0f, weight_sum);
    const float sample = distribution(context.rng);
    float cumulative = 0.0f;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        cumulative += weights[index];
        if (sample <= cumulative) {
            return moves[index];
        }
    }
    return moves.back();
}

void policy_impl(const MCTSContext& context, float* output) {
    std::fill(output, output + BOARD_CELLS, 0.0f);
    if (!context.root || context.root->children.empty()) {
        return;
    }

    float visit_sum = 0.0f;
    for (const auto& edge : context.root->children) {
        visit_sum += static_cast<float>(edge.visit_count.load(std::memory_order_relaxed));
    }
    if (visit_sum <= 0.0f) {
        return;
    }
    for (const auto& edge : context.root->children) {
        output[edge.move] = static_cast<float>(edge.visit_count.load(std::memory_order_relaxed))
            / visit_sum;
    }
}

void print_top_moves_impl(const MCTSContext& context) {
    if (!context.root) {
        return;
    }
    std::vector<const MCTSEdge*> children;
    children.reserve(context.root->children.size());
    for (const auto& edge : context.root->children) {
        children.push_back(&edge);
    }
    std::sort(children.begin(), children.end(), [](const MCTSEdge* left, const MCTSEdge* right) {
        return left->visit_count.load(std::memory_order_relaxed)
            > right->visit_count.load(std::memory_order_relaxed);
    });

    const int count = std::min<int>(5, children.size());
    for (int index = 0; index < count; ++index) {
        const MCTSEdge* edge = children[index];
        const int row = edge->move / BOARD_SIZE;
        const int column = edge->move % BOARD_SIZE;
        const float win_rate = (edge->q_value() + 1.0f) * 50.0f;
        std::cout << "  " << static_cast<char>('A' + column) << (row + 1)
                  << " | Visits: " << edge->visit_count.load(std::memory_order_relaxed)
                  << " | WinRate(Black): " << win_rate << "%"
                  << " | Prior: " << edge->prior_prob << std::endl;
    }
}

void accumulate_second_stone_visits(
    const MCTSNode& node,
    int total_stones,
    const int* thresholds,
    int threshold_count,
    long long* output
) {
    // n>=2 的偶数棋子局面表示当前玩家已经落下本回合第一颗子。
    if (node.expanded && total_stones >= 2 && total_stones % 2 == 0) {
        const int visits = node.total_visits();
        for (int index = 0; index < threshold_count; ++index) {
            if (visits >= thresholds[index]) {
                output[index]++;
            }
        }
    }
    for (const MCTSEdge& edge : node.children) {
        if (edge.child) {
            accumulate_second_stone_visits(
                *edge.child,
                total_stones + 1,
                thresholds,
                threshold_count,
                output
            );
        }
    }
}

void run_simulations_impl(
    const std::vector<MCTSContext*>& contexts,
    const std::vector<int>& requested_simulations
) {
    if (
        (G_EVAL_CALLBACK == nullptr && G_PAIR_EVAL_CALLBACK == nullptr)
        || contexts.empty()
        || contexts.size() != requested_simulations.size()
    ) {
        return;
    }

    std::vector<int> remaining(requested_simulations.size(), 0);
    long long total_remaining = 0;
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        if (contexts[index] != nullptr && requested_simulations[index] > 0) {
            remaining[index] = requested_simulations[index];
            total_remaining += requested_simulations[index];
        }
    }
    if (total_remaining == 0) {
        return;
    }

    // 每棵树仍保留原来的 G_BATCH_SIZE 个虚拟选择，多个棋局只在网络评估
    // 层合批。这样不会因为并发棋局数增加而改变单棵树的搜索形状。
    const int request_capacity = G_BATCH_SIZE * static_cast<int>(contexts.size());
    G_SEARCH_WORKSPACE.ensure(
        request_capacity,
        G_BATCH_SIZE,
        static_cast<int>(contexts.size())
    );
    auto& batch_contexts = G_SEARCH_WORKSPACE.batch_contexts;
    auto& batch_leaf_nodes = G_SEARCH_WORKSPACE.batch_leaf_nodes;
    auto& batch_leaf_edges = G_SEARCH_WORKSPACE.batch_leaf_edges;
    auto& batch_leaf_parents = G_SEARCH_WORKSPACE.batch_leaf_parents;
    auto& batch_paths = G_SEARCH_WORKSPACE.batch_paths;
    auto& batch_path_lengths = G_SEARCH_WORKSPACE.batch_path_lengths;
    auto& batch_boards = G_SEARCH_WORKSPACE.batch_boards;
    auto& batch_valid = G_SEARCH_WORKSPACE.batch_valid;
    auto& batch_force_exact = G_SEARCH_WORKSPACE.batch_force_exact;
    auto& batch_can_use_pair = G_SEARCH_WORKSPACE.batch_can_use_pair;
    auto& batch_refresh_nodes = G_SEARCH_WORKSPACE.batch_refresh_nodes;
    auto& batch_refresh_boards = G_SEARCH_WORKSPACE.batch_refresh_boards;
    auto& eval_slot = G_SEARCH_WORKSPACE.eval_slot;
    auto& unique_keys = G_SEARCH_WORKSPACE.unique_keys;
    auto& unique_boards = G_SEARCH_WORKSPACE.unique_boards;
    auto& unique_policies = G_SEARCH_WORKSPACE.unique_policies;
    auto& unique_values = G_SEARCH_WORKSPACE.unique_values;
    auto& unique_use_pair = G_SEARCH_WORKSPACE.unique_use_pair;
    auto& unique_defer_refresh = G_SEARCH_WORKSPACE.unique_defer_refresh;
    auto& unique_request_counts = G_SEARCH_WORKSPACE.unique_request_counts;
    auto& unique_first_moves = G_SEARCH_WORKSPACE.unique_first_moves;
    auto& unique_pair_parents = G_SEARCH_WORKSPACE.unique_pair_parents;
    auto& unique_pair_data = G_SEARCH_WORKSPACE.unique_pair_data;
    auto& callback_boards = G_SEARCH_WORKSPACE.callback_boards;
    auto& callback_policies = G_SEARCH_WORKSPACE.callback_policies;
    auto& callback_values = G_SEARCH_WORKSPACE.callback_values;
    auto& callback_candidate_factors = G_SEARCH_WORKSPACE.callback_candidate_factors;
    auto& callback_first_factors = G_SEARCH_WORKSPACE.callback_first_factors;
    auto& callback_pair_values = G_SEARCH_WORKSPACE.callback_pair_values;
    auto& miss_to_unique = G_SEARCH_WORKSPACE.miss_to_unique;
    auto& selection_contexts = G_SEARCH_WORKSPACE.selection_contexts;
    auto& leaf_hash_keys = G_SEARCH_WORKSPACE.leaf_hash_keys;
    auto& leaf_hash_values = G_SEARCH_WORKSPACE.leaf_hash_values;
    auto& board_hash_values = G_SEARCH_WORKSPACE.board_hash_values;
    std::deque<PairRefreshJob> refresh_queue;
    std::vector<PairRefreshJob> callback_refresh_jobs(G_BATCH_SIZE);

    while (total_remaining > 0) {
        int active_batch = 0;
        for (std::size_t context_index = 0; context_index < contexts.size(); ++context_index) {
            if (remaining[context_index] <= 0 || contexts[context_index] == nullptr) {
                continue;
            }
            const int tree_batch = std::min(G_BATCH_SIZE, remaining[context_index]);
            for (int request = 0; request < tree_batch; ++request) {
                batch_contexts[active_batch] = contexts[context_index];
                active_batch++;
            }
            remaining[context_index] -= tree_batch;
            total_remaining -= tree_batch;
        }

        int selection_context_count = 0;
        for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
            MCTSContext* context = batch_contexts[batch_index];
            bool already_present = false;
            for (int index = 0; index < selection_context_count; ++index) {
                if (selection_contexts[index] == context) {
                    already_present = true;
                    break;
                }
            }
            if (!already_present) {
                selection_contexts[selection_context_count++] = context;
            }
        }

        auto select_one_request = [&](int batch_index) {
            MCTSContext* context = batch_contexts[batch_index];
            Connect6Board scratch = context->board;
            MCTSNode* node = context->root.get();
            MCTSEdge* leaf_edge = nullptr;
            MCTSNode* leaf_parent = nullptr;
            bool force_exact = false;
            MCTSEdge** path = batch_paths.data() + batch_index * BOARD_CELLS;
            int path_length = 0;
            batch_refresh_nodes[batch_index] = nullptr;

            while (node != nullptr && node->expanded && !node->children.empty()) {
                if (
                    node->provisional_pair
                    && node->total_visits() >= pair_refresh_visits(node->next_player)
                ) {
                    if (G_PAIR_DEFERRED_REFRESH) {
                        if (batch_refresh_nodes[batch_index] == nullptr) {
                            bool expected = false;
                            if (node->pair_refresh_queued.compare_exchange_strong(
                                    expected,
                                    true,
                                    std::memory_order_relaxed
                                )) {
                                batch_refresh_nodes[batch_index] = node;
                                batch_refresh_boards[batch_index] = scratch;
                            }
                        }
                    } else {
                        // 同步模式保留为强度基线：再次访问时停在本节点精确刷新。
                        force_exact = true;
                        break;
                    }
                }
                const int parent_visits = node->total_visits();
                const float exploration_scale = std::sqrt(static_cast<float>(std::max(parent_visits, 1)));
                float best_score = -std::numeric_limits<float>::infinity();
                MCTSEdge* best_edge = nullptr;

                const int active_children = active_child_count(*node);
                for (int child_index = 0; child_index < active_children; ++child_index) {
                    MCTSEdge* edge = &node->children[child_index];
                    float q = edge->q_value();
                    if (node->next_player == WHITE) {
                        q = -q;
                    }
                    const int child_visits = edge->visit_count.load(std::memory_order_relaxed)
                        + edge->virtual_loss.load(std::memory_order_relaxed);
                    const float exploration = G_CPUCT * edge->prior_prob * exploration_scale
                        / static_cast<float>(1 + child_visits);
                    const float score = q + exploration;
                    if (score > best_score) {
                        best_score = score;
                        best_edge = edge;
                    }
                }

                if (best_edge == nullptr) {
                    break;
                }
                best_edge->virtual_loss.fetch_add(1, std::memory_order_relaxed);
                path[path_length++] = best_edge;
                scratch.make_move(best_edge->move);
                if (!best_edge->child) {
                    leaf_parent = node;
                    leaf_edge = best_edge;
                    node = nullptr;
                    break;
                }
                node = best_edge->child.get();
            }

            batch_leaf_nodes[batch_index] = node;
            batch_leaf_edges[batch_index] = leaf_edge;
            batch_leaf_parents[batch_index] = leaf_parent;
            batch_path_lengths[batch_index] = path_length;
            batch_boards[batch_index] = scratch;
            batch_valid[batch_index] = 0;
            batch_force_exact[batch_index] = force_exact ? 1 : 0;
            batch_can_use_pair[batch_index] = 0;

            const int last_move = leaf_edge != nullptr
                ? leaf_edge->move
                : (node != nullptr ? node->last_move : -1);
            const int winner = scratch.check_win(last_move);
            if (winner != 0 || scratch.total_stones >= BOARD_CELLS) {
                const float terminal_value = winner == BLACK ? 1.0f : (winner == WHITE ? -1.0f : 0.0f);
                backpropagate(*context->root, path, path_length, terminal_value);
                return;
            }
            if (node == nullptr && leaf_edge == nullptr) {
                return;
            }
            batch_can_use_pair[batch_index] = (
                !force_exact
                && G_PAIR_EVAL_CALLBACK != nullptr
                && G_PAIR_PARAMS_READY
                && scratch.stones_in_turn == 1
                && leaf_parent != nullptr
                && leaf_parent->pair_data != nullptr
            ) ? 1 : 0;
            batch_valid[batch_index] = 1;
        };

        if (G_DETERMINISTIC_TREE_SELECTION) {
            // 验证/门控模式：同一棵树内顺序选择，不同棋局之间并行。
            #pragma omp parallel for schedule(dynamic)
            for (int selection_index = 0; selection_index < selection_context_count; ++selection_index) {
                for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
                    if (batch_contexts[batch_index] == selection_contexts[selection_index]) {
                        select_one_request(batch_index);
                    }
                }
            }
        } else {
            // 生产自对弈模式：保留原实现的同树原子并行，以吞吐和探索多样性优先。
            #pragma omp parallel for schedule(dynamic)
            for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
                select_one_request(batch_index);
            }
        }

        if (G_PAIR_DEFERRED_REFRESH) {
            for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
                if (batch_refresh_nodes[batch_index] == nullptr) {
                    continue;
                }
                PairRefreshJob job;
                job.node = batch_refresh_nodes[batch_index];
                job.board = batch_refresh_boards[batch_index];
                job.key = make_board_key(job.board);
                refresh_queue.push_back(std::move(job));
            }
        }

        int valid_count = 0;
        for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
            valid_count += batch_valid[batch_index] ? 1 : 0;
        }
        if (valid_count == 0) {
            continue;
        }

        G_TOTAL_SEARCH_BATCHES++;

        // 先在整个跨棋局批次内去重，再查询跨批次缓存。这样相同开局不会因为
        // 同时存在于多盘棋而重复占用 GPU batch。
        int unique_count = 0;
        const int hash_capacity = G_SEARCH_WORKSPACE.hash_capacity;
        const int hash_mask = hash_capacity - 1;
        std::fill(leaf_hash_values.begin(), leaf_hash_values.begin() + hash_capacity, -1);
        std::fill(board_hash_values.begin(), board_hash_values.begin() + hash_capacity, -1);
        for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
            if (!batch_valid[batch_index]) {
                continue;
            }

            const bool is_edge = batch_leaf_edges[batch_index] != nullptr;
            const void* leaf_pointer = is_edge
                ? static_cast<const void*>(batch_leaf_edges[batch_index])
                : static_cast<const void*>(batch_leaf_nodes[batch_index]);
            const uintptr_t leaf_identity = reinterpret_cast<uintptr_t>(leaf_pointer)
                | static_cast<uintptr_t>(is_edge ? 1 : 0);
            uint64_t mixed_leaf = static_cast<uint64_t>(leaf_identity);
            mixed_leaf ^= mixed_leaf >> 33;
            mixed_leaf *= 0xff51afd7ed558ccdULL;
            mixed_leaf ^= mixed_leaf >> 33;
            int leaf_bucket = static_cast<int>(mixed_leaf) & hash_mask;
            while (leaf_hash_values[leaf_bucket] >= 0
                   && leaf_hash_keys[leaf_bucket] != leaf_identity) {
                leaf_bucket = (leaf_bucket + 1) & hash_mask;
            }
            if (leaf_hash_values[leaf_bucket] >= 0) {
                const int slot = leaf_hash_values[leaf_bucket];
                eval_slot[batch_index] = slot;
                unique_request_counts[slot]++;
                continue;
            }

            const BoardKey key = make_board_key(batch_boards[batch_index]);
            int slot = -1;
            int board_bucket = static_cast<int>(BoardKeyHash{}(key)) & hash_mask;
            while (board_hash_values[board_bucket] >= 0) {
                const int candidate_slot = board_hash_values[board_bucket];
                if (unique_keys[candidate_slot] == key) {
                    slot = candidate_slot;
                    break;
                }
                board_bucket = (board_bucket + 1) & hash_mask;
            }
            if (slot < 0) {
                slot = unique_count++;
                unique_keys[slot] = key;
                unique_use_pair[slot] = batch_can_use_pair[batch_index];
                unique_defer_refresh[slot] = 0;
                unique_request_counts[slot] = 1;
                unique_first_moves[slot] = batch_leaf_edges[batch_index] != nullptr
                    ? batch_leaf_edges[batch_index]->move
                    : -1;
                unique_pair_parents[slot] = batch_leaf_parents[batch_index];
                unique_pair_data[slot] = batch_leaf_parents[batch_index] != nullptr
                    ? batch_leaf_parents[batch_index]->pair_data
                    : nullptr;
                for (int cell = 0; cell < BOARD_CELLS; ++cell) {
                    unique_boards[slot * BOARD_CELLS + cell] = key.cells[cell];
                }
                board_hash_values[board_bucket] = slot;
            } else {
                unique_request_counts[slot]++;
                if (!batch_can_use_pair[batch_index]) {
                    // 同一局面只要有一个请求要求精确刷新，整个去重槽都走完整网络。
                    unique_use_pair[slot] = 0;
                    unique_defer_refresh[slot] = 0;
                    unique_pair_parents[slot] = nullptr;
                    unique_pair_data[slot].reset();
                }
            }
            leaf_hash_keys[leaf_bucket] = leaf_identity;
            leaf_hash_values[leaf_bucket] = slot;
            eval_slot[batch_index] = slot;
        }

        for (int unique_index = 0; unique_index < unique_count; ++unique_index) {
            if (
                unique_use_pair[unique_index]
                && unique_request_counts[unique_index] >= pair_refresh_visits(
                    unique_keys[unique_index].current_player
                )
            ) {
                if (G_PAIR_DEFERRED_REFRESH) {
                    // 先完成当前模拟，节点创建后排队填入下一次完整网络的空槽。
                    unique_defer_refresh[unique_index] = 1;
                } else {
                    // 同步强度基线：同批多次请求的节点立即精确评估。
                    unique_use_pair[unique_index] = 0;
                    unique_pair_parents[unique_index] = nullptr;
                    unique_pair_data[unique_index].reset();
                    G_TOTAL_PAIR_EAGER_EXACT_EVALUATIONS++;
                }
            }
        }

        G_TOTAL_LEAF_REQUESTS += valid_count;
        G_TOTAL_BATCH_UNIQUE_POSITIONS += unique_count;

        int miss_count = 0;
        for (int unique_index = 0; unique_index < unique_count; ++unique_index) {
            float* policy_output = unique_policies.data() + unique_index * BOARD_CELLS;
            if (unique_use_pair[unique_index]) {
                synthesize_pair_evaluation(
                    *unique_pair_parents[unique_index],
                    unique_first_moves[unique_index],
                    policy_output,
                    unique_values[unique_index]
                );
                G_TOTAL_PAIR_PROVISIONAL_EVALUATIONS++;
                continue;
            }
            unique_pair_data[unique_index].reset();
            if (lookup_eval_cache(
                    unique_keys[unique_index],
                    policy_output,
                    unique_values[unique_index],
                    unique_pair_data[unique_index]
                )) {
                G_TOTAL_CACHE_HITS++;
                continue;
            }

            G_TOTAL_CACHE_MISSES++;
            miss_to_unique[miss_count] = unique_index;
            miss_count++;
        }

        int miss_start = 0;
        while (
            miss_start < miss_count
            || (
                G_PAIR_DEFERRED_REFRESH
                && static_cast<int>(refresh_queue.size()) >= G_BATCH_SIZE
            )
        ) {
            const int exact_batch = std::min(
                G_BATCH_SIZE,
                miss_count - miss_start
            );
            const int refresh_batch = G_PAIR_DEFERRED_REFRESH
                ? std::min<int>(
                    G_BATCH_SIZE - exact_batch,
                    refresh_queue.size()
                )
                : 0;
            const int callback_batch = exact_batch + refresh_batch;
            for (int callback_index = 0; callback_index < exact_batch; ++callback_index) {
                const int unique_index = miss_to_unique[miss_start + callback_index];
                std::copy_n(
                    unique_boards.data() + unique_index * BOARD_CELLS,
                    BOARD_CELLS,
                    callback_boards.data() + callback_index * BOARD_CELLS
                );
            }
            for (int refresh_index = 0; refresh_index < refresh_batch; ++refresh_index) {
                callback_refresh_jobs[refresh_index] = std::move(refresh_queue.front());
                refresh_queue.pop_front();
                const PairRefreshJob& job = callback_refresh_jobs[refresh_index];
                std::copy_n(
                    job.board.board,
                    BOARD_CELLS,
                    callback_boards.data()
                        + (exact_batch + refresh_index) * BOARD_CELLS
                );
            }
            if (G_PAIR_EVAL_CALLBACK != nullptr) {
                G_PAIR_EVAL_CALLBACK(
                    callback_batch,
                    callback_boards.data(),
                    callback_policies.data(),
                    callback_values.data(),
                    callback_candidate_factors.data(),
                    callback_first_factors.data(),
                    callback_pair_values.data()
                );
            } else {
                G_EVAL_CALLBACK(
                    callback_batch,
                    callback_boards.data(),
                    callback_policies.data(),
                    callback_values.data()
                );
            }
            G_TOTAL_NEURAL_EVALUATIONS += callback_batch;
            for (int callback_index = 0; callback_index < exact_batch; ++callback_index) {
                const int unique_index = miss_to_unique[miss_start + callback_index];
                if (unique_keys[unique_index].stones_in_turn == 1) {
                    G_TOTAL_SECOND_STONE_EVALUATIONS++;
                }
            }
            G_TOTAL_SECOND_STONE_EVALUATIONS += refresh_batch;
            G_TOTAL_EVAL_BATCHES++;

            for (int callback_index = 0; callback_index < exact_batch; ++callback_index) {
                const int unique_index = miss_to_unique[miss_start + callback_index];
                const float* source_policy = callback_policies.data() + callback_index * BOARD_CELLS;
                float* destination_policy = unique_policies.data() + unique_index * BOARD_CELLS;
                std::copy_n(source_policy, BOARD_CELLS, destination_policy);
                unique_values[unique_index] = callback_values[callback_index];
                std::shared_ptr<const PairData> pair_data;
                if (
                    G_PAIR_EVAL_CALLBACK != nullptr
                    && unique_keys[unique_index].stones_in_turn == 0
                ) {
                    auto mutable_pair = std::make_shared<PairData>();
                    std::copy_n(
                        callback_candidate_factors.data()
                            + callback_index * PAIR_FACTOR_COUNT,
                        PAIR_FACTOR_COUNT,
                        mutable_pair->candidate_factors.begin()
                    );
                    std::copy_n(
                        callback_first_factors.data()
                            + callback_index * PAIR_FACTOR_COUNT,
                        PAIR_FACTOR_COUNT,
                        mutable_pair->first_factors.begin()
                    );
                    std::copy_n(
                        callback_pair_values.data()
                            + callback_index * BOARD_CELLS,
                        BOARD_CELLS,
                        mutable_pair->values.begin()
                    );
                    pair_data = std::move(mutable_pair);
                }
                unique_pair_data[unique_index] = pair_data;
                store_eval_cache(
                    unique_keys[unique_index],
                    source_policy,
                    callback_values[callback_index],
                    pair_data
                );
            }
            for (int refresh_index = 0; refresh_index < refresh_batch; ++refresh_index) {
                const int callback_index = exact_batch + refresh_index;
                PairRefreshJob& job = callback_refresh_jobs[refresh_index];
                const float* source_policy = callback_policies.data()
                    + callback_index * BOARD_CELLS;
                if (job.node != nullptr) {
                    if (job.node->provisional_pair) {
                        refresh_node_priors(job.node, job.board, source_policy);
                        G_TOTAL_PAIR_EXACT_REFRESHES++;
                    }
                    job.node->pair_refresh_queued.store(
                        false,
                        std::memory_order_relaxed
                    );
                }
                store_eval_cache(
                    job.key,
                    source_policy,
                    callback_values[callback_index],
                    nullptr
                );
            }
            miss_start += exact_batch;
        }

        // 树结构扩展保持串行，避免同一个叶子在一个批次内被多次扩展；选择阶段
        // 仍由 OpenMP 并行。跨棋局批处理后这里通常不是主要瓶颈。
        for (int batch_index = 0; batch_index < active_batch; ++batch_index) {
            if (!batch_valid[batch_index]) {
                continue;
            }
            MCTSContext* context = batch_contexts[batch_index];
            MCTSNode* node = batch_leaf_nodes[batch_index];
            MCTSEdge* leaf_edge = batch_leaf_edges[batch_index];
            const int slot = eval_slot[batch_index];
            const Connect6Board& board = batch_boards[batch_index];
            const float raw_value = unique_values[slot];
            const float black_value = board.current_player == BLACK ? raw_value : -raw_value;

            if (leaf_edge != nullptr) {
                if (!leaf_edge->child) {
                    leaf_edge->child = std::make_unique<MCTSNode>(
                        leaf_edge->move,
                        board.current_player,
                        leaf_edge
                    );
                }
                node = leaf_edge->child.get();
            }
            if (node != nullptr && !node->expanded) {
                expand_node(node, board, unique_policies.data() + slot * BOARD_CELLS);
                node->provisional_pair = unique_use_pair[slot] != 0;
                if (!node->provisional_pair && board.stones_in_turn == 0) {
                    node->pair_data = unique_pair_data[slot];
                }
            } else if (
                node != nullptr
                && node->provisional_pair
                && !unique_use_pair[slot]
            ) {
                refresh_node_priors(
                    node,
                    board,
                    unique_policies.data() + slot * BOARD_CELLS
                );
                G_TOTAL_PAIR_EXACT_REFRESHES++;
            } else if (
                node != nullptr
                && !unique_use_pair[slot]
                && board.stones_in_turn == 0
                && unique_pair_data[slot]
            ) {
                node->pair_data = unique_pair_data[slot];
            }
            if (
                G_PAIR_DEFERRED_REFRESH
                && node != nullptr
                && node->provisional_pair
                && unique_defer_refresh[slot]
            ) {
                bool expected = false;
                if (node->pair_refresh_queued.compare_exchange_strong(
                        expected,
                        true,
                        std::memory_order_relaxed
                    )) {
                    PairRefreshJob job;
                    job.node = node;
                    job.board = board;
                    job.key = unique_keys[slot];
                    refresh_queue.push_back(std::move(job));
                    G_TOTAL_PAIR_EAGER_EXACT_EVALUATIONS++;
                }
            }
            backpropagate(
                *context->root,
                batch_paths.data() + batch_index * BOARD_CELLS,
                batch_path_lengths[batch_index],
                black_value
            );
        }
    }
    // 未凑满一个 batch 的后台刷新不单独触发 GPU；下次搜索再次访问时可重新入队。
    for (PairRefreshJob& job : refresh_queue) {
        if (job.node != nullptr) {
            job.node->pair_refresh_queued.store(false, std::memory_order_relaxed);
        }
    }
}

}  // namespace

extern "C" {

EXPORT void set_eval_callback(EvalCallback callback) {
    // 不同回调通常代表不同网络，旧网络结果绝不能泄漏到新网络。
    if (callback != G_EVAL_CALLBACK) {
        clear_eval_cache_impl();
    }
    G_EVAL_CALLBACK = callback;
}

EXPORT void set_pair_eval_callback(PairEvalCallback callback) {
    if (callback != G_PAIR_EVAL_CALLBACK) {
        clear_eval_cache_impl();
    }
    G_PAIR_EVAL_CALLBACK = callback;
}

EXPORT void set_pair_policy_params(
    float base_scale,
    const float* relative_bias,
    int relative_count,
    int refresh_visits
) {
    if (relative_bias == nullptr || relative_count <= 0) {
        G_PAIR_PARAMS_READY = false;
        return;
    }
    G_PAIR_BASE_SCALE = base_scale;
    G_PAIR_REFRESH_VISITS = std::max(refresh_visits, 1);
    G_PAIR_REFRESH_VISITS_BLACK = G_PAIR_REFRESH_VISITS;
    G_PAIR_REFRESH_VISITS_WHITE = G_PAIR_REFRESH_VISITS;
    constexpr int relative_size = BOARD_SIZE * 2 - 1;
    if (relative_count == relative_size * relative_size) {
        for (int first = 0; first < BOARD_CELLS; ++first) {
            const int first_row = first / BOARD_SIZE;
            const int first_column = first % BOARD_SIZE;
            for (int move = 0; move < BOARD_CELLS; ++move) {
                const int relative_row = move / BOARD_SIZE
                    - first_row + BOARD_SIZE - 1;
                const int relative_column = move % BOARD_SIZE
                    - first_column + BOARD_SIZE - 1;
                G_PAIR_RELATIVE_LOOKUP[first * BOARD_CELLS + move] = relative_bias[
                    relative_row * relative_size + relative_column
                ];
            }
        }
    } else if (relative_count == BOARD_CELLS * BOARD_CELLS) {
        std::copy_n(
            relative_bias,
            BOARD_CELLS * BOARD_CELLS,
            G_PAIR_RELATIVE_LOOKUP.begin()
        );
    } else {
        G_PAIR_PARAMS_READY = false;
        return;
    }
    G_PAIR_PARAMS_READY = true;
    clear_eval_cache_impl();
}

EXPORT void set_pair_refresh_mode(int deferred) {
    G_PAIR_DEFERRED_REFRESH = deferred != 0;
}

EXPORT void set_pair_refresh_visits_by_player(
    int black_refresh_visits,
    int white_refresh_visits
) {
    G_PAIR_REFRESH_VISITS_BLACK = std::max(black_refresh_visits, 1);
    G_PAIR_REFRESH_VISITS_WHITE = std::max(white_refresh_visits, 1);
}

EXPORT void set_pair_relative_gate(
    const float* gate,
    int value_count,
    int rank
) {
    const int expected = (BOARD_SIZE * 2 - 1)
        * (BOARD_SIZE * 2 - 1)
        * PAIR_RANK;
    if (gate == nullptr || value_count != expected || rank != PAIR_RANK) {
        G_PAIR_RELATIVE_GATE_READY = false;
        return;
    }
    std::copy_n(gate, expected, G_PAIR_RELATIVE_GATE.begin());
    G_PAIR_RELATIVE_GATE_READY = true;
    clear_eval_cache_impl();
}

EXPORT void set_pair_value_scale(float scale) {
    G_PAIR_VALUE_SCALE = std::max(0.0f, std::min(scale, 1.0f));
}

EXPORT void set_mcts_params(int batch_size, int num_threads) {
    if (batch_size > 0) {
        G_BATCH_SIZE = batch_size;
    }
    if (num_threads > 0) {
        G_NUM_THREADS = num_threads;
    }
    omp_set_num_threads(G_NUM_THREADS);
}

EXPORT void set_mcts_search_params(float cpuct, int widening_base, float widening_scale) {
    if (cpuct > 0.0f) {
        G_CPUCT = cpuct;
    }
    if (widening_base > 0) {
        G_PROGRESSIVE_WIDENING_BASE = widening_base;
    }
    if (widening_scale >= 0.0f) {
        G_PROGRESSIVE_WIDENING_SCALE = widening_scale;
    }
}

EXPORT void set_mcts_selection_mode(int deterministic) {
    G_DETERMINISTIC_TREE_SELECTION = deterministic != 0;
}

EXPORT void set_eval_cache_capacity(long long capacity) {
    G_EVAL_CACHE_CAPACITY = capacity > 0 ? static_cast<std::size_t>(capacity) : 0;
    clear_eval_cache_impl();
    if (G_EVAL_CACHE_CAPACITY > 0) {
        G_EVAL_CACHE.reserve(G_EVAL_CACHE_CAPACITY);
    }
}

EXPORT void clear_eval_cache() {
    clear_eval_cache_impl();
}

EXPORT long long get_eval_cache_size() {
    return static_cast<long long>(G_EVAL_CACHE.size());
}

EXPORT void reset_mcts_statistics() {
    reset_statistics_impl();
}

EXPORT long long get_total_leaf_requests() {
    return G_TOTAL_LEAF_REQUESTS;
}

EXPORT long long get_total_search_batches() {
    return G_TOTAL_SEARCH_BATCHES;
}

EXPORT long long get_total_batch_unique_positions() {
    return G_TOTAL_BATCH_UNIQUE_POSITIONS;
}

EXPORT long long get_total_unique_evaluations() {
    return G_TOTAL_NEURAL_EVALUATIONS;
}

EXPORT long long get_total_second_stone_evaluations() {
    return G_TOTAL_SECOND_STONE_EVALUATIONS;
}

EXPORT long long get_total_eval_batches() {
    return G_TOTAL_EVAL_BATCHES;
}

EXPORT long long get_total_cache_hits() {
    return G_TOTAL_CACHE_HITS;
}

EXPORT long long get_total_cache_misses() {
    return G_TOTAL_CACHE_MISSES;
}

EXPORT long long get_total_pair_provisional_evaluations() {
    return G_TOTAL_PAIR_PROVISIONAL_EVALUATIONS;
}

EXPORT long long get_total_pair_exact_refreshes() {
    return G_TOTAL_PAIR_EXACT_REFRESHES;
}

EXPORT long long get_total_pair_eager_exact_evaluations() {
    return G_TOTAL_PAIR_EAGER_EXACT_EVALUATIONS;
}

// 旧单棋局 API 保持不变，现有训练循环可以继续使用。
EXPORT void init_game() {
    G_DEFAULT_CONTEXT.reset();
    reset_statistics_impl();
}

EXPORT void set_random_seed(int seed) {
    G_DEFAULT_CONTEXT.rng.seed(seed);
}

EXPORT void play_move(int move) {
    play_move_impl(G_DEFAULT_CONTEXT, move);
}

EXPORT void run_mcts_simulations(int simulations) {
    run_simulations_impl({&G_DEFAULT_CONTEXT}, {simulations});
}

EXPORT int get_best_move(float temperature) {
    return best_move_impl(G_DEFAULT_CONTEXT, temperature);
}

EXPORT float get_root_value() {
    return G_DEFAULT_CONTEXT.root ? G_DEFAULT_CONTEXT.root->root_q_value() : 0.0f;
}

EXPORT void get_policy(float* output) {
    policy_impl(G_DEFAULT_CONTEXT, output);
}

EXPORT int get_root_action_count() {
    return G_DEFAULT_CONTEXT.root ? static_cast<int>(G_DEFAULT_CONTEXT.root->children.size()) : 0;
}

EXPORT int get_root_active_action_count() {
    return G_DEFAULT_CONTEXT.root ? active_child_count(*G_DEFAULT_CONTEXT.root) : 0;
}

EXPORT void print_top_moves() {
    print_top_moves_impl(G_DEFAULT_CONTEXT);
}

// 新 API：多个棋局共享同一个 Python/TensorRT 回调和 GPU 批次。
EXPORT void* create_mcts_context(int seed) {
    return new MCTSContext(seed);
}

EXPORT void destroy_mcts_context(void* handle) {
    delete static_cast<MCTSContext*>(handle);
}

EXPORT void reset_mcts_context(void* handle) {
    if (handle != nullptr) {
        static_cast<MCTSContext*>(handle)->reset();
    }
}

EXPORT void set_mcts_context_seed(void* handle, int seed) {
    if (handle != nullptr) {
        static_cast<MCTSContext*>(handle)->rng.seed(seed);
    }
}

EXPORT void play_move_context(void* handle, int move) {
    if (handle != nullptr) {
        play_move_impl(*static_cast<MCTSContext*>(handle), move);
    }
}

EXPORT void run_mcts_simulations_context(void* handle, int simulations) {
    if (handle != nullptr) {
        run_simulations_impl({static_cast<MCTSContext*>(handle)}, {simulations});
    }
}

EXPORT void run_mcts_simulations_multi(
    void* const* handles,
    const int* simulations,
    int context_count
) {
    if (handles == nullptr || simulations == nullptr || context_count <= 0) {
        return;
    }
    std::vector<MCTSContext*> contexts;
    std::vector<int> budgets;
    contexts.reserve(context_count);
    budgets.reserve(context_count);
    for (int index = 0; index < context_count; ++index) {
        contexts.push_back(static_cast<MCTSContext*>(handles[index]));
        budgets.push_back(simulations[index]);
    }
    run_simulations_impl(contexts, budgets);
}

EXPORT int get_best_move_context(void* handle, float temperature) {
    return handle != nullptr ? best_move_impl(*static_cast<MCTSContext*>(handle), temperature) : -1;
}

EXPORT float get_root_value_context(void* handle) {
    if (handle == nullptr) {
        return 0.0f;
    }
    const MCTSContext* context = static_cast<MCTSContext*>(handle);
    return context->root ? context->root->root_q_value() : 0.0f;
}

EXPORT void get_policy_context(void* handle, float* output) {
    if (handle == nullptr) {
        std::fill(output, output + BOARD_CELLS, 0.0f);
        return;
    }
    policy_impl(*static_cast<MCTSContext*>(handle), output);
}

EXPORT int get_root_action_count_context(void* handle) {
    if (handle == nullptr) {
        return 0;
    }
    const MCTSContext* context = static_cast<MCTSContext*>(handle);
    return context->root ? static_cast<int>(context->root->children.size()) : 0;
}

EXPORT int get_root_active_action_count_context(void* handle) {
    if (handle == nullptr) {
        return 0;
    }
    const MCTSContext* context = static_cast<MCTSContext*>(handle);
    return context->root ? active_child_count(*context->root) : 0;
}

EXPORT void print_top_moves_context(void* handle) {
    if (handle != nullptr) {
        print_top_moves_impl(*static_cast<MCTSContext*>(handle));
    }
}

EXPORT void get_second_stone_visit_counts_context(
    void* handle,
    const int* thresholds,
    int threshold_count,
    long long* output
) {
    if (output == nullptr || threshold_count <= 0) {
        return;
    }
    std::fill(output, output + threshold_count, 0LL);
    if (handle == nullptr || thresholds == nullptr) {
        return;
    }
    const MCTSContext* context = static_cast<MCTSContext*>(handle);
    if (!context->root) {
        return;
    }
    accumulate_second_stone_visits(
        *context->root,
        context->board.total_stones,
        thresholds,
        threshold_count,
        output
    );
}

}  // extern "C"
