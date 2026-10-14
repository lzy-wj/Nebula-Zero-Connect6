#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int BOARD_SIZE = 19;
constexpr int BOARD_POINTS = BOARD_SIZE * BOARD_SIZE;
constexpr int BLACK = 1;
constexpr int WHITE = -1;
constexpr int PASS_MOVE = -1;

struct Board {
    std::array<int8_t, BOARD_POINTS> cells{};
    std::array<uint64_t, (BOARD_POINTS + 63) / 64> black_bits{};
    std::array<uint64_t, (BOARD_POINTS + 63) / 64> white_bits{};
    int8_t current_player = BLACK;
    int16_t total_stones = 0;
    bool terminal = false;
    int8_t winner = 0;

    void place(int move, int player) {
        cells[move] = static_cast<int8_t>(player);
        auto& bits = player == BLACK ? black_bits : white_bits;
        bits[move / 64] |= uint64_t{1} << (move % 64);
    }

    void clear(int move, int player) {
        cells[move] = 0;
        auto& bits = player == BLACK ? black_bits : white_bits;
        bits[move / 64] &= ~(uint64_t{1} << (move % 64));
    }

    bool check_win(int move, int player) const {
        const int row = move / BOARD_SIZE;
        const int column = move % BOARD_SIZE;
        constexpr int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
        for (const auto& direction : directions) {
            int count = 1;
            for (int sign : {-1, 1}) {
                int test_row = row + sign * direction[0];
                int test_column = column + sign * direction[1];
                while (
                    test_row >= 0 && test_row < BOARD_SIZE
                    && test_column >= 0 && test_column < BOARD_SIZE
                    && cells[test_row * BOARD_SIZE + test_column] == player
                ) {
                    ++count;
                    test_row += sign * direction[0];
                    test_column += sign * direction[1];
                }
            }
            if (count >= 6) {
                return true;
            }
        }
        return false;
    }

    bool immediate_win(int move, int player) {
        if (move < 0 || move >= BOARD_POINTS || cells[move] != 0) {
            return false;
        }
        place(move, player);
        const bool won = check_win(move, player);
        clear(move, player);
        return won;
    }

    bool apply_action(int first, int second) {
        if (terminal || first < 0 || first >= BOARD_POINTS || cells[first] != 0) {
            return false;
        }
        const bool opening = total_stones == 0;
        if (opening && second != PASS_MOVE) {
            return false;
        }
        if (!opening && second != PASS_MOVE) {
            if (second <= first || second >= BOARD_POINTS || cells[second] != 0) {
                return false;
            }
        }
        const int actor = current_player;
        place(first, actor);
        ++total_stones;
        if (check_win(first, actor)) {
            terminal = true;
            winner = static_cast<int8_t>(actor);
            current_player = static_cast<int8_t>(-actor);
            return true;
        }
        if (total_stones == BOARD_POINTS) {
            terminal = true;
            winner = 0;
            current_player = static_cast<int8_t>(-actor);
            return true;
        }
        if (second == PASS_MOVE) {
            if (!opening) {
                clear(first, actor);
                --total_stones;
                return false;
            }
            current_player = static_cast<int8_t>(-actor);
            return true;
        }
        place(second, actor);
        ++total_stones;
        if (check_win(second, actor)) {
            terminal = true;
            winner = static_cast<int8_t>(actor);
        } else if (total_stones == BOARD_POINTS) {
            terminal = true;
            winner = 0;
        }
        current_player = static_cast<int8_t>(-actor);
        return true;
    }
};

struct Node;

struct Edge {
    int16_t first = 0;
    int16_t second = 0;
    float prior = 0.0F;
    int visits = 0;
    double value_sum = 0.0;
    std::unique_ptr<Node> child;

    Edge(int first_move, int second_move, float action_prior)
        : first(static_cast<int16_t>(first_move)),
          second(static_cast<int16_t>(second_move)),
          prior(action_prior) {}

    Edge(Edge&&) noexcept = default;
    Edge& operator=(Edge&&) noexcept = default;
    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;
};

struct CandidateAction {
    int16_t first = 0;
    int16_t second = 0;
    float prior = 0.0F;
};

struct Node {
    bool expanded = false;
    int total_visits = 0;
    std::vector<Edge> edges;
    std::vector<CandidateAction> candidates;
    std::size_t next_candidate = 0;
};

struct Context {
    Board board;
    std::unique_ptr<Node> root = std::make_unique<Node>();
    std::mt19937 rng;
    long long last_simulations = 0;
    long long last_neural_evaluations = 0;
    long long last_unique_nodes = 0;

    explicit Context(uint32_t seed) : rng(seed) {}
};

using EvalCallback = int (*)(
    int,
    const int8_t*,
    const int8_t*,
    int16_t*,
    int16_t*,
    float*,
    float*
);

EvalCallback G_EVAL_CALLBACK = nullptr;
// The Python evaluator (and its CUDA stream/state) is shared by both lane
// callback threads.  Keep the CPU preparation overlap, but serialize callback
// entry so two lanes never concurrently touch Torch/Python state.
std::mutex G_EVAL_CALLBACK_MUTEX;
int G_TOP_K = 64;
int G_RANDOM_CANDIDATES = 0;
int G_CPU_THREADS = 1;
int G_PROGRESSIVE_WIDENING_BASE = 0;
float G_PROGRESSIVE_WIDENING_SCALE = 0.0F;
float G_CPUCT = 1.5F;
float G_ROOT_NOISE_EPSILON = 0.25F;
float G_ROOT_DIRICHLET_ALPHA = 0.15F;
long long G_TOTAL_SIMULATIONS = 0;
long long G_TOTAL_EVALUATIONS = 0;
long long G_TOTAL_EVAL_BATCHES = 0;
long long G_TOTAL_TACTICAL_EXPANSIONS = 0;

struct LeafJob {
    Context* context = nullptr;
    int context_index = -1;
    Node* node = nullptr;
    Board board;
    struct PathStep {
        Node* node = nullptr;
        Edge* edge = nullptr;
    };
    std::array<PathStep, BOARD_POINTS> path{};
    int path_size = 0;
    std::vector<std::pair<int, int>> opponent_threats;
};

struct WinningActions {
    std::vector<int> singles;
    std::vector<std::pair<int, int>> pairs;
};

struct TacticalActions {
    WinningActions current;
    WinningActions opponent;
};

struct WinningWindow {
    std::array<int16_t, 6> points{};
    std::array<uint64_t, (BOARD_POINTS + 63) / 64> mask{};
};

const std::vector<WinningWindow>& winning_windows() {
    static const std::vector<WinningWindow> windows = [] {
        std::vector<WinningWindow> result;
        result.reserve(924);
        constexpr int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
        for (const auto& direction : directions) {
            for (int row = 0; row < BOARD_SIZE; ++row) {
                for (int column = 0; column < BOARD_SIZE; ++column) {
                    const int end_row = row + 5 * direction[0];
                    const int end_column = column + 5 * direction[1];
                    if (
                        end_row < 0 || end_row >= BOARD_SIZE
                        || end_column < 0 || end_column >= BOARD_SIZE
                    ) {
                        continue;
                    }
                    WinningWindow window;
                    for (int offset = 0; offset < 6; ++offset) {
                        const int point =
                            (row + offset * direction[0]) * BOARD_SIZE
                            + column + offset * direction[1];
                        window.points[offset] = static_cast<int16_t>(point);
                        window.mask[point / 64] |= uint64_t{1} << (point % 64);
                    }
                    result.push_back(window);
                }
            }
        }
        return result;
    }();
    return windows;
}

TacticalActions tactical_actions(const Board& board) {
    TacticalActions actions;
    std::array<bool, BOARD_POINTS> current_singles{};
    std::array<bool, BOARD_POINTS> opponent_singles{};
    std::unordered_set<int> current_pairs;
    std::unordered_set<int> opponent_pairs;
    const int current = board.current_player;

    const auto& current_bits = current == BLACK ? board.black_bits : board.white_bits;
    const auto& opponent_bits = current == BLACK ? board.white_bits : board.black_bits;
    for (const WinningWindow& window : winning_windows()) {
        int current_count = 0;
        int opponent_count = 0;
        for (std::size_t word = 0; word < window.mask.size(); ++word) {
            current_count += __builtin_popcountll(current_bits[word] & window.mask[word]);
            opponent_count += __builtin_popcountll(opponent_bits[word] & window.mask[word]);
        }
        const int empty_count = 6 - current_count - opponent_count;
        int empties[2] = {PASS_MOVE, PASS_MOVE};
        if (empty_count <= 2) {
            int empty_index = 0;
            for (int point : window.points) {
                if (board.cells[point] == 0) {
                    empties[empty_index++] = point;
                }
            }
        }

        if (opponent_count == 0 && current_count == 5 && empty_count == 1) {
            current_singles[empties[0]] = true;
        } else if (
            opponent_count == 0 && current_count == 4 && empty_count == 2
        ) {
            const int first = std::min(empties[0], empties[1]);
            const int second = std::max(empties[0], empties[1]);
            const int key = first * BOARD_POINTS + second;
            if (current_pairs.insert(key).second) {
                actions.current.pairs.emplace_back(first, second);
            }
        }

        if (current_count == 0 && opponent_count == 5 && empty_count == 1) {
            opponent_singles[empties[0]] = true;
        } else if (
            current_count == 0 && opponent_count == 4 && empty_count == 2
        ) {
            const int first = std::min(empties[0], empties[1]);
            const int second = std::max(empties[0], empties[1]);
            const int key = first * BOARD_POINTS + second;
            if (opponent_pairs.insert(key).second) {
                actions.opponent.pairs.emplace_back(first, second);
            }
        }
    }

    for (int move = 0; move < BOARD_POINTS; ++move) {
        if (current_singles[move]) {
            actions.current.singles.push_back(move);
        }
        if (opponent_singles[move]) {
            actions.opponent.singles.push_back(move);
        }
    }
    return actions;
}

bool blocks_all_threats(
    int first,
    int second,
    const std::vector<std::pair<int, int>>& threats
) {
    for (const auto& threat : threats) {
        if (
            first != threat.first && first != threat.second
            && second != threat.first && second != threat.second
        ) {
            return false;
        }
    }
    return true;
}

void normalize_priors(Node& node) {
    double total = 0.0;
    for (const Edge& edge : node.edges) {
        if (std::isfinite(edge.prior) && edge.prior > 0.0F) {
            total += edge.prior;
        }
    }
    if (total <= 0.0) {
        const float uniform = node.edges.empty()
            ? 0.0F
            : 1.0F / static_cast<float>(node.edges.size());
        for (Edge& edge : node.edges) {
            edge.prior = uniform;
        }
        return;
    }
    for (Edge& edge : node.edges) {
        edge.prior = static_cast<float>(
            (std::isfinite(edge.prior) && edge.prior > 0.0F ? edge.prior : 0.0F)
            / total
        );
    }
}

void apply_root_noise(Node& node, std::mt19937& rng) {
    if (
        node.edges.size() < 2
        || G_ROOT_NOISE_EPSILON <= 0.0F
        || G_ROOT_DIRICHLET_ALPHA <= 0.0F
    ) {
        return;
    }
    std::gamma_distribution<float> gamma(G_ROOT_DIRICHLET_ALPHA, 1.0F);
    std::vector<float> noise(node.edges.size());
    double total = 0.0;
    for (float& value : noise) {
        value = gamma(rng);
        total += value;
    }
    if (total <= 0.0) {
        return;
    }
    for (std::size_t index = 0; index < node.edges.size(); ++index) {
        const float normalized = static_cast<float>(noise[index] / total);
        node.edges[index].prior =
            (1.0F - G_ROOT_NOISE_EPSILON) * node.edges[index].prior
            + G_ROOT_NOISE_EPSILON * normalized;
    }
    normalize_priors(node);
}

void normalize_candidate_priors(std::vector<CandidateAction>& candidates) {
    double total = 0.0;
    for (const CandidateAction& action : candidates) {
        if (std::isfinite(action.prior) && action.prior > 0.0F) {
            total += action.prior;
        }
    }
    const float uniform = candidates.empty()
        ? 0.0F
        : 1.0F / static_cast<float>(candidates.size());
    for (CandidateAction& action : candidates) {
        action.prior = total > 0.0
            ? static_cast<float>(
                (std::isfinite(action.prior) && action.prior > 0.0F
                    ? action.prior
                    : 0.0F) / total
            )
            : uniform;
    }
}

void apply_root_noise(
    std::vector<CandidateAction>& candidates,
    std::mt19937& rng
) {
    if (
        candidates.size() < 2
        || G_ROOT_NOISE_EPSILON <= 0.0F
        || G_ROOT_DIRICHLET_ALPHA <= 0.0F
    ) {
        return;
    }
    std::gamma_distribution<float> gamma(G_ROOT_DIRICHLET_ALPHA, 1.0F);
    std::vector<float> noise(candidates.size());
    double total = 0.0;
    for (float& value : noise) {
        value = gamma(rng);
        total += value;
    }
    if (total <= 0.0) {
        return;
    }
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        candidates[index].prior =
            (1.0F - G_ROOT_NOISE_EPSILON) * candidates[index].prior
            + G_ROOT_NOISE_EPSILON * static_cast<float>(noise[index] / total);
    }
    normalize_candidate_priors(candidates);
}

void expand_actions(
    LeafJob& job,
    const int16_t* first_moves,
    const int16_t* second_moves,
    const float* priors
) {
    Node& node = *job.node;
    node.expanded = true;
    node.edges.clear();
    node.candidates.clear();
    node.next_candidate = 0;
    std::vector<CandidateAction> candidates;
    candidates.reserve(static_cast<std::size_t>(G_TOP_K));
    std::unordered_set<int> seen;
    const bool opening = job.board.total_stones == 0;
    const int random_candidates = std::min(G_RANDOM_CANDIDATES, G_TOP_K);
    const int network_candidates = G_TOP_K - random_candidates;

    for (int index = 0; index < network_candidates; ++index) {
        int first = first_moves[index];
        int second = second_moves[index];
        if (opening) {
            second = PASS_MOVE;
        } else if (first > second) {
            std::swap(first, second);
        }
        if (
            first < 0 || first >= BOARD_POINTS
            || job.board.cells[first] != 0
        ) {
            continue;
        }
        if (opening) {
            if (second != PASS_MOVE) {
                continue;
            }
        } else if (
            second <= first || second >= BOARD_POINTS
            || job.board.cells[second] != 0
        ) {
            continue;
        }
        if (!blocks_all_threats(first, second, job.opponent_threats)) {
            continue;
        }
        const int key = first * (BOARD_POINTS + 1) + (second + 1);
        if (!seen.insert(key).second) {
            continue;
        }
        candidates.push_back({
            static_cast<int16_t>(first),
            static_cast<int16_t>(second),
            priors[index],
        });
    }

    if (random_candidates > 0) {
        std::vector<int> empty;
        empty.reserve(BOARD_POINTS - job.board.total_stones);
        for (int move = 0; move < BOARD_POINTS; ++move) {
            if (job.board.cells[move] == 0) {
                empty.push_back(move);
            }
        }
        int added = 0;
        int attempts = 0;
        if (!empty.empty()) {
            std::uniform_int_distribution<std::size_t> distribution(
                0,
                empty.size() - 1
            );
            while (added < random_candidates && attempts < random_candidates * 64) {
                ++attempts;
                int first = empty[distribution(job.context->rng)];
                int second = PASS_MOVE;
                if (!opening) {
                    if (empty.size() < 2) {
                        break;
                    }
                    do {
                        second = empty[distribution(job.context->rng)];
                    } while (second == first);
                    if (first > second) {
                        std::swap(first, second);
                    }
                }
                if (!blocks_all_threats(first, second, job.opponent_threats)) {
                    continue;
                }
                const int key = first * (BOARD_POINTS + 1) + (second + 1);
                if (!seen.insert(key).second) {
                    continue;
                }
                candidates.push_back({
                    static_cast<int16_t>(first),
                    static_cast<int16_t>(second),
                    1.0F / G_TOP_K,
                });
                ++added;
            }
        }
    }

    if (candidates.empty() && !opening && !job.opponent_threats.empty()) {
        for (int first = 0; first < BOARD_POINTS; ++first) {
            if (job.board.cells[first] != 0) {
                continue;
            }
            for (int second = first + 1; second < BOARD_POINTS; ++second) {
                if (job.board.cells[second] != 0) {
                    continue;
                }
                if (blocks_all_threats(first, second, job.opponent_threats)) {
                    candidates.push_back({
                        static_cast<int16_t>(first),
                        static_cast<int16_t>(second),
                        1.0F,
                    });
                    if (static_cast<int>(candidates.size()) >= G_TOP_K) {
                        break;
                    }
                }
            }
            if (static_cast<int>(candidates.size()) >= G_TOP_K) {
                break;
            }
        }
    }
    if (candidates.empty()) {
        int first = PASS_MOVE;
        int second = PASS_MOVE;
        for (int move = 0; move < BOARD_POINTS; ++move) {
            if (job.board.cells[move] == 0) {
                if (first == PASS_MOVE) {
                    first = move;
                } else {
                    second = move;
                    break;
                }
            }
        }
        if (first != PASS_MOVE) {
            candidates.push_back({
                static_cast<int16_t>(first),
                static_cast<int16_t>(opening ? PASS_MOVE : second),
                1.0F,
            });
        }
    }
    normalize_candidate_priors(candidates);
    if (G_PROGRESSIVE_WIDENING_BASE > 0) {
        const bool root = job.node == job.context->root.get();
        if (root) {
            apply_root_noise(candidates, job.context->rng);
        }
        if (root || random_candidates > 0) {
            std::stable_sort(
                candidates.begin(),
                candidates.end(),
                [](const CandidateAction& left, const CandidateAction& right) {
                    return left.prior > right.prior;
                }
            );
        }
        node.candidates = std::move(candidates);
        const std::size_t initial = std::min(
            node.candidates.size(),
            static_cast<std::size_t>(G_PROGRESSIVE_WIDENING_BASE)
        );
        node.edges.reserve(initial);
        while (node.next_candidate < initial) {
            const CandidateAction& action = node.candidates[node.next_candidate++];
            node.edges.emplace_back(action.first, action.second, action.prior);
        }
    } else {
        node.edges.reserve(candidates.size());
        for (const CandidateAction& action : candidates) {
            node.edges.emplace_back(action.first, action.second, action.prior);
        }
        if (job.node == job.context->root.get()) {
            apply_root_noise(node, job.context->rng);
        }
    }
}

void widen(Node& node) {
    if (
        G_PROGRESSIVE_WIDENING_BASE <= 0
        || node.next_candidate >= node.candidates.size()
    ) {
        return;
    }
    const int widened = G_PROGRESSIVE_WIDENING_BASE + static_cast<int>(
        G_PROGRESSIVE_WIDENING_SCALE
        * std::sqrt(static_cast<float>(node.total_visits))
    );
    const std::size_t target = std::min(
        node.candidates.size(),
        static_cast<std::size_t>(std::max(widened, 1))
    );
    while (node.next_candidate < target) {
        const CandidateAction& action = node.candidates[node.next_candidate++];
        node.edges.emplace_back(action.first, action.second, action.prior);
    }
}

Edge* select_edge(Node& node) {
    widen(node);
    const float exploration_scale = std::sqrt(
        static_cast<float>(node.total_visits + 1)
    );
    Edge* best = nullptr;
    float best_score = -1.0e30F;
    for (Edge& edge : node.edges) {
        const float quality = edge.visits > 0
            ? static_cast<float>(edge.value_sum / edge.visits)
            : 0.0F;
        const float exploration = G_CPUCT * edge.prior * exploration_scale
            / static_cast<float>(edge.visits + 1);
        const float score = quality + exploration;
        if (best == nullptr || score > best_score) {
            best = &edge;
            best_score = score;
        }
    }
    return best;
}

void backup(const LeafJob& job, float leaf_value) {
    float value = leaf_value;
    for (int index = job.path_size - 1; index >= 0; --index) {
        value = -value;
        const LeafJob::PathStep& step = job.path[index];
        Edge* edge = step.edge;
        ++edge->visits;
        edge->value_sum += value;
        ++step.node->total_visits;
    }
}

bool expand_tactical(LeafJob& job, float& value) {
    TacticalActions actions = tactical_actions(job.board);
    WinningActions& wins = actions.current;
    if (!wins.singles.empty()) {
        job.node->expanded = true;
        job.node->edges.clear();
        job.node->candidates.clear();
        job.node->next_candidate = 0;
        job.node->edges.reserve(wins.singles.size());
        const float prior = 1.0F / static_cast<float>(wins.singles.size());
        for (int move : wins.singles) {
            job.node->edges.emplace_back(move, PASS_MOVE, prior);
        }
        if (job.node == job.context->root.get()) {
            apply_root_noise(*job.node, job.context->rng);
        }
        value = 1.0F;
        return true;
    }
    if (!wins.pairs.empty()) {
        job.node->expanded = true;
        job.node->edges.clear();
        job.node->candidates.clear();
        job.node->next_candidate = 0;
        job.node->edges.reserve(wins.pairs.size());
        const float prior = 1.0F / static_cast<float>(wins.pairs.size());
        for (const auto& action : wins.pairs) {
            job.node->edges.emplace_back(action.first, action.second, prior);
        }
        if (job.node == job.context->root.get()) {
            apply_root_noise(*job.node, job.context->rng);
        }
        value = 1.0F;
        return true;
    }
    const WinningActions& threats = actions.opponent;
    for (int threat : threats.singles) {
        job.opponent_threats.emplace_back(threat, threat);
    }
    job.opponent_threats.insert(
        job.opponent_threats.end(),
        threats.pairs.begin(),
        threats.pairs.end()
    );
    return false;
}

int run_multi_impl(Context** contexts, const int* budgets, int context_count) {
    if (G_EVAL_CALLBACK == nullptr || context_count <= 0) {
        return 1;
    }
    std::vector<int> remaining(context_count);
    std::unordered_set<Context*> unique_contexts;
    long long total_remaining = 0;
    for (int index = 0; index < context_count; ++index) {
        if (contexts[index] == nullptr || budgets[index] < 0) {
            return 1;
        }
        if (!unique_contexts.insert(contexts[index]).second) {
            return 1;
        }
        remaining[index] = budgets[index];
        total_remaining += budgets[index];
        contexts[index]->last_simulations = 0;
        contexts[index]->last_neural_evaluations = 0;
        contexts[index]->last_unique_nodes = 0;
    }

    std::vector<LeafJob> jobs;
    jobs.reserve(context_count);
    std::vector<LeafJob*> network_jobs;
    network_jobs.reserve(context_count);
    std::vector<float> tactical_values;
    tactical_values.reserve(context_count);
    std::vector<uint8_t> tactical_flags;
    tactical_flags.reserve(context_count);
    std::vector<int8_t> callback_boards;
    std::vector<int8_t> callback_players;
    std::vector<int16_t> callback_first;
    std::vector<int16_t> callback_second;
    std::vector<float> callback_priors;
    std::vector<float> callback_values;

    while (total_remaining > 0) {
        jobs.clear();
        network_jobs.clear();
        for (int context_index = 0; context_index < context_count; ++context_index) {
            if (remaining[context_index] <= 0) {
                continue;
            }
            Context& context = *contexts[context_index];
            if (context.board.terminal) {
                total_remaining -= remaining[context_index];
                remaining[context_index] = 0;
                continue;
            }
            jobs.emplace_back();
            LeafJob& job = jobs.back();
            job.context = &context;
            job.context_index = context_index;
            job.node = context.root.get();
            job.board = context.board;
            bool completed = false;
            while (job.node->expanded && !job.node->edges.empty()) {
                Edge* edge = select_edge(*job.node);
                if (edge == nullptr || !job.board.apply_action(edge->first, edge->second)) {
                    return 1;
                }
                if (job.path_size >= BOARD_POINTS) {
                    return 1;
                }
                job.path[job.path_size++] = {job.node, edge};
                if (edge->child == nullptr) {
                    edge->child = std::make_unique<Node>();
                }
                job.node = edge->child.get();
                if (job.board.terminal) {
                    const float terminal_value = job.board.winner == 0 ? 0.0F : -1.0F;
                    backup(job, terminal_value);
                    --remaining[context_index];
                    --total_remaining;
                    ++context.last_simulations;
                    ++G_TOTAL_SIMULATIONS;
                    completed = true;
                    break;
                }
            }
            if (completed) {
                jobs.pop_back();
            }
        }

        tactical_values.assign(jobs.size(), 0.0F);
        tactical_flags.assign(jobs.size(), 0);
#pragma omp parallel for if(G_CPU_THREADS > 1) num_threads(G_CPU_THREADS) schedule(static)
        for (int index = 0; index < static_cast<int>(jobs.size()); ++index) {
            tactical_flags[index] = static_cast<uint8_t>(
                expand_tactical(jobs[index], tactical_values[index])
            );
        }
        for (std::size_t index = 0; index < jobs.size(); ++index) {
            LeafJob& job = jobs[index];
            if (tactical_flags[index]) {
                backup(job, tactical_values[index]);
                --remaining[job.context_index];
                --total_remaining;
                ++job.context->last_simulations;
                ++job.context->last_unique_nodes;
                ++G_TOTAL_SIMULATIONS;
                ++G_TOTAL_TACTICAL_EXPANSIONS;
            } else {
                network_jobs.push_back(&job);
            }
        }

        if (network_jobs.empty()) {
            continue;
        }
        const int batch = static_cast<int>(network_jobs.size());
        callback_boards.resize(static_cast<std::size_t>(batch) * BOARD_POINTS);
        callback_players.resize(batch);
        callback_first.resize(static_cast<std::size_t>(batch) * G_TOP_K);
        callback_second.resize(static_cast<std::size_t>(batch) * G_TOP_K);
        callback_priors.resize(static_cast<std::size_t>(batch) * G_TOP_K);
        callback_values.resize(batch);
        for (int index = 0; index < batch; ++index) {
            const LeafJob& job = *network_jobs[index];
            std::copy(
                job.board.cells.begin(),
                job.board.cells.end(),
                callback_boards.begin() + static_cast<std::size_t>(index) * BOARD_POINTS
            );
            callback_players[index] = job.board.current_player;
        }
        const int callback_status = G_EVAL_CALLBACK(
            batch,
            callback_boards.data(),
            callback_players.data(),
            callback_first.data(),
            callback_second.data(),
            callback_priors.data(),
            callback_values.data()
        );
        if (callback_status != 0) {
            return callback_status;
        }
        ++G_TOTAL_EVAL_BATCHES;
        G_TOTAL_EVALUATIONS += batch;
        for (int index = 0; index < batch; ++index) {
            LeafJob& job = *network_jobs[index];
            expand_actions(
                job,
                callback_first.data() + static_cast<std::size_t>(index) * G_TOP_K,
                callback_second.data() + static_cast<std::size_t>(index) * G_TOP_K,
                callback_priors.data() + static_cast<std::size_t>(index) * G_TOP_K
            );
            const float value = std::clamp(callback_values[index], -1.0F, 1.0F);
            backup(job, value);
            ++job.context->last_simulations;
            ++job.context->last_neural_evaluations;
            ++job.context->last_unique_nodes;
            ++G_TOTAL_SIMULATIONS;
            --remaining[job.context_index];
            --total_remaining;
        }
    }
    return 0;
}

// The synchronous path above is retained as the reference implementation.
// This lane-pipelined variant overlaps one GPU callback with CPU tree/leaf
// preparation from the other half of the contexts.  Both lane threads may be
// alive, but callback entry is serialized by G_EVAL_CALLBACK_MUTEX so the
// Python/Torch evaluator remains serialized and deterministic.
struct AsyncBatch {
    std::vector<LeafJob> jobs;
    std::vector<LeafJob*> network_jobs;
    std::vector<int8_t> boards;
    std::vector<int8_t> players;
    std::vector<int16_t> first;
    std::vector<int16_t> second;
    std::vector<float> priors;
    std::vector<float> values;
    std::thread callback_thread;
    int callback_status = 0;
    bool launched = false;
};

bool prepare_async_lane(
    Context** contexts,
    const std::vector<int>& lane_indices,
    std::vector<int>& remaining,
    long long& total_remaining,
    AsyncBatch& batch
) {
    batch.jobs.clear();
    batch.network_jobs.clear();
    batch.launched = false;
    batch.callback_status = 0;
    batch.jobs.reserve(lane_indices.size());
    batch.network_jobs.reserve(lane_indices.size());
    for (int context_index : lane_indices) {
        if (remaining[context_index] <= 0) {
            continue;
        }
        Context& context = *contexts[context_index];
        if (context.board.terminal) {
            total_remaining -= remaining[context_index];
            remaining[context_index] = 0;
            continue;
        }
        batch.jobs.emplace_back();
        LeafJob& job = batch.jobs.back();
        job.context = &context;
        job.context_index = context_index;
        job.node = context.root.get();
        job.board = context.board;
        bool completed = false;
        while (job.node->expanded && !job.node->edges.empty()) {
            Edge* edge = select_edge(*job.node);
            if (edge == nullptr || !job.board.apply_action(edge->first, edge->second)) {
                return false;
            }
            if (job.path_size >= BOARD_POINTS) {
                return false;
            }
            job.path[job.path_size++] = {job.node, edge};
            if (edge->child == nullptr) {
                edge->child = std::make_unique<Node>();
            }
            job.node = edge->child.get();
            if (job.board.terminal) {
                const float terminal_value = job.board.winner == 0 ? 0.0F : -1.0F;
                backup(job, terminal_value);
                --remaining[context_index];
                --total_remaining;
                ++context.last_simulations;
                ++G_TOTAL_SIMULATIONS;
                completed = true;
                break;
            }
        }
        if (completed) {
            batch.jobs.pop_back();
        }
    }

    std::vector<float> tactical_values(batch.jobs.size(), 0.0F);
    std::vector<uint8_t> tactical_flags(batch.jobs.size(), 0);
#pragma omp parallel for if(G_CPU_THREADS > 1) num_threads(G_CPU_THREADS) schedule(static)
    for (int index = 0; index < static_cast<int>(batch.jobs.size()); ++index) {
        tactical_flags[index] = static_cast<uint8_t>(
            expand_tactical(batch.jobs[index], tactical_values[index])
        );
    }
    for (std::size_t index = 0; index < batch.jobs.size(); ++index) {
        LeafJob& job = batch.jobs[index];
        if (tactical_flags[index]) {
            backup(job, tactical_values[index]);
            --remaining[job.context_index];
            --total_remaining;
            ++job.context->last_simulations;
            ++job.context->last_unique_nodes;
            ++G_TOTAL_SIMULATIONS;
            ++G_TOTAL_TACTICAL_EXPANSIONS;
        } else {
            batch.network_jobs.push_back(&job);
        }
    }
    if (batch.network_jobs.empty()) {
        return true;
    }
    const int count = static_cast<int>(batch.network_jobs.size());
    batch.boards.resize(static_cast<std::size_t>(count) * BOARD_POINTS);
    batch.players.resize(count);
    batch.first.resize(static_cast<std::size_t>(count) * G_TOP_K);
    batch.second.resize(static_cast<std::size_t>(count) * G_TOP_K);
    batch.priors.resize(static_cast<std::size_t>(count) * G_TOP_K);
    batch.values.resize(count);
    for (int index = 0; index < count; ++index) {
        const LeafJob& job = *batch.network_jobs[index];
        std::copy(
            job.board.cells.begin(),
            job.board.cells.end(),
            batch.boards.begin() + static_cast<std::size_t>(index) * BOARD_POINTS
        );
        batch.players[index] = job.board.current_player;
    }
    batch.callback_thread = std::thread([&batch, count]() {
        std::lock_guard<std::mutex> lock(G_EVAL_CALLBACK_MUTEX);
        batch.callback_status = G_EVAL_CALLBACK(
            count,
            batch.boards.data(),
            batch.players.data(),
            batch.first.data(),
            batch.second.data(),
            batch.priors.data(),
            batch.values.data()
        );
    });
    batch.launched = true;
    return true;
}

bool finish_async_lane(
    std::vector<int>& remaining,
    long long& total_remaining,
    AsyncBatch& batch
) {
    if (!batch.launched) {
        return true;
    }
    batch.callback_thread.join();
    batch.launched = false;
    if (batch.callback_status != 0) {
        return false;
    }
    const int count = static_cast<int>(batch.network_jobs.size());
    ++G_TOTAL_EVAL_BATCHES;
    G_TOTAL_EVALUATIONS += count;
    for (int index = 0; index < count; ++index) {
        LeafJob& job = *batch.network_jobs[index];
        expand_actions(
            job,
            batch.first.data() + static_cast<std::size_t>(index) * G_TOP_K,
            batch.second.data() + static_cast<std::size_t>(index) * G_TOP_K,
            batch.priors.data() + static_cast<std::size_t>(index) * G_TOP_K
        );
        const float value = std::clamp(batch.values[index], -1.0F, 1.0F);
        backup(job, value);
        ++job.context->last_simulations;
        ++job.context->last_neural_evaluations;
        ++job.context->last_unique_nodes;
        ++G_TOTAL_SIMULATIONS;
        --remaining[job.context_index];
        --total_remaining;
    }
    return true;
}

int run_multi_async_impl(Context** contexts, const int* budgets, int context_count) {
    if (G_EVAL_CALLBACK == nullptr || context_count <= 0) {
        return 1;
    }
    std::vector<int> remaining(context_count);
    std::unordered_set<Context*> unique_contexts;
    long long total_remaining = 0;
    for (int index = 0; index < context_count; ++index) {
        if (contexts[index] == nullptr || budgets[index] < 0) {
            return 1;
        }
        if (!unique_contexts.insert(contexts[index]).second) {
            return 1;
        }
        remaining[index] = budgets[index];
        total_remaining += budgets[index];
        contexts[index]->last_simulations = 0;
        contexts[index]->last_neural_evaluations = 0;
        contexts[index]->last_unique_nodes = 0;
    }
    std::vector<int> lanes[2];
    lanes[0].reserve((context_count + 1) / 2);
    lanes[1].reserve(context_count / 2);
    for (int index = 0; index < context_count; ++index) {
        lanes[index & 1].push_back(index);
    }
    AsyncBatch batches[2];
    while (total_remaining > 0) {
        if (!prepare_async_lane(
                contexts, lanes[0], remaining, total_remaining, batches[0]
            )) {
            if (batches[0].launched) {
                batches[0].callback_thread.join();
                batches[0].launched = false;
            }
            return 1;
        }
        if (!prepare_async_lane(
                contexts, lanes[1], remaining, total_remaining, batches[1]
            )) {
            if (batches[0].launched) {
                batches[0].callback_thread.join();
                batches[0].launched = false;
            }
            if (batches[1].launched) {
                batches[1].callback_thread.join();
                batches[1].launched = false;
            }
            return 1;
        }
        if (!finish_async_lane(remaining, total_remaining, batches[0])) {
            if (batches[1].launched) {
                batches[1].callback_thread.join();
                batches[1].launched = false;
            }
            return 1;
        }
        if (!finish_async_lane(remaining, total_remaining, batches[1])) {
            return 1;
        }
    }
    return 0;
}

int choose_action(Context& context, float temperature, int* first, int* second) {
    if (!context.root->expanded || context.root->edges.empty()) {
        return 0;
    }
    Edge* selected = nullptr;
    if (temperature <= 1.0e-6F) {
        selected = &*std::max_element(
            context.root->edges.begin(),
            context.root->edges.end(),
            [](const Edge& left, const Edge& right) {
                if (left.visits != right.visits) {
                    return left.visits < right.visits;
                }
                return left.prior < right.prior;
            }
        );
    } else {
        std::vector<double> weights;
        weights.reserve(context.root->edges.size());
        bool has_visits = false;
        for (const Edge& edge : context.root->edges) {
            has_visits = has_visits || edge.visits > 0;
        }
        for (const Edge& edge : context.root->edges) {
            const double base = has_visits ? edge.visits : edge.prior;
            weights.push_back(
                base > 0.0
                    ? std::pow(base, 1.0 / temperature)
                    : 0.0
            );
        }
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        selected = &context.root->edges[distribution(context.rng)];
    }
    *first = selected->first;
    *second = selected->second;
    return 1;
}

}  // namespace

EXPORT int pair_mcts_abi_version() {
    return 1;
}

EXPORT void pair_mcts_set_eval_callback(EvalCallback callback) {
    G_EVAL_CALLBACK = callback;
}

EXPORT void pair_mcts_set_params(
    int top_k,
    float cpuct,
    float root_noise_epsilon,
    float root_dirichlet_alpha
) {
    G_TOP_K = std::clamp(top_k, 1, 512);
    G_CPUCT = std::max(cpuct, 0.0F);
    G_ROOT_NOISE_EPSILON = std::clamp(root_noise_epsilon, 0.0F, 1.0F);
    G_ROOT_DIRICHLET_ALPHA = std::max(root_dirichlet_alpha, 0.0F);
}

EXPORT void pair_mcts_set_random_candidates(int count) {
    G_RANDOM_CANDIDATES = std::clamp(count, 0, G_TOP_K);
}

EXPORT void pair_mcts_set_cpu_threads(int count) {
    G_CPU_THREADS = std::max(count, 1);
}

EXPORT void pair_mcts_set_progressive_widening(int base, float scale) {
    G_PROGRESSIVE_WIDENING_BASE = std::max(base, 0);
    G_PROGRESSIVE_WIDENING_SCALE = std::isfinite(scale)
        ? std::max(scale, 0.0F)
        : 0.0F;
}

EXPORT void* pair_mcts_create_context(uint32_t seed) {
    return new Context(seed);
}

EXPORT void pair_mcts_destroy_context(void* handle) {
    delete static_cast<Context*>(handle);
}

EXPORT int pair_mcts_run_multi(void** handles, const int* budgets, int count) {
    return run_multi_impl(reinterpret_cast<Context**>(handles), budgets, count);
}

EXPORT int pair_mcts_run_multi_async(void** handles, const int* budgets, int count) {
    return run_multi_async_impl(reinterpret_cast<Context**>(handles), budgets, count);
}

EXPORT int pair_mcts_choose_action(
    void* handle,
    float temperature,
    int* first,
    int* second
) {
    if (handle == nullptr || first == nullptr || second == nullptr) {
        return 0;
    }
    return choose_action(*static_cast<Context*>(handle), temperature, first, second);
}

EXPORT int pair_mcts_get_policy(
    void* handle,
    int16_t* first,
    int16_t* second,
    float* probabilities,
    int capacity
) {
    if (handle == nullptr || capacity <= 0) {
        return 0;
    }
    Context& context = *static_cast<Context*>(handle);
    const int count = std::min(capacity, static_cast<int>(context.root->edges.size()));
    double total_visits = 0.0;
    for (int index = 0; index < count; ++index) {
        total_visits += context.root->edges[index].visits;
    }
    double prior_total = 0.0;
    if (total_visits <= 0.0) {
        for (int index = 0; index < count; ++index) {
            prior_total += context.root->edges[index].prior;
        }
    }
    for (int index = 0; index < count; ++index) {
        const Edge& edge = context.root->edges[index];
        first[index] = edge.first;
        second[index] = edge.second;
        probabilities[index] = total_visits > 0.0
            ? static_cast<float>(edge.visits / total_visits)
            : static_cast<float>(edge.prior / std::max(prior_total, 1.0e-12));
    }
    return count;
}

EXPORT float pair_mcts_get_root_value(void* handle) {
    if (handle == nullptr) {
        return 0.0F;
    }
    const Context& context = *static_cast<Context*>(handle);
    double value_sum = 0.0;
    long long visits = 0;
    for (const Edge& edge : context.root->edges) {
        value_sum += edge.value_sum;
        visits += edge.visits;
    }
    return visits > 0 ? static_cast<float>(value_sum / visits) : 0.0F;
}

EXPORT int pair_mcts_play_action(void* handle, int first, int second) {
    if (handle == nullptr) {
        return 0;
    }
    Context& context = *static_cast<Context*>(handle);
    std::unique_ptr<Node> next_root;
    if (context.root != nullptr) {
        for (Edge& edge : context.root->edges) {
            if (edge.first == first && edge.second == second) {
                next_root = std::move(edge.child);
                break;
            }
        }
    }
    if (!context.board.apply_action(first, second)) {
        return 0;
    }
    context.root = next_root != nullptr ? std::move(next_root) : std::make_unique<Node>();
    return 1;
}

EXPORT void pair_mcts_reset_tree(void* handle) {
    if (handle != nullptr) {
        static_cast<Context*>(handle)->root = std::make_unique<Node>();
    }
}

EXPORT int pair_mcts_get_winner(void* handle) {
    return handle != nullptr ? static_cast<Context*>(handle)->board.winner : 0;
}

EXPORT int pair_mcts_is_terminal(void* handle) {
    return handle != nullptr && static_cast<Context*>(handle)->board.terminal ? 1 : 0;
}

EXPORT int pair_mcts_get_total_stones(void* handle) {
    return handle != nullptr ? static_cast<Context*>(handle)->board.total_stones : 0;
}

EXPORT int pair_mcts_get_current_player(void* handle) {
    return handle != nullptr ? static_cast<Context*>(handle)->board.current_player : 0;
}

EXPORT void pair_mcts_get_last_stats(
    void* handle,
    long long* simulations,
    long long* neural_evaluations,
    long long* unique_nodes
) {
    if (handle == nullptr) {
        return;
    }
    const Context& context = *static_cast<Context*>(handle);
    if (simulations != nullptr) {
        *simulations = context.last_simulations;
    }
    if (neural_evaluations != nullptr) {
        *neural_evaluations = context.last_neural_evaluations;
    }
    if (unique_nodes != nullptr) {
        *unique_nodes = context.last_unique_nodes;
    }
}

EXPORT void pair_mcts_reset_global_stats() {
    G_TOTAL_SIMULATIONS = 0;
    G_TOTAL_EVALUATIONS = 0;
    G_TOTAL_EVAL_BATCHES = 0;
    G_TOTAL_TACTICAL_EXPANSIONS = 0;
}

EXPORT void pair_mcts_get_global_stats(
    long long* simulations,
    long long* evaluations,
    long long* batches,
    long long* tactical_expansions
) {
    if (simulations != nullptr) {
        *simulations = G_TOTAL_SIMULATIONS;
    }
    if (evaluations != nullptr) {
        *evaluations = G_TOTAL_EVALUATIONS;
    }
    if (batches != nullptr) {
        *batches = G_TOTAL_EVAL_BATCHES;
    }
    if (tactical_expansions != nullptr) {
        *tactical_expansions = G_TOTAL_TACTICAL_EXPANSIONS;
    }
}
