#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <random>

namespace {

constexpr int BOARD_SIZE = 19;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int PAIR_RANK = 16;

// 防止编译器把整个基准当作无用计算删除。
volatile float G_CHECKSUM = 0.0f;

int dot_i8_scalar(const int8_t* left, const int8_t* right) {
    int result = 0;
    for (int index = 0; index < PAIR_RANK; ++index) {
        result += static_cast<int>(left[index]) * static_cast<int>(right[index]);
    }
    return result;
}

int horizontal_sum_i32(__m256i values) {
    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

int dot_i8_avx2(const int8_t* left, const int8_t* right) {
    // 16 个 int8 一次扩展成 int16，再用 madd 将相邻乘积累加为 int32。
    const __m128i left_i8 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(left)
    );
    const __m128i right_i8 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(right)
    );
    const __m256i left_i16 = _mm256_cvtepi8_epi16(left_i8);
    const __m256i right_i16 = _mm256_cvtepi8_epi16(right_i8);
    return horizontal_sum_i32(_mm256_madd_epi16(left_i16, right_i16));
}

template <bool UseSimd>
void make_logits(
    const int8_t* candidate_factors,
    const int8_t* first_factors,
    const float* base_logits,
    const float* relative_lookup,
    int first_move,
    float factor_scale,
    float* output
) {
    const int8_t* first = first_factors + first_move * PAIR_RANK;
    const float* relative = relative_lookup + first_move * BOARD_CELLS;
    for (int move = 0; move < BOARD_CELLS; ++move) {
        const int8_t* candidate = candidate_factors + move * PAIR_RANK;
        const int dot = UseSimd
            ? dot_i8_avx2(candidate, first)
            : dot_i8_scalar(candidate, first);
        output[move] = base_logits[move]
            + relative[move]
            + static_cast<float>(dot) * factor_scale;
    }
}

void softmax(float* values) {
    const float maximum = *std::max_element(values, values + BOARD_CELLS);
    float sum = 0.0f;
    for (int index = 0; index < BOARD_CELLS; ++index) {
        values[index] = std::exp(values[index] - maximum);
        sum += values[index];
    }
    const float inverse_sum = 1.0f / sum;
    for (int index = 0; index < BOARD_CELLS; ++index) {
        values[index] *= inverse_sum;
    }
}

template <typename Function>
double benchmark(const char* name, int iterations, Function&& function) {
    for (int iteration = 0; iteration < 2000; ++iteration) {
        function(iteration);
    }
    const auto started_at = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        function(iteration);
    }
    const auto stopped_at = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stopped_at - started_at).count();
    const double microseconds = seconds * 1e6 / static_cast<double>(iterations);
    std::cout << std::left << std::setw(22) << name
              << std::right << std::fixed << std::setprecision(3)
              << microseconds << " us/policy, "
              << std::setprecision(1) << (1e6 / microseconds) << " policy/s\n";
    return microseconds;
}

}  // namespace

int main() {
    alignas(64) std::array<int8_t, BOARD_CELLS * PAIR_RANK> candidate_factors{};
    alignas(64) std::array<int8_t, BOARD_CELLS * PAIR_RANK> first_factors{};
    alignas(64) std::array<float, BOARD_CELLS> base_logits{};
    alignas(64) std::array<float, BOARD_CELLS * BOARD_CELLS> relative_lookup{};
    alignas(64) std::array<float, BOARD_CELLS> output{};

    std::mt19937 generator(2026);
    std::uniform_int_distribution<int> factor_distribution(-127, 127);
    std::normal_distribution<float> logit_distribution(0.0f, 1.0f);
    for (int8_t& value : candidate_factors) {
        value = static_cast<int8_t>(factor_distribution(generator));
    }
    for (int8_t& value : first_factors) {
        value = static_cast<int8_t>(factor_distribution(generator));
    }
    for (float& value : base_logits) {
        value = logit_distribution(generator);
    }

    // 实际模型只保存 37x37 位移偏置；基准预展开为查询表，避免热路径整数除法。
    std::array<float, (BOARD_SIZE * 2 - 1) * (BOARD_SIZE * 2 - 1)> relative_bias{};
    for (float& value : relative_bias) {
        value = logit_distribution(generator) * 0.1f;
    }
    for (int first = 0; first < BOARD_CELLS; ++first) {
        const int first_row = first / BOARD_SIZE;
        const int first_column = first % BOARD_SIZE;
        for (int move = 0; move < BOARD_CELLS; ++move) {
            const int relative_row = move / BOARD_SIZE - first_row + BOARD_SIZE - 1;
            const int relative_column = move % BOARD_SIZE - first_column + BOARD_SIZE - 1;
            relative_lookup[first * BOARD_CELLS + move] = relative_bias[
                relative_row * (BOARD_SIZE * 2 - 1) + relative_column
            ];
        }
    }

    constexpr float factor_scale = 0.00025f;
    constexpr int iterations = 200000;
    benchmark("int8 scalar logits", iterations, [&](int iteration) {
        make_logits<false>(
            candidate_factors.data(),
            first_factors.data(),
            base_logits.data(),
            relative_lookup.data(),
            iteration % BOARD_CELLS,
            factor_scale,
            output.data()
        );
        G_CHECKSUM += output[iteration % BOARD_CELLS] * 1e-12f;
    });
    benchmark("int8 AVX2 logits", iterations, [&](int iteration) {
        make_logits<true>(
            candidate_factors.data(),
            first_factors.data(),
            base_logits.data(),
            relative_lookup.data(),
            iteration % BOARD_CELLS,
            factor_scale,
            output.data()
        );
        G_CHECKSUM += output[iteration % BOARD_CELLS] * 1e-12f;
    });
    benchmark("int8 AVX2 + softmax", iterations, [&](int iteration) {
        make_logits<true>(
            candidate_factors.data(),
            first_factors.data(),
            base_logits.data(),
            relative_lookup.data(),
            iteration % BOARD_CELLS,
            factor_scale,
            output.data()
        );
        softmax(output.data());
        G_CHECKSUM += output[iteration % BOARD_CELLS] * 1e-12f;
    });
    std::cout << "checksum: " << G_CHECKSUM << '\n';
    return 0;
}
