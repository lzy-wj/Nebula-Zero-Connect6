#ifndef C6_LOGIC_H
#define C6_LOGIC_H

#include <algorithm>
#include <cstring>
#include <set>
#include <vector>


const int BOARD_SIZE = 19;
const int N = BOARD_SIZE * BOARD_SIZE;
const int EMPTY = 0;
const int BLACK = 1;
const int WHITE = -1;

class Connect6Board {
public:
  int board[N];
  int current_player;
  int stones_in_turn;
  int total_stones;

  Connect6Board() { reset(); }

  void reset() {
    std::memset(board, 0, sizeof(board));
    current_player = BLACK;
    stones_in_turn = 0;
    total_stones = 0;
  }

  inline int idx(int r, int c) const { return r * BOARD_SIZE + c; }
  inline bool is_on_board(int r, int c) const {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
  }

  void make_move(int move_idx) {
    if (board[move_idx] != EMPTY)
      return;
    board[move_idx] = current_player;
    stones_in_turn++;
    total_stones++;

    if (total_stones == 1) {
      current_player = -current_player;
      stones_in_turn = 0;
    } else if (stones_in_turn >= 2) {
      current_player = -current_player;
      stones_in_turn = 0;
    }
  }

  int check_win(int move_idx) const {
    if (move_idx < 0 || move_idx >= N)
      return 0;
    int r = move_idx / BOARD_SIZE;
    int c = move_idx % BOARD_SIZE;
    int color = board[move_idx];
    if (color == EMPTY)
      return 0;

    int dr[4] = {0, 1, 1, 1};
    int dc[4] = {1, 0, 1, -1};

    for (int i = 0; i < 4; ++i) {
      int count = 1;
      for (int k = 1; k < 6; ++k) {
        int nr = r + dr[i] * k, nc = c + dc[i] * k;
        if (!is_on_board(nr, nc) || board[idx(nr, nc)] != color)
          break;
        count++;
      }
      for (int k = 1; k < 6; ++k) {
        int nr = r - dr[i] * k, nc = c - dc[i] * k;
        if (!is_on_board(nr, nc) || board[idx(nr, nc)] != color)
          break;
        count++;
      }
      if (count >= 6)
        return color;
    }
    return 0;
  }

  // ================================================================
  //                   新增：滑动窗口威胁检测
  // ================================================================

  // 检查在 move_idx 放置 color 后，是否在任一方向形成威胁
  // 使用滑动窗口方法，可检测跳子棋型如 ●●●_●
  bool check_threat_window(int move_idx, int color, int required_stones) const {
    int r = move_idx / BOARD_SIZE;
    int c = move_idx % BOARD_SIZE;
    int dr[4] = {0, 1, 1, 1};
    int dc[4] = {1, 0, 1, -1};

    for (int dir = 0; dir < 4; ++dir) {
      // 在 6 格窗口内滑动检测
      // 窗口起点从 (r,c) 向后最多 5 格开始
      for (int start = -5; start <= 0; ++start) {
        int stones = 0;
        int empties = 0;
        int target_in_window = false;

        for (int offset = 0; offset < 6; ++offset) {
          int nr = r + dr[dir] * (start + offset);
          int nc = c + dc[dir] * (start + offset);

          if (!is_on_board(nr, nc)) {
            stones = -1; // 无效窗口
            break;
          }

          int pos = idx(nr, nc);
          if (pos == move_idx) {
            // 假设在 move_idx 放置棋子
            stones++;
            target_in_window = true;
          } else if (board[pos] == color) {
            stones++;
          } else if (board[pos] == EMPTY) {
            empties++;
          } else {
            // 对方棋子，窗口无效
            stones = -1;
            break;
          }
        }

        // 窗口内有 required_stones 颗同色棋子（包含假设放置的那颗）
        if (stones >= required_stones && target_in_window) {
          return true;
        }
      }
    }
    return false;
  }

  // 检查该位置是否是潜在威胁点（对手落子后形成5连或跳5连）
  bool check_potential_threat(int move_idx, int color) const {
    return check_threat_window(move_idx, color, 5);
  }

  // 检查落子后是否直接成6连（包括跳子成6连）
  bool check_winning_threat(int move_idx, int color) const {
    return check_threat_window(move_idx, color, 6);
  }

  // ================================================================
  //                   连续棋子检测（保留原逻辑）
  // ================================================================

  // 检测连续 n 连（不含跳子），用于必杀判断
  bool check_connect_n(int move_idx, int color, int n) const {
    int r = move_idx / BOARD_SIZE;
    int c = move_idx % BOARD_SIZE;
    int dr[4] = {0, 1, 1, 1};
    int dc[4] = {1, 0, 1, -1};

    for (int i = 0; i < 4; ++i) {
      int count = 1;
      int f_r = r, f_c = c, b_r = r, b_c = c;

      for (int k = 1; k < 6; ++k) {
        int nr = r + dr[i] * k, nc = c + dc[i] * k;
        if (!is_on_board(nr, nc) || board[idx(nr, nc)] != color)
          break;
        count++;
        f_r = nr;
        f_c = nc;
      }
      for (int k = 1; k < 6; ++k) {
        int nr = r - dr[i] * k, nc = c - dc[i] * k;
        if (!is_on_board(nr, nc) || board[idx(nr, nc)] != color)
          break;
        count++;
        b_r = nr;
        b_c = nc;
      }

      if (count >= 6)
        return true;
      if (count >= n) {
        int nfr = f_r + dr[i], nfc = f_c + dc[i];
        int nbr = b_r - dr[i], nbc = b_c - dc[i];
        bool f_open = is_on_board(nfr, nfc) && board[idx(nfr, nfc)] == EMPTY;
        bool b_open = is_on_board(nbr, nbc) && board[idx(nbr, nbc)] == EMPTY;
        if (f_open || b_open)
          return true;
      }
    }
    return false;
  }

  // ================================================================
  //                   必杀点检测
  // ================================================================

  std::vector<int> get_winning_moves() const {
    std::vector<int> winning;
    for (int i = 0; i < N; ++i) {
      if (board[i] == EMPTY) {
        // 直接成6连（连续或跳子）
        if (check_winning_threat(i, current_player)) {
          winning.push_back(i);
          continue;
        }
        // 回合第一子成5连（还能再下一子）
        if (stones_in_turn == 0 && check_potential_threat(i, current_player)) {
          winning.push_back(i);
        }
      }
    }
    return winning;
  }

  // ================================================================
  //                   强制防守点检测（含跳子威胁）
  // ================================================================

  std::vector<int> get_forced_moves() const {
    std::vector<int> forced;
    int opponent = -current_player;
    for (int i = 0; i < N; ++i) {
      if (board[i] == EMPTY && check_potential_threat(i, opponent)) {
        forced.push_back(i);
      }
    }
    return forced;
  }

  // ================================================================
  //                   高级威胁检测：统计威胁数量
  // ================================================================

  // 统计某个颜色在某方向上的威胁数量（活四、冲四等）
  int count_threats(int color, int min_stones) const {
    int threat_count = 0;
    std::set<int> counted; // 避免重复计数

    for (int i = 0; i < N; ++i) {
      if (board[i] == EMPTY) {
        if (check_threat_window(i, color, min_stones)) {
          if (counted.find(i) == counted.end()) {
            threat_count++;
            counted.insert(i);
          }
        }
      }
    }
    return threat_count;
  }

  // 检查对手是否有双杀威胁（两个以上的强制防守点）
  bool has_double_threat(int color) const {
    int threat_count = 0;
    for (int i = 0; i < N; ++i) {
      if (board[i] == EMPTY && check_potential_threat(i, color)) {
        threat_count++;
        if (threat_count >= 2)
          return true;
      }
    }
    return false;
  }

  // ================================================================
  //                   合法着法
  // ================================================================

  std::vector<int> get_legal_moves() const {
    std::vector<int> moves;
    moves.reserve(361 - total_stones);
    for (int i = 0; i < N; ++i) {
      if (board[i] == EMPTY)
        moves.push_back(i);
    }
    return moves;
  }
};

#endif
