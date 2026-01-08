#ifndef C6_LOGIC_H
#define C6_LOGIC_H

#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>

// Board Size
const int BOARD_SIZE = 19;
const int N = BOARD_SIZE * BOARD_SIZE;

// Players
const int EMPTY = 0;
const int BLACK = 1;
const int WHITE = -1;

class Connect6Board {
public:
    int board[N];
    int current_player;
    int stones_in_turn; // Stones played in current turn
    int total_stones;   // Total stones played

    Connect6Board() {
        reset();
    }

    void reset() {
        std::memset(board, 0, sizeof(board));
        current_player = BLACK; // Black starts
        stones_in_turn = 0;
        total_stones = 0;
    }

    // Coord conversion
    inline int idx(int r, int c) const {
        return r * BOARD_SIZE + c;
    }

    inline bool is_on_board(int r, int c) const {
        return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
    }

    // Make move and update state
    void make_move(int move_idx) {
        if (board[move_idx] != EMPTY) return;
        
        board[move_idx] = current_player;
        stones_in_turn++;
        total_stones++;
        
        // Switch player logic
        // Rule: 1st move (total=1) switch.
        // Then every 2 moves (stones_in_turn=2) switch.
        
        if (total_stones == 1) {
            current_player = -current_player;
            stones_in_turn = 0;
        } else if (stones_in_turn >= 2) {
            current_player = -current_player;
            stones_in_turn = 0;
        }
    }

    // Fast win check (last move only)
    // Returns: 0(None), 1(Black), -1(White)
    int check_win(int move_idx) const {
        if (move_idx < 0 || move_idx >= N) return 0;
        
        int r = move_idx / BOARD_SIZE;
        int c = move_idx % BOARD_SIZE;
        int color = board[move_idx];
        
        if (color == EMPTY) return 0;

        // 4 directions: Horizontal, Vertical, Diagonal, Anti-Diagonal
        int dr[4] = {0, 1, 1, 1};
        int dc[4] = {1, 0, 1, -1};

        for (int i = 0; i < 4; ++i) {
            int count = 1;
            // Forward
            for (int k = 1; k < 6; ++k) {
                int nr = r + dr[i] * k;
                int nc = c + dc[i] * k;
                if (!is_on_board(nr, nc) || board[idx(nr, nc)] != color) break;
                count++;
            }
            // Backward
            for (int k = 1; k < 6; ++k) {
                int nr = r - dr[i] * k;
                int nc = c - dc[i] * k;
                if (!is_on_board(nr, nc) || board[idx(nr, nc)] != color) break;
                count++;
            }

            if (count >= 6) return color;
        }
        return 0;
    }

    // Get legal moves
    std::vector<int> get_legal_moves() const {
        std::vector<int> moves;
        moves.reserve(361 - total_stones);
        for (int i = 0; i < N; ++i) {
            if (board[i] == EMPTY) {
                moves.push_back(i);
            }
        }
        return moves;
    }
};

#endif
