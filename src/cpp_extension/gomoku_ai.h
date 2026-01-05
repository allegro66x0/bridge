#pragma once

#include <vector>
#include <array>
#include <set>
#include <map>
#include <string>

// --- 定数 ---
constexpr int BOARD_SIZE = 13;
constexpr int AI_PLAYER = 1;
constexpr int HUMAN_PLAYER = 2;
constexpr int EMPTY = 0;

// --- 型の定義 ---

// 盤面 (13x13の2次元配列)
using Board = std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE>;

// 手 (x, y)
struct Move {
    int x = -1;
    int y = -1;

    // 盤面の中央 (定石用)
    static Move center() {
        return {BOARD_SIZE / 2, BOARD_SIZE / 2};
    }

    // std::map のキーとして使うための比較演算子
    bool operator<(const Move& other) const {
        if (y != other.y) return y < other.y;
        return x < other.x;
    }
};

// 評価値と手をセットにする (Minimaxの戻り値用)
struct ScoreMove {
    double score = -1e18; // マイナス無限大
    Move move = {-1, -1};
};


// --- AIのメイン関数 (Pythonから呼び出すことを想定) ---

/**
 * @brief 現在の盤面と探索深度を受け取り、AIの最善手を返す
 * @param board 現在の盤面 (13x13)
 * @param depth メインの探索深度 (例: 5)
 * @return 最善と判断された手 (Move {x, y})
 */
Move find_best_move_parallel(Board board, int depth);