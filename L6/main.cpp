#include "gomoku_ai.h"
#include <iostream>
#include <chrono> // 時間計測

// 盤面を描画するヘルパー関数
void print_board(const Board& board) {
    std::cout << "   ";
    for (int i = 0; i < BOARD_SIZE; ++i) {
        printf("%02d ", i);
    }
    std::cout << "\n";
    std::cout << "------------------------------------------\n";
    for (int r = 0; r < BOARD_SIZE; ++r) {
        printf("%02d| ", r);
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (board[r][c] == AI_PLAYER) std::cout << "X  ";
            else if (board[r][c] == HUMAN_PLAYER) std::cout << "O  ";
            else std::cout << ".  ";
        }
        std::cout << "\n";
    }
}

int main() {
    // 1. 空の盤面を作成 (C++では {} でゼロ初期化される)
    Board current_board = {}; 
    int search_depth = 5; // PythonのSEARCH_DEPTH

    std::cout << "Gomoku AI (C++) Engine Test" << std::endl;
    std::cout << "AIが先手(X)です。探索深度: " << search_depth << std::endl;
    print_board(current_board);

    // 2. AIの初手を計算 (定石が使われるはず)
    auto start = std::chrono::high_resolution_clock::now();
    
    Move ai_move = find_best_move_parallel(current_board, search_depth);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::cout << "\nAIが " << elapsed_ms.count() << " ミリ秒で着手しました。" << std::endl;
    printf("AIの手: %02d%02d\n", ai_move.x, ai_move.y);

    // 3. 盤面に反映
    current_board[ai_move.y][ai_move.x] = AI_PLAYER;
    print_board(current_board);

    return 0;
}