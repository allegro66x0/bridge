#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // NumPy配列を受け取るために必要
#include <pybind11/stl.h>   // C++のSTLコンテナを自動変換
#include "gomoku_ai.h"       // 私たちが作ったAIヘッダー

namespace py = pybind11;

/**
 * @brief Pythonから渡されたNumPy配列を、C++の 'Board' 型にコピーする
 */
Board convert_py_board(py::array_t<int, py::array::c_style> py_board) {
    if (py_board.ndim() != 2 || py_board.shape(0) != BOARD_SIZE || py_board.shape(1) != BOARD_SIZE) {
        throw std::runtime_error("Invalid board shape or dimensions.");
    }
    
    Board cpp_board = {}; // C++の盤面をゼロ初期化
    auto buf = py_board.request();
    int* ptr = static_cast<int*>(buf.ptr);
    
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            cpp_board[r][c] = ptr[r * BOARD_SIZE + c];
        }
    }
    return cpp_board;
}

/**
 * @brief Pythonから呼び出されるメイン関数
 *
 * @param py_board Pythonから渡される 13x13 の NumPy 配列
 * @param depth 探索深度
 * @return (x, y) のタプル
 */
py::tuple find_best_move_wrapper(py::array_t<int, py::array::c_style> py_board, int depth) {
    // 1. PythonのNumPy配列をC++のBoardに変換
    Board cpp_board = convert_py_board(py_board);
    
    // 2. C++のAIエンジンを呼び出す (これが高速！)
    Move best_move = find_best_move_parallel(cpp_board, depth);
    
    // 3. C++のMove {x, y} を Pythonの (x, y) タプルに変換して返す
    return py::make_tuple(best_move.x, best_move.y);
}

bool check_win_wrapper(py::array_t<int, py::array::c_style> py_board, int x, int y, int player) {
    Board cpp_board = convert_py_board(py_board);
    return check_win(cpp_board, x, y, player);
}

std::vector<int> count_pattern_counts_wrapper(py::array_t<int, py::array::c_style> py_board, int player) {
    Board cpp_board = convert_py_board(py_board);
    return count_pattern_counts(cpp_board, player);
}

// "cpp_gomoku_ai" という名前のモジュールを定義
PYBIND11_MODULE(cpp_gomoku_ai, m) {
    m.doc() = "Gomoku AI C++ engine"; // モジュールの説明
    
    // Python側から "find_best_move" という名前で呼び出せるようにする
    m.def("find_best_move", &find_best_move_wrapper, "Find the best move using C++ engine",
          py::arg("board"), py::arg("depth"));

    m.def("check_win", &check_win_wrapper, "Check if the move is a winning move",
          py::arg("board"), py::arg("x"), py::arg("y"), py::arg("player"));
          
    m.def("count_pattern_counts", &count_pattern_counts_wrapper, "Count patterns (5,4,3,2) for player",
          py::arg("board"), py::arg("player"));
}