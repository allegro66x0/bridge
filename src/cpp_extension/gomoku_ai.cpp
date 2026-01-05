#include "gomoku_ai.h"
#include <cmath>        // std::abs
#include <limits>       // std::numeric_limits
#include <algorithm>    // std::max, std::min, std::sort
#include <vector>
#include <map>
#include <set>
#include <future>       // std::async, std::future (並列処理)
#include <thread>       // std::thread

// --- PythonのV7.3ロジックをC++に移植 ---

// --- ユーティリティ ---
// (Pythonの `OPENING_BOOK` に相当)
std::map<Board, Move> OPENING_BOOK = {
    // 空のボード(Board{}) に対しては、中央(Move::center()) を返す
    {Board{}, Move::center()}
};

// (Pythonの `check_win`)
bool check_win(const Board& board, int x, int y, int player) {
    const std::array<std::pair<int, int>, 4> directions = {
        {{1, 0}, {0, 1}, {1, 1}, {1, -1}}
    };
    for (const auto& dir : directions) {
        int dx = dir.first;
        int dy = dir.second;
        int count = 1;
        for (int i = 1; i < 5; ++i) {
            int nx = x + i * dx;
            int ny = y + i * dy;
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE || board[ny][nx] != player) break;
            count++;
        }
        for (int i = 1; i < 5; ++i) {
            int nx = x - i * dx;
            int ny = y - i * dy;
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE || board[ny][nx] != player) break;
            count++;
        }
        if (count >= 5) return true;
    }
    return false;
}

// (Pythonの `calculate_score` - V7.1の攻撃的ロジック)
double calculate_score(const Board& board, int player) {
    double score = 0;
    int opponent = (player == AI_PLAYER) ? HUMAN_PLAYER : AI_PLAYER;
    const std::array<std::pair<int, int>, 4> directions = {
        {{0, 1}, {1, 0}, {1, 1}, {1, -1}}
    };

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            for (const auto& dir : directions) {
                if (board[r][c] == player) {
                    int dr = dir.first;
                    int dc = dir.second;
                    int stone_count = 0;
                    int open_ends = 0;

                    // 始点の前
                    int r_prev = r - dr;
                    int c_prev = c - dc;
                    if (r_prev >= 0 && r_prev < BOARD_SIZE && c_prev >= 0 && c_prev < BOARD_SIZE) {
                        if (board[r_prev][c_prev] == EMPTY) open_ends++;
                    }

                    // 連続する石
                    for (int i = 0; i < 5; ++i) {
                        int nr = r + i * dr;
                        int nc = c + i * dc;
                        if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                            if (board[nr][nc] == player) {
                                stone_count++;
                            } else if (board[nr][nc] == EMPTY) {
                                break;
                            } else {
                                stone_count = -1; // 妨害
                                break;
                            }
                        } else {
                            break; // 盤外
                        }
                    }

                    if (stone_count <= 0) continue;

                    // 終点の次
                    int r_next = r + stone_count * dr;
                    int c_next = c + stone_count * dc;
                    if (r_next >= 0 && r_next < BOARD_SIZE && c_next >= 0 && c_next < BOARD_SIZE) {
                        if (board[r_next][c_next] == EMPTY) open_ends++;
                    }

                    // スコア計算
                    if (stone_count >= 5) score += 1000000;
                    else if (stone_count == 4) score += (open_ends == 2) ? 100000 : 10000;
                    else if (stone_count == 3) score += (open_ends == 2) ? 5000 : 100;
                    else if (stone_count == 2) score += (open_ends == 2) ? 50 : 10;
                }
            }
        }
    }
    return score;
}

// (Pythonの `evaluate_board` - V7.1の攻撃的ロジック)
double evaluate_board(const Board& board) {
    double ai_score = calculate_score(board, AI_PLAYER);
    double player_score = calculate_score(board, HUMAN_PLAYER);
    return ai_score * 1.5 - player_score;
}

// (Pythonの `get_candidate_moves`)
std::vector<Move> get_candidate_moves(const Board& board) {
    std::set<Move> candidates; // setで重複を自動的に排除
    bool has_stone = false;

    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            if (board[y][x] != EMPTY) {
                has_stone = true;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] == EMPTY) {
                            candidates.insert({nx, ny});
                        }
                    }
                }
            }
        }
    }

    if (!has_stone) {
        return {Move::center()}; // 最初の1手
    }
    
    // set から vector に変換して返す
    return std::vector<Move>(candidates.begin(), candidates.end());
}

// (Pythonの `get_move_score` - V7.1の攻撃優先ロジック)
double get_move_score(Board& board, int x, int y, int player) {
    int opponent = (player == AI_PLAYER) ? HUMAN_PLAYER : AI_PLAYER;
    
    // 勝利
    board[y][x] = player;
    if (check_win(board, x, y, player)) {
        board[y][x] = EMPTY; return 1000000;
    }
    board[y][x] = EMPTY;
    
    double score = 0;

    // 攻撃評価
    board[y][x] = player;
    for (const auto& dir : std::array<std::pair<int, int>, 4>{{{1, 0}, {0, 1}, {1, 1}, {1, -1}}}) {
        int count = 1, open_ends = 0;
        for (int i = 1; i < 4; ++i) {
            int nx = x + i * dir.first, ny = y + i * dir.second;
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE || board[ny][nx] != player) {
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] == EMPTY) open_ends++;
                break;
            }
            count++;
        }
        for (int i = 1; i < 4; ++i) {
            int nx = x - i * dir.first, ny = y - i * dir.second;
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE || board[ny][nx] != player) {
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] == EMPTY) open_ends++;
                break;
            }
            count++;
        }
        if (count == 4 && open_ends >= 1) score += 20000; // 攻撃4
        else if (count == 3 && open_ends == 2) score += 5000; // 攻撃3
    }
    board[y][x] = EMPTY;

    // 防御評価
    board[y][x] = opponent;
    if (check_win(board, x, y, opponent)) {
        board[y][x] = EMPTY; return 500000; // 相手の勝利を阻止
    }
    for (const auto& dir : std::array<std::pair<int, int>, 4>{{{1, 0}, {0, 1}, {1, 1}, {1, -1}}}) {
        int count = 1, open_ends = 0;
        for (int i = 1; i < 4; ++i) {
            int nx = x + i * dir.first, ny = y + i * dir.second;
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE || board[ny][nx] != opponent) {
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] == EMPTY) open_ends++;
                break;
            }
            count++;
        }
        for (int i = 1; i < 4; ++i) {
            int nx = x - i * dir.first, ny = y - i * dir.second;
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE || board[ny][nx] != opponent) {
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] == EMPTY) open_ends++;
                break;
            }
            count++;
        }
        if (count == 4 && open_ends >= 1) score += 10000; // 防御4
        else if (count == 3 && open_ends == 2) score += 5000; // 防御3
    }
    board[y][x] = EMPTY;
    return score;
}


// --- メインの探索関数 (Minimax + 静止探索) ---

// 相互再帰のため、minimaxの宣言が先に関必要
ScoreMove minimax(Board& board, int depth, int player, double alpha, double beta);

// (Pythonの `quiescence_search` - V7.3の効率化ロジック)
double quiescence_search(Board& board, int depth, int player, double alpha, double beta) {
    double stand_pat_score = evaluate_board(board);

    if (player == AI_PLAYER) {
        alpha = std::max(alpha, stand_pat_score);
    } else {
        beta = std::min(beta, stand_pat_score);
    }

    if (depth == 0 || beta <= alpha) {
        return stand_pat_score;
    }

    auto all_moves = get_candidate_moves(board);
    std::vector<std::pair<double, Move>> forcing_moves;
    for (const auto& move : all_moves) {
        double score = get_move_score(board, move.x, move.y, player);
        if (score >= 10000) { // V7.3: 4並び(10000)以上の手のみ
            forcing_moves.push_back({score, move});
        }
    }

    if (forcing_moves.empty()) {
        return stand_pat_score;
    }

    std::sort(forcing_moves.rbegin(), forcing_moves.rend()); // スコアで降順ソート

    if (player == AI_PLAYER) {
        double max_eval = stand_pat_score;
        for (const auto& scored_move : forcing_moves) {
            Move move = scored_move.second;
            board[move.y][move.x] = AI_PLAYER;
            double eval = quiescence_search(board, depth - 1, HUMAN_PLAYER, alpha, beta);
            board[move.y][move.x] = EMPTY;
            max_eval = std::max(max_eval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;
        }
        return max_eval;
    } else {
        double min_eval = stand_pat_score;
        for (const auto& scored_move : forcing_moves) {
            Move move = scored_move.second;
            board[move.y][move.x] = HUMAN_PLAYER;
            double eval = quiescence_search(board, depth - 1, AI_PLAYER, alpha, beta);
            board[move.y][move.x] = EMPTY;
            min_eval = std::min(min_eval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

// (Pythonの `minimax` - V7.2の静止探索呼び出しロジック)
ScoreMove minimax(Board& board, int depth, int player, double alpha, double beta) {
    if (depth == 0) {
        // 深さ0に達したら、静止探索 (深さ3) を呼び出す
        double score = quiescence_search(board, 3, player, alpha, beta);
        return {score, {-1, -1}};
    }

    auto moves = get_candidate_moves(board);
    if (moves.empty()) {
        return {evaluate_board(board), {-1, -1}};
    }

    ScoreMove best_result;
    if (player == AI_PLAYER) {
        best_result.score = -std::numeric_limits<double>::infinity();
        
        std::vector<std::pair<double, Move>> scored_moves;
        for (const auto& move : moves) {
            scored_moves.push_back({get_move_score(board, move.x, move.y, AI_PLAYER), move});
        }
        std::sort(scored_moves.rbegin(), scored_moves.rend());

        for (const auto& scored_move : scored_moves) {
            Move move = scored_move.second;
            board[move.y][move.x] = AI_PLAYER;
            if (check_win(board, move.x, move.y, AI_PLAYER)) {
                board[move.y][move.x] = EMPTY;
                return {1000000.0 + depth, move}; // 勝利
            }
            
            ScoreMove result = minimax(board, depth - 1, HUMAN_PLAYER, alpha, beta);
            board[move.y][move.x] = EMPTY;

            if (result.score > best_result.score) {
                best_result.score = result.score;
                best_result.move = move;
            }
            alpha = std::max(alpha, best_result.score);
            if (beta <= alpha) break;
        }
    } else {
        best_result.score = std::numeric_limits<double>::infinity();

        std::vector<std::pair<double, Move>> scored_moves;
        for (const auto& move : moves) {
            scored_moves.push_back({get_move_score(board, move.x, move.y, HUMAN_PLAYER), move});
        }
        std::sort(scored_moves.rbegin(), scored_moves.rend());

        for (const auto& scored_move : scored_moves) {
            Move move = scored_move.second;
            board[move.y][move.x] = HUMAN_PLAYER;
            if (check_win(board, move.x, move.y, HUMAN_PLAYER)) {
                board[move.y][move.x] = EMPTY;
                return {-1000000.0 - depth, move}; // 敗北
            }
            
            ScoreMove result = minimax(board, depth - 1, AI_PLAYER, alpha, beta);
            board[move.y][move.x] = EMPTY;

            if (result.score < best_result.score) {
                best_result.score = result.score;
                best_result.move = move;
            }
            beta = std::min(beta, best_result.score);
            if (beta <= alpha) break;
        }
    }
    return best_result;
}

// (Pythonの `_eval_move_task`)
// 並列処理されるタスク。盤面はコピー(値渡し)で受け取る
ScoreMove eval_move_task(Board board, Move move, int depth) {
    board[move.y][move.x] = AI_PLAYER;
    
    // 勝利チェック
    if (check_win(board, move.x, move.y, AI_PLAYER)) {
        return {1000000.0 + depth, move};
    }

    // 相手の手番でminimaxを呼び出す
    ScoreMove result = minimax(board, depth - 1, HUMAN_PLAYER, 
                               -std::numeric_limits<double>::infinity(), 
                               std::numeric_limits<double>::infinity());
    
    // このタスクの評価値は「result.score」だが、
    // 最終的に必要なのは「どの "最初の手(move)" が最高の結果に繋がったか」
    return {result.score, move};
}


// (Pythonの `find_best_move_parallel`)
Move find_best_move_parallel(Board board, int depth) {
    // 1. 定石チェック
    if (OPENING_BOOK.count(board)) {
        return OPENING_BOOK.at(board);
    }

    // 2. 候補手を取得し、Move Ordering
    auto moves = get_candidate_moves(board);
    if (moves.empty()) return Move::center(); // 万が一

    std::vector<std::pair<double, Move>> scored_moves;
    for (const auto& move : moves) {
        // C++ではBoard&が変更されるため、ダミーのBoardをコピーして渡す
        Board temp_board = board;
        scored_moves.push_back({get_move_score(temp_board, move.x, move.y, AI_PLAYER), move});
    }
    std::sort(scored_moves.rbegin(), scored_moves.rend());

    // 3. 並列処理のタスクを準備
    std::vector<std::future<ScoreMove>> futures;
    
    for (const auto& scored_move : scored_moves) {
        Move move = scored_move.second;
        // std::asyncでタスクを非同期に起動
        // board (コピーが渡される), move, depth-1
        futures.push_back(
            std::async(std::launch::async, eval_move_task, board, move, depth)
        );
    }

    // 4. 結果を収集
    ScoreMove best_result;
    best_result.score = -std::numeric_limits<double>::infinity();

    for (auto& f : futures) {
        ScoreMove result = f.get(); // スレッドの結果が返ってくるまで待機
        if (result.score > best_result.score) {
            best_result = result; // 最高のスコアと手を更新
        }
    }

    return best_result.move;
}