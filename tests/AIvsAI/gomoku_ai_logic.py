# gomoku_ai_logic.py
import numpy as np

BOARD_SIZE_AI = 13
GOMOKU_AI_COLOR = -1  # 後手 (白)
RENJU_AI_COLOR = 1   # 先手 (黒)

# board.board (リストのリスト) を直接参照するように修正
def check_win(board_obj, x, y, player):
    """(x, y)に置かれた石によって、playerが勝利したかどうかを判定する"""
    # この関数は五目並べAIの評価関数内でのみ使用される
    # 実際の勝利判定は battle_board.py の check_win を使う
    board = board_obj.board # Boardオブジェクトから盤面リストを取得
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if not (0 <= nx < BOARD_SIZE_AI and 0 <= ny < BOARD_SIZE_AI and board[ny][nx] == player): break
            count += 1
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if not (0 <= nx < BOARD_SIZE_AI and 0 <= ny < BOARD_SIZE_AI and board[ny][nx] == player): break
            count += 1
        # 五目並べは6以上でも勝ち
        if count >= 5: return True
    return False

def evaluate_line(line, player):
    opponent = RENJU_AI_COLOR if player == GOMOKU_AI_COLOR else GOMOKU_AI_COLOR
    if opponent in line: return 0
    player_stones = line.count(player)
    if player_stones == 5: return 100000
    if player_stones == 4: return 1000
    if player_stones == 3: return 100
    if player_stones == 2: return 10
    return 0

def count_patterns_for_player(board_obj, player):
    board = board_obj.board
    score = 0
    for r in range(BOARD_SIZE_AI):
        for c in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board[r][c+i] for i in range(5)], player)
    for c in range(BOARD_SIZE_AI):
        for r in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board[r+i][c] for i in range(5)], player)
    for r in range(BOARD_SIZE_AI - 4):
        for c in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board[r+i][c+i] for i in range(5)], player)
    for r in range(4, BOARD_SIZE_AI):
        for c in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board[r-i][c+i] for i in range(5)], player)
    return score

def evaluate(board_obj):
    # GOMOKU_AI_COLOR (白, -1) のスコア - RENJU_AI_COLOR (黒, 1) のスコア
    return count_patterns_for_player(board_obj, GOMOKU_AI_COLOR) - count_patterns_for_player(board_obj, RENJU_AI_COLOR) * 1.5

def get_candidate_moves(board_obj):
    # こちらは五目並べAIの探索範囲ロジック
    board = board_obj.board
    candidates = set()
    for y in range(BOARD_SIZE_AI):
        for x in range(BOARD_SIZE_AI):
            if board[y][x] != 0:
                for dy in range(-3, 4): # 3マス範囲
                    for dx in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < BOARD_SIZE_AI and 0 <= ny < BOARD_SIZE_AI and board[ny][nx] == 0:
                            candidates.add((nx, ny))
    if not candidates: 
        # 盤面に石がない場合、中央を返す
        return {(BOARD_SIZE_AI // 2, BOARD_SIZE_AI // 2)}
    return candidates

# minimaxもboardオブジェクトを引数に取るように修正
def minimax(board_obj, depth, player, alpha, beta):
    if depth == 0: return evaluate(board_obj), None
    
    # 候補手は連珠AIの `generate_moves` (2マス範囲) ではなく
    # 五目並べAIの `get_candidate_moves` (3マス範囲) を使用
    moves = get_candidate_moves(board_obj)
    best_move = next(iter(moves)) if moves else None
    
    board = board_obj.board # 内部のリストにアクセス

    if player == GOMOKU_AI_COLOR: # 後手 (白, -1) -> 最大化
        max_eval = -float("inf")
        for move in moves:
            x, y = move
            if board[y][x] == 0: # 安全のためチェック
                board[y][x] = GOMOKU_AI_COLOR
                if check_win(board_obj, x, y, GOMOKU_AI_COLOR):
                    board[y][x] = 0; return 1000000, move
                eval, _ = minimax(board_obj, depth - 1, RENJU_AI_COLOR, alpha, beta)
                board[y][x] = 0
                if eval > max_eval: max_eval, best_move = eval, move
                alpha = max(alpha, eval)
                if beta <= alpha: break
        return max_eval, best_move
    else: # player == RENJU_AI_COLOR (先手, 黒, 1) -> 最小化
        min_eval = float("inf")
        for move in moves:
            x, y = move
            if board[y][x] == 0:
                board[y][x] = RENJU_AI_COLOR
                if check_win(board_obj, x, y, RENJU_AI_COLOR):
                    board[y][x] = 0; return -1000000, move
                eval, _ = minimax(board_obj, depth - 1, GOMOKU_AI_COLOR, alpha, beta)
                board[y][x] = 0
                if eval < min_eval: min_eval, best_move = eval, move
                beta = min(beta, eval)
                if beta <= alpha: break
        return min_eval, best_move