# ai.py (最終完成版：「静止探索」搭載)
import math
import time
from evaluate import evaluate_board, get_move_value, PATTERNS

AI_COLOR = 1
PLAYER_COLOR = -1

class RenjuAI:
    def __init__(self, board, max_depth=4, max_time=5.0): # 基本深度を4に調整
        self.board = board
        self.max_depth = max_depth
        self.max_time = max_time
        self.start_time = None

    def _get_move_priority(self, move, color):
        x, y = move
        if color == AI_COLOR and self.board.is_forbidden_move(x, y): return -1
        return get_move_value(self.board, x, y, color)

    def get_best_move(self):
        self.start_time = time.time()
        if not any(any(row) for row in self.board.board):
            center = self.board.size // 2
            print(f"AI chose opening move: ({center}, {center})")
            return (center, center)

        legal_moves = self.board.generate_moves()
        
        # 攻撃価値と防御価値を総合的に評価してソート
        sorted_moves = sorted(legal_moves, 
                              key=lambda m: self._get_move_priority(m, AI_COLOR) + self._get_move_priority(m, PLAYER_COLOR) * 1.5, 
                              reverse=True)
        
        best_move = sorted_moves[0] if sorted_moves else None
        
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.max_time:
                print(f"Time limit reached. Using best move from depth {depth-1}.")
                break
            
            print(f"Searching at depth {depth}...")
            best_move_at_depth, best_score = self._search(depth, sorted_moves)
            if best_move_at_depth: best_move = best_move_at_depth
            
            if abs(best_score) >= PATTERNS['FIVE']:
                 print(f"Found decisive move at depth {depth}.")
                 break
        
        print(f"AI chose move: {best_move}")
        return best_move

    def _search(self, depth, moves):
        best_move = None; best_score = -math.inf
        for move in moves[:15]:
            if time.time() - self.start_time > self.max_time: return None, 0
            x, y = move
            self.board.place_stone(x, y, AI_COLOR)
            score = self.minimax(depth - 1, -math.inf, math.inf, False)
            self.board.remove_stone(x, y)
            if score > best_score:
                best_score = score; best_move = move
        print(f"Depth {depth}: Best move {best_move} with score {best_score}")
        return best_move, best_score

    def minimax(self, depth, alpha, beta, maximizing):
        if time.time() - self.start_time > self.max_time: return 0

        # ▼▼▼【最終兵器】静止探索（Quiescence Search）の実装▼▼▼
        # 通常の探索の末端で、危険な局面なら追加で読みを深める
        if depth == 0:
            return self.quiescence_search(alpha, beta, maximizing, depth=2) # 2手先まで危険をチェック
        # ▲▲▲【最終兵器】▲▲▲

        legal_moves = self.board.generate_moves()
        if not legal_moves: return evaluate_board(self.board, AI_COLOR)
        
        if maximizing:
            value = -math.inf
            for move in legal_moves[:7]:
                if self.board.is_forbidden_move(move[0], move[1]): continue
                self.board.place_stone(move[0], move[1], AI_COLOR)
                value = max(value, self.minimax(depth - 1, alpha, beta, False))
                self.board.remove_stone(move[0], move[1])
                alpha = max(alpha, value)
                if alpha >= beta: break
            return value
        else:
            value = math.inf
            for move in legal_moves[:7]:
                self.board.place_stone(move[0], move[1], PLAYER_COLOR)
                value = min(value, self.minimax(depth - 1, alpha, beta, True))
                self.board.remove_stone(move[0], move[1])
                beta = min(beta, value)
                if beta <= alpha: break
            return value

    def quiescence_search(self, alpha, beta, maximizing, depth):
        if time.time() - self.start_time > self.max_time: return 0
        
        stand_pat = evaluate_board(self.board, AI_COLOR)
        if depth == 0: return stand_pat

        if maximizing:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)
            
            # 危険な手（王手など）だけを生成
            moves = [m for m in self.board.generate_moves() if self._get_move_priority(m, AI_COLOR) > PATTERNS['THREE']]
            for move in moves:
                self.board.place_stone(move[0], move[1], AI_COLOR)
                score = self.quiescence_search(alpha, beta, False, depth - 1)
                self.board.remove_stone(move[0], move[1])
                alpha = max(alpha, score)
                if alpha >= beta: break
            return alpha
        else: # Minimizing
            if stand_pat <= alpha: return alpha
            beta = min(beta, stand_pat)

            moves = [m for m in self.board.generate_moves() if self._get_move_priority(m, PLAYER_COLOR) > PATTERNS['THREE']]
            for move in moves:
                self.board.place_stone(move[0], move[1], PLAYER_COLOR)
                score = self.quiescence_search(alpha, beta, True, depth - 1)
                self.board.remove_stone(move[0], move[1])
                beta = min(beta, score)
                if beta <= alpha: break
            return beta