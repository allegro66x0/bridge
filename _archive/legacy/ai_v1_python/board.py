# board.py (タイポ修正 最終完成版)
import itertools

class Board:
    def __init__(self, size=15):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.winner = 0

    def place_stone(self, x, y, color):
        if not (0 <= x < self.size and 0 <= y < self.size and self.board[y][x] == 0):
            return False
        self.board[y][x] = color
        return True

    def remove_stone(self, x, y):
        self.board[y][x] = 0

    def generate_moves(self):
        moves = set()
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] != 0: continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] != 0:
                            moves.add((x, y))
                            break
                    else: continue
                    break
        return list(moves) if moves else [(self.size // 2, self.size // 2)]

    def is_forbidden_move(self, x, y):
        if self.board[y][x] != 0: return False
        self.board[y][x] = 1

        if self._get_line_length(x, y, 1) > 5:
            self.board[y][x] = 0; return True
        if self._count_patterns_at(x, y, 1, "FOUR") >= 2:
            self.board[y][x] = 0; return True
        if self._count_true_open_threes_at(x, y, 1) >= 2:
            self.board[y][x] = 0; return True
        
        self.board[y][x] = 0
        return False

    def _get_line_string(self, x, y, dx, dy, color):
        line = ""
        for i in range(-5, 6):
            # ▼▼▼【最終修正】タイポを修正▼▼▼
            nx = x + i * dx
            ny = y + i * dy
            # ▲▲▲【最終修正】▲▲▲
            if 0 <= nx < self.size and 0 <= ny < self.size:
                s = self.board[ny][nx]; line += "1" if s == color else "2" if s != 0 else "0"
            else: line += "3"
        return line
    
    def _count_true_open_threes_at(self, x, y, color):
        count = 0
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line = self._get_line_string(x, y, dx, dy, color)
            if "1111" in line: continue
            
            center_idx = 5
            patterns = ["01110", "010110", "011010"]
            for p in patterns:
                try:
                    idx = line.index(p)
                    if idx <= center_idx < idx + len(p):
                        count += 1; break
                except ValueError:
                    continue
        return count

    def _count_patterns_at(self, x, y, color, pattern_type):
        if pattern_type == "OPEN_THREE": return self._count_true_open_threes_at(x, y, color)
        count = 0
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line = self._get_line_string(x, y, dx, dy, color)
            if "11111" in line: continue
            patterns = ["1111", "11101", "11011", "10111"]
            for p in patterns:
                if p in line:
                    count += 1; break
        return count

    def _get_line_length(self, x, y, color):
        max_len = 0
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1
            for d in [1, -1]:
                for i in range(1, self.size):
                    nx, ny = x + i * dx * d, y + i * dy * d
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] == color:
                        count += 1
                    else: break
            max_len = max(max_len, count)
        return max_len

    def is_four_three(self, x, y, color):
        is_empty = self.board[y][x] == 0
        if is_empty: self.board[y][x] = color
        fours = self._count_patterns_at(x, y, color, "FOUR")
        threes = self._count_true_open_threes_at(x, y, color)
        if is_empty: self.board[y][x] = 0
        return fours >= 1 and threes >= 1

    def count_open_patterns(self, x, y, color, length):
        is_empty = self.board[y][x] == 0
        if is_empty: self.board[y][x] = color
        count = self._count_patterns_at(x, y, color, "OPEN_THREE" if length == 3 else "FOUR")
        if is_empty: self.board[y][x] = 0
        return count
    
    def count_open_threes(self, x, y, color):
        return self.count_open_patterns(x, y, color, 3)

    def check_win(self, x, y, color):
        length = self._get_line_length_readonly(x, y, color)
        if color == 1: return length == 5
        else: return length >= 5
            
    def _get_line_length_readonly(self, x, y, color):
        max_len = 0
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1
            for d in [1, -1]:
                for i in range(1, self.size):
                    nx, ny = x + i * dx * d, y + i * dy * d
                    if nx == x and ny == y: continue
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] == color:
                        count += 1
                    else: break
            max_len = max(max_len, count)
        return max_len