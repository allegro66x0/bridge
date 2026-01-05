# battle_evaluate.py (元の evaluate.py と同一)

PATTERNS = {
    'FIVE': 10000000,
    'FOUR_THREE': 1000000,
    'OPEN_FOUR': 100000,
    'LEAP_FOUR': 90000,
    'FOUR': 5000,
    'OPEN_THREE': 4000,
    'THREAT_MULTIPLIER': 4,
    'THREE': 500,
    'TWO': 100,
}

def evaluate_board(board, color):
    # ... (この関数は変更なし) ...
    score = 0
    for y in range(board.size):
        for x in range(board.size):
            if board.board[y][x] == color:
                score += get_move_value(board, x, y, color, placed=True)
            elif board.board[y][x] != 0:
                score -= get_move_value(board, x, y, -color, placed=True)
    return score

def get_move_value(board, x, y, color, placed=False):
    # ... (この関数は変更なし) ...
    if not placed:
        if board.board[y][x] != 0: return 0
        board.board[y][x] = color

    score = 0
    if board.check_win(x, y, color):
        score = PATTERNS['FIVE']
    elif board.is_four_three(x, y, color):
        score = PATTERNS['FOUR_THREE']
    else:
        threat_count = 0
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line_score, is_threat = evaluate_line(board, x, y, dx, dy, color)
            score += line_score
            if is_threat: threat_count += 1
        
        if threat_count >= 2:
            score += PATTERNS['OPEN_THREE'] * (PATTERNS['THREAT_MULTIPLIER'] ** (threat_count - 1))

    if not placed:
        board.board[y][x] = 0
    return score

def evaluate_line(board, x, y, dx, dy, color):
    # ... (この関数は変更なし) ...
    line = "";
    for i in range(-5, 6):
        nx, ny = x + i*dx, y + i*dy
        if 0 <= nx < board.size and 0 <= ny < board.size:
            s = board.board[ny][nx]; line += '1' if s == color else '2' if s != 0 else '0'
        else: line += '3'

    is_threat = False
    score = 0
    
    if "011110" in line:
        score += PATTERNS['OPEN_FOUR']; is_threat = True
    
    leap_four_patterns = ["11101", "11011", "10111"]
    for p in leap_four_patterns:
        if p in line:
            score += PATTERNS['LEAP_FOUR']; is_threat = True
            break
            
    if "01110" in line:
        score += PATTERNS['OPEN_THREE']; is_threat = True
    
    if "211110" in line or "011112" in line: score += PATTERNS['FOUR']
    if "21110" in line or "01112" in line: score += PATTERNS['THREE']
    if "0110" in line: score += PATTERNS['TWO']
    
    return score, is_threat