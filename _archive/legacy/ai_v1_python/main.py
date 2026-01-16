import cv2
import numpy as np
import time
import gomoku_board_recognition as gbr  # 提出されたファイルをインポート
from PIL import Image, ImageDraw, ImageFont

# ===============================================================
# AI思考ロジック
# ===============================================================
BOARD_SIZE_AI = 13
# [MODIFIED] AI先手(黒=1), プレイヤー後手(白=2)
PLAYER, AI = 2, 1 
board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]

# (check_win, evaluate_line, count_patterns_for_player, evaluate, get_candidate_moves, minimax は変更なし)
def check_win(x, y, player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if not (0 <= nx < BOARD_SIZE_AI and 0 <= ny < BOARD_SIZE_AI and board_for_ai[ny][nx] == player): break
            count += 1
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if not (0 <= nx < BOARD_SIZE_AI and 0 <= ny < BOARD_SIZE_AI and board_for_ai[ny][nx] == player): break
            count += 1
        if count >= 5: return True
    return False

def evaluate_line(line, player):
    opponent = PLAYER if player == AI else AI
    if opponent in line: return 0
    player_stones = line.count(player)
    if player_stones == 5: return 100000
    if player_stones == 4: return 1000
    if player_stones == 3: return 100
    if player_stones == 2: return 10
    return 0

def count_patterns_for_player(player):
    score = 0
    for r in range(BOARD_SIZE_AI):
        for c in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board_for_ai[r][c+i] for i in range(5)], player)
    for c in range(BOARD_SIZE_AI):
        for r in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board_for_ai[r+i][c] for i in range(5)], player)
    for r in range(BOARD_SIZE_AI - 4):
        for c in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board_for_ai[r+i][c+i] for i in range(5)], player)
    for r in range(4, BOARD_SIZE_AI):
        for c in range(BOARD_SIZE_AI - 4):
            score += evaluate_line([board_for_ai[r-i][c+i] for i in range(5)], player)
    return score

def evaluate():
    return count_patterns_for_player(AI) - count_patterns_for_player(PLAYER) * 1.5

def get_candidate_moves():
    candidates = set()
    for y in range(BOARD_SIZE_AI):
        for x in range(BOARD_SIZE_AI):
            if board_for_ai[y][x] != 0:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < BOARD_SIZE_AI and 0 <= ny < BOARD_SIZE_AI and board_for_ai[ny][nx] == 0:
                            candidates.add((nx, ny))
    if not candidates: return {(BOARD_SIZE_AI // 2, BOARD_SIZE_AI // 2)}
    return candidates

def minimax(depth, player, alpha, beta):
    if depth == 0: return evaluate(), None
    moves = get_candidate_moves()
    best_move = next(iter(moves)) if moves else None
    if player == AI:
        max_eval = -float("inf")
        for move in moves:
            x, y = move
            board_for_ai[y][x] = AI
            if check_win(x, y, AI):
                board_for_ai[y][x] = 0; return 1000000, move
            eval, _ = minimax(depth - 1, PLAYER, alpha, beta)
            board_for_ai[y][x] = 0
            if eval > max_eval: max_eval, best_move = eval, move
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in moves:
            x, y = move
            board_for_ai[y][x] = PLAYER
            if check_win(x, y, PLAYER):
                board_for_ai[y][x] = 0; return -1000000, move
            eval, _ = minimax(depth - 1, AI, alpha, beta)
            board_for_ai[y][x] = 0
            if eval < min_eval: min_eval, best_move = eval, move
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval, best_move

def convert_discs_to_ai_board(confirmed_discs):
    # (gbr.Discオブジェクトに合わせて修正)
    global board_for_ai
    board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]
    for d in confirmed_discs:
        row, col = d.cell
        if 0 <= row < BOARD_SIZE_AI and 0 <= col < BOARD_SIZE_AI:
            # [MODIFIED] AI(黒=BLACK=0)が1, プレイヤー(白=WHITE=1)が2
            board_for_ai[row][col] = AI if d.color == gbr.DiscColor.BLACK else PLAYER

# ===============================================================
# メインプログラム (gbr使用・座標固定・AI先手版)
# ===============================================================
BOARD_SIZE = 13
CAM_INDEX = 2 # ご自身のカメラ番号

FIXED_CORNER_POINTS = [(237, 116), (578, 98), (580, 425), (260, 436)]

FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"
try:
    font_ui = ImageFont.truetype(FONT_PATH, 24)
    font_warning = ImageFont.truetype(FONT_PATH, 40)
    font_gameover = ImageFont.truetype(FONT_PATH, 80)
except IOError:
    print(f"フォントファイルが見つかりません: {FONT_PATH}")
    print("FONT_PATHをPCに存在する日本語フォントファイルのパスに修正してください。")
    exit()

# --- グローバル変数 ---
recognizer = gbr.AutomaticRecognizer(board_size=BOARD_SIZE)
background_frame, latest_board_ui = None, None
saved_intersections = None 
intersection_map = {}
confirmed_discs = [] # gbr.Disc オブジェクトを格納
game_over, winner = False, None
recovery_mode = False
game_started = False 

ui_message_line1 = "盤の座標は固定されました。"
ui_message_line2 = "盤面を空にして、「s」キーを押してください。"
ui_message_line3 = "(AI(黒)の初手が打たれます)"


def draw_japanese_text(image, text, position, font, color_bgr):
    """(変更なし) OpenCVの画像に日本語を描画する"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_grid_points(points):
    """(変更なし) 4点の座標から盤上の全ての交点座標を計算する"""
    src_pts = np.array(points, dtype=np.float32)
    side_length = (BOARD_SIZE - 1) * 40 # 仮想的なグリッドサイズ
    dst_pts = np.array([[0,0], [side_length,0], [side_length,side_length], [0,side_length]], dtype=np.float32)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    ideal_grid_points = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            ideal_grid_points.append([c * 40, r * 40])
    real_grid_points_np = cv2.perspectiveTransform(np.array([ideal_grid_points], dtype=np.float32), M_inv)
    final_intersections, final_intersection_map, idx = [], {}, 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            point = (int(real_grid_points_np[0][idx][0]), int(real_grid_points_np[0][idx][1]))
            final_intersections.append(point)
            final_intersection_map[point] = (r, c)
            idx += 1
    return final_intersections, final_intersection_map

def draw_board_ui(board_size, cell_size):
    """(変更なし) UI用の碁盤を描画する"""
    margin = cell_size; grid_size = cell_size * (board_size-1); img_size = grid_size + margin*2
    board_img = np.full((img_size, img_size, 3), (218, 179, 125), dtype=np.uint8)
    for i in range(board_size):
        pos = margin + i * cell_size
        cv2.line(board_img, (pos, margin), (pos, margin + grid_size), (0,0,0), 2)
        cv2.line(board_img, (margin, pos), (margin + grid_size, pos), (0,0,0), 2)
    return board_img

# --- メインループ ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened(): print(f"❌ カメラ({CAM_INDEX})起動失敗"); exit()

saved_intersections, intersection_map = calculate_grid_points(FIXED_CORNER_POINTS)

cv2.namedWindow("Live")
cv2.namedWindow("Result") 
cv2.namedWindow("Info") 

while True:
    ret, frame = cap.read()
    if not ret: break
    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    display_hand_warning = False
    if background_frame is not None and not game_over and not recovery_mode:
        ret_warn, result_warn, _ = recognizer.detectDisc(background_frame, frame, saved_intersections)
        if ret_warn:
            # ★★★ BUG FIX ★★★
            # 差分（result_warn.disc）から、確定済みの石（confirmed_discs）を除外して警告する
            confirmed_cells = [d.cell for d in confirmed_discs]
            real_time_changes = [d for d in result_warn.disc if d.cell not in confirmed_cells]
            if len(real_time_changes) > 2:
                display_hand_warning = True
            # ★★★★★★★★★★★★★

    if saved_intersections:
        for point in saved_intersections:
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

    # [MODIFIED] 's'キー (AIの初手実行)
    if key == ord('s') and not game_started:
        background_frame = frame.copy() # (1) 空の盤面が背景になる
        latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
        game_started = True
        
        ui_message_line1 = "背景を記憶。AI(黒)の初手を思考中です..."
        ui_message_line2 = ""
        ui_message_line3 = ""
        
        info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
        info_panel = draw_japanese_text(info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
        cv2.imshow("Info", info_panel)
        cv2.imshow("Result", latest_board_ui) 
        cv2.waitKey(1)

        convert_discs_to_ai_board(confirmed_discs)
        _, ai_next_move = minimax(2, AI, -float("inf"), float("inf")) 
        
        r, c = ai_next_move[1], ai_next_move[0] 
        print(f"AI move: {c:02d}{r:02d}")
        
        new_ai_disc = gbr.Disc()
        new_ai_disc.color = gbr.DiscColor.BLACK
        new_ai_disc.cell = (r, c)
        confirmed_discs.append(new_ai_disc) # (2) AIの石がリストに追加される
        
        convert_discs_to_ai_board(confirmed_discs)
        
        cv2.circle(latest_board_ui, (40 + c*40, 40 + r*40), 18, (10,10,10), -1) # 黒
        overlay = latest_board_ui.copy()
        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1) 
        latest_board_ui = cv2.addWeighted(overlay, 0.5, latest_board_ui, 0.5, 0)
        
        ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
        ui_message_line2 = "あなたの番(白)です。"
        ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
        
        # (3) 's'キーの最後では背景を更新しない (空のまま)

    # [MODIFIED] 'n'キー (プレイヤー(白)の手番)
    if key == ord('n') and background_frame is not None and not game_over and game_started and not recovery_mode:
        
        # (1) gbr ライブラリで背景(空の盤面)との差分をすべて検出
        ret_disc, result_disc, _ = recognizer.detectDisc(background_frame, frame, saved_intersections)
        
        # (2) ★★★ BUG FIX ★★★
        # 検出した石(result_disc.disc)から、すでに確定済みの石(confirmed_discs)を除く
        if ret_disc:
            confirmed_cells = [d.cell for d in confirmed_discs]
            newly_found_discs = [d for d in result_disc.disc if d.cell not in confirmed_cells]
        else:
            newly_found_discs = []
        # ★★★★★★★★★★★★★
            
        # (3) これで newly_found_discs は「新しく置かれた石」だけになる
        if len(newly_found_discs) == 1:
            new_disc = newly_found_discs[0]
            
            # (4) 手番と色のチェック (変更なし)
            black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
            white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
            is_player_turn = (black_count > white_count)
            is_correct_color = (is_player_turn and new_disc.color == gbr.DiscColor.WHITE)

            if is_correct_color:
                confirmed_discs.append(new_disc)
                y_idx, x_idx = new_disc.cell
                
                convert_discs_to_ai_board(confirmed_discs)
                if check_win(x_idx, y_idx, PLAYER): game_over, winner = True, PLAYER
                
                board_ui_img = draw_board_ui(BOARD_SIZE, 40)
                ai_next_move = None
                
                current_black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
                current_white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
                
                if not game_over and current_black_count == current_white_count:
                    ui_message_line1 = "AI(黒)が思考中です..."
                    ui_message_line2 = ""
                    ui_message_line3 = ""
                    
                    info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
                    info_panel = draw_japanese_text(info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
                    cv2.imshow("Info", info_panel)
                    cv2.waitKey(1)
                    
                    _, ai_next_move = minimax(2, AI, -float("inf"), float("inf")) 
                
                for d in confirmed_discs:
                    r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                    cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)
                
                if ai_next_move:
                    r, c = ai_next_move[1], ai_next_move[0]
                    print(f"AI move: {c:02d}{r:02d}")
                    
                    new_ai_disc = gbr.Disc()
                    new_ai_disc.color = gbr.DiscColor.BLACK
                    new_ai_disc.cell = (r, c)
                    confirmed_discs.append(new_ai_disc)
                    
                    cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, (10,10,10), -1) 

                    convert_discs_to_ai_board(confirmed_discs)
                    if check_win(c, r, AI): game_over, winner = True, AI

                    overlay = board_ui_img.copy()
                    cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
                    board_ui_img = cv2.addWeighted(overlay, 0.5, board_ui_img, 0.5, 0)
                    
                    if not game_over:
                        ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
                        ui_message_line2 = "あなたの番(白)です。"
                        ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
                
                if game_over:
                    message = "あなたの勝ち！" if winner == PLAYER else "AIの勝ち！"
                    bbox = font_gameover.getbbox(message)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x_pos = (board_ui_img.shape[1] - w) // 2
                    y_pos = (board_ui_img.shape[0] - h) // 2
                    board_ui_img = draw_japanese_text(board_ui_img, message, (x_pos, y_pos), font_gameover, (0, 0, 255))
                    ui_message_line1 = "'r'キーでリセット、'q'キーで終了"
                    ui_message_line2 = ""
                    ui_message_line3 = ""

                latest_board_ui = board_ui_img
                
                # (5) [重要] プレイヤーの石とAIの石が両方置かれた状態のフレームを、新しい背景として更新する
                if not game_over: background_frame = frame.copy()
            else:
                ui_message_line1 = "エラー：白の番です。白石を置いてください。"
                ui_message_line2 = ""
                ui_message_line3 = ""
        
        elif len(newly_found_discs) > 1:
            # (6) 修正により、この分岐には入らなくなるはず
            recovery_mode = True
            ui_message_line1 = f"エラー(修正後)：石を複数({len(newly_found_discs)})検知しました。"
            ui_message_line2 = "今置いた石を一度どけて、"
            ui_message_line3 = "「j」キーを押してください。"
        
        else: # len == 0
            ui_message_line1 = "エラー：石を認識できませんでした。"
            ui_message_line2 = "石を置き直して、再度「n」キーを押してください。"
            ui_message_line3 = ""

    # 'j'キー (リカバリーモード)
    if key == ord('j') and recovery_mode:
        background_frame = frame.copy()
        recovery_mode = False
        ui_message_line1 = "背景を再設定しました。"
        ui_message_line2 = "あなたの番(白)です。"
        ui_message_line3 = "石を置いて「n」キーを押してください。"

    # 'r'キー (リセット)
    if key == ord('r'):
        background_frame, latest_board_ui, confirmed_discs = None, None, []
        game_over, recovery_mode = False, False
        winner = None
        game_started = False
        board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]
        
        ui_message_line1 = "リセットしました。"
        ui_message_line2 = "盤面を空にして、「s」キーを押してください。"
        ui_message_line3 = "(AI(黒)の初手が打たれます)"

    # --- 描画処理 ---
    cv2.imshow("Live", display_frame)
    
    if latest_board_ui is not None:
        cv2.imshow("Result", latest_board_ui)

    info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
    color_bgr = (255, 0, 0) # 青
    font_to_use = font_ui

    if display_hand_warning:
        message_to_draw = "手をどけてください"
        font_to_use = font_warning
        color_bgr = (0, 0, 255) # 赤
        bbox = font_to_use.getbbox(message_to_draw)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x_pos = (info_panel.shape[1] - w) // 2
        y_pos = (info_panel.shape[0] - h) // 2
        info_panel = draw_japanese_text(info_panel, message_to_draw, (x_pos, y_pos), font_to_use, color_bgr)
    
    elif recovery_mode:
        msg_l1 = "今置いた石を一度どけて"
        msg_l2 = "「j」キーを押してください"
        font_to_use = font_warning # 40pt
        color_bgr = (0, 0, 255) # 赤
        
        bbox1 = font_to_use.getbbox(msg_l1)
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        x_pos1 = (info_panel.shape[1] - w1) // 2
        y_pos1 = (info_panel.shape[0] - h1*2 - 10) // 2 
        
        bbox2 = font_to_use.getbbox(msg_l2)
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox1[1]
        x_pos2 = (info_panel.shape[1] - w2) // 2
        y_pos2 = y_pos1 + h1 + 10 

        info_panel = draw_japanese_text(info_panel, msg_l1, (x_pos1, y_pos1), font_to_use, color_bgr)
        info_panel = draw_japanese_text(info_panel, msg_l2, (x_pos2, y_pos2), font_to_use, color_bgr)

    else:
        # 通常/ゲームオーバーのメッセージ
        info_panel = draw_japanese_text(info_panel, ui_message_line1, (15, 10), font_ui, color_bgr)
        info_panel = draw_japanese_text(info_panel, ui_message_line2, (15, 45), font_ui, color_bgr) 
        info_panel = draw_japanese_text(info_panel, ui_message_line3, (15, 80), font_ui, color_bgr)

    cv2.imshow("Info", info_panel)

cap.release()
cv2.destroyAllWindows()