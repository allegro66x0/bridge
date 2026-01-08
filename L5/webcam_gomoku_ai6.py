import cv2
import numpy as np
import time
import gomoku_board_recognition as gbr
from PIL import Image, ImageDraw, ImageFont

# ===============================================================
# AI思考ロジック (変更なし)
# ===============================================================
BOARD_SIZE_AI = 13
# [MODIFIED] AI先手(黒=1), プレイヤー後手(白=2)
PLAYER, AI = 2, 1 
board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]

def check_win(x, y, player):
    """(x, y)に置かれた石によって、playerが勝利したかどうかを判定する"""
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

# (evaluate_line, count_patterns_for_player, evaluate, get_candidate_moves, minimax は変更なし)
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
    global board_for_ai
    board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]
    for d in confirmed_discs:
        row, col = d.cell
        if 0 <= row < BOARD_SIZE_AI and 0 <= col < BOARD_SIZE_AI:
            # [MODIFIED] AI(黒), プレイヤー(白)
            board_for_ai[row][col] = AI if d.color == gbr.DiscColor.BLACK else PLAYER

# ===============================================================
# メインプログラム (UI分離・AI先手版)
# ===============================================================
BOARD_SIZE = 13
CAM_INDEX = 0
FIXED_CORNER_POINTS = [(171, 58), (512, 58), (508, 406), (161, 398)]
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
background_frame, latest_board_ui, saved_intersections = None, None, None
confirmed_discs, intersection_map = [], {}
game_over, winner = False, None
recovery_mode = False
game_started = False 

ui_message_line1 = "'s'キーで背景をセットしてください"
ui_message_line2 = ""
ui_message_line3 = ""


def draw_japanese_text(image, text, position, font, color_bgr):
    """OpenCVの画像(NumPy配列)に日本語を描画する"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)
    # [FIX] 前回のバグ修正 (RGB_BGR -> RGB2BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# (calculate_grid_points, detect_discs_on_frame, draw_board_ui は変更なし)
def calculate_grid_points(points):
    """4点の座標から盤上の全ての交点座標を計算する (変更なし)"""
    src_pts = np.array(points, dtype=np.float32)
    side_length = (BOARD_SIZE - 1) * 40
    dst_pts = np.array([[0,0], [side_length,0], [side_length,side_length], [0,side_length]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
    ideal_grid_points = [[c * 40, r * 40] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
    real_grid_points_np = cv2.perspectiveTransform(np.array([ideal_grid_points], dtype=np.float32), M)
    final_intersections, final_intersection_map, idx = [], {}, 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            point = (int(real_grid_points_np[0][idx][0]), int(real_grid_points_np[0][idx][1]))
            final_intersections.append(point); final_intersection_map[point] = (r, c); idx += 1
    return final_intersections, final_intersection_map

def detect_discs_on_frame(bg_frame, curr_frame, intersection_map):
    """フレームの差分から石を検出する (変更なし)"""
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(bg_gray, curr_gray)
    newly_found, ROI_SIZE, CHANGE_THRESHOLD = [], 12, 35
    for point, cell_coord in intersection_map.items():
        x, y = point
        x_s, x_e, y_s, y_e = x-ROI_SIZE//2, x+ROI_SIZE//2, y-ROI_SIZE//2, y+ROI_SIZE//2
        if 0 <= y_s and y_e < bg_frame.shape[0] and 0 <= x_s and x_e < bg_frame.shape[1]:
            if np.mean(diff_frame[y_s:y_e, x_s:x_e]) > CHANGE_THRESHOLD:
                disc = gbr.Disc(); disc.color = gbr.DiscColor.WHITE if np.mean(curr_gray[y_s:y_e, x_s:x_e]) > 128 else gbr.DiscColor.BLACK
                disc.cell = cell_coord; newly_found.append(disc)
    return newly_found

def draw_board_ui(board_size, cell_size):
    """UI用の碁盤を描画する (変更なし)"""
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
cv2.namedWindow("Info") 

while True:
    ret, frame = cap.read()
    if not ret: break
    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    display_hand_warning = False
    if background_frame is not None and not game_over and not recovery_mode:
        real_time_changes = [d for d in detect_discs_on_frame(background_frame, frame, intersection_map) if d.cell not in [cd.cell for cd in confirmed_discs]]
        if len(real_time_changes) > 2:
            display_hand_warning = True

    if saved_intersections:
        for point in saved_intersections:
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

    # [MODIFIED] 's'キー (AIの初手実行)
    if key == ord('s') and not game_started:
        background_frame = frame.copy()
        latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
        
        ui_message_line1 = "背景を記憶。AI(黒)の初手を思考中です..."
        ui_message_line2 = ""
        ui_message_line3 = ""
        
        # Infoウィンドウに「思考中」を表示
        temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
        temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
        cv2.imshow("Info", temp_info_panel)
        cv2.imshow("Result", latest_board_ui) 
        cv2.waitKey(1)

        # AIの初手を計算 (空のボードを渡す)
        convert_discs_to_ai_board(confirmed_discs)
        _, ai_next_move = minimax(2, AI, -float("inf"), float("inf")) # AI(黒)の手番
        
        r, c = ai_next_move[1], ai_next_move[0] 
        print(f"{c:02d}{r:02d}")
        
        new_disc = gbr.Disc(); new_disc.color = gbr.DiscColor.BLACK; new_disc.cell = (r, c)
        confirmed_discs.append(new_disc)
        
        # AIの内部ボードも更新
        convert_discs_to_ai_board(confirmed_discs)
        
        # UIに描画
        color = (10,10,10) # 黒
        cv2.circle(latest_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)
        overlay = latest_board_ui.copy()
        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1) # ハイライト
        latest_board_ui = cv2.addWeighted(overlay, 0.5, latest_board_ui, 0.5, 0)
        
        ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
        ui_message_line2 = "あなたの番(白)です。"
        ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
        game_started = True 

    # 'n'キー (プレイヤーの手番)
    if key == ord('n') and background_frame is not None and not game_over and game_started and not recovery_mode:
        newly_found_discs = [d for d in detect_discs_on_frame(background_frame, frame, intersection_map) if d.cell not in [cd.cell for cd in confirmed_discs]]
        
        if len(newly_found_discs) == 1:
            new_disc = newly_found_discs[0]
            black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
            white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
            
            # [MODIFIED] プレイヤー(白)の手番は black > white の時
            is_player_turn = (black_count > white_count) 
            # [MODIFIED] プレイヤー(白)は白石を置かなければならない
            is_correct_color = (is_player_turn and new_disc.color == gbr.DiscColor.WHITE) 

            if is_correct_color:
                confirmed_discs.append(new_disc)
                y_idx, x_idx = new_disc.cell
                last_player = AI if new_disc.color == gbr.DiscColor.BLACK else PLAYER
                
                convert_discs_to_ai_board(confirmed_discs)
                # プレイヤー(白)の勝利判定
                if check_win(x_idx, y_idx, last_player): game_over, winner = True, last_player
                
                board_ui_img = draw_board_ui(BOARD_SIZE, 40)
                ai_next_move = None
                current_black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
                current_white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
                
                # [MODIFIED] AI(黒)の思考は black == white の時
                if not game_over and current_black_count == current_white_count:
                    ui_message_line1 = "AI(黒)が思考中です..."
                    ui_message_line2 = ""
                    ui_message_line3 = ""
                    
                    # 盤面を先に描画
                    temp_board_ui = draw_board_ui(BOARD_SIZE, 40)
                    for d in confirmed_discs: 
                        r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                        cv2.circle(temp_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)

                    # Infoウィンドウも思考中に更新
                    temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
                    temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
                    cv2.imshow("Info", temp_info_panel)
                    cv2.imshow("Result", temp_board_ui) 
                    cv2.waitKey(1)

                    _, ai_next_move = minimax(2, AI, -float("inf"), float("inf")) # AI(黒)の手番
                
                # 盤面の再描画 (プレイヤーの石 + AIの石)
                for d in confirmed_discs:
                    r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                    cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)
                
                if ai_next_move:
                    r, c = ai_next_move[1], ai_next_move[0]
                    print(f"{c:02d}{r:02d}")
                    
                    # [MODIFIED] AI(黒)の石を置く
                    new_ai_disc = gbr.Disc(); new_ai_disc.color = gbr.DiscColor.BLACK; new_ai_disc.cell = (r, c)
                    confirmed_discs.append(new_ai_disc)
                    cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, (10,10,10), -1) # 黒色

                    # AI(黒)の着手後にも勝利判定
                    convert_discs_to_ai_board(confirmed_discs)
                    if check_win(c, r, AI): game_over, winner = True, AI
                    
                    overlay = board_ui_img.copy()
                    cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
                    board_ui_img = cv2.addWeighted(overlay, 0.5, board_ui_img, 0.5, 0)
                    
                    if not game_over:
                        # [MODIFIED] メッセージを修正
                        ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
                        ui_message_line2 = "あなたの番(白)です。"
                        ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
                else: 
                    # プレイヤーが勝った場合
                    if not game_over:
                       ui_message_line1 = "あなたの番です(白)。"
                       ui_message_line2 = "好きな場所に白石を置き、「n」キーを押してください。"
                       ui_message_line3 = ""
                
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
                if not game_over: background_frame = frame.copy()
            else:
                # [MODIFIED] エラーメッセージを修正
                if is_player_turn:
                    ui_message_line1 = "エラー：白の番です。白石を置いてください。"
                    ui_message_line2 = ""
                    ui_message_line3 = ""
                else:
                    # この分岐は 'is_player_turn' が False の場合 (＝AIの番のはず)
                    ui_message_line1 = "エラー：AIの番です。"
                    ui_message_line2 = ""
                    ui_message_line3 = ""
        
        elif len(newly_found_discs) > 1:
            recovery_mode = True
            ui_message_line1 = "" 
            ui_message_line2 = ""
            ui_message_line3 = ""

    # 'j'キー (リカバリーモード)
    if key == ord('j') and recovery_mode:
        background_frame = frame.copy()
        recovery_mode = False
        # [MODIFIED] メッセージを修正
        ui_message_line1 = "背景を再設定しました。"
        ui_message_line2 = "あなたの番(白)です。石を置いて「n」キーを押してください。"
        ui_message_line3 = ""

    # 'r'キー (リセット)
    if key == ord('r'):
        background_frame, latest_board_ui, confirmed_discs = None, None, []
        game_over, recovery_mode = False, False
        winner = None
        game_started = False 
        ui_message_line1 = "リセットしました。's'キーで背景をセットしてください。"
        ui_message_line2 = ""
        ui_message_line3 = ""

    cv2.imshow("Live", display_frame)
    
    # 1. 盤面ウィンドウ ("Result") の表示
    if latest_board_ui is not None:
        cv2.imshow("Result", latest_board_ui)

    # 2. 情報ウィンドウ ("Info") の作成と表示
    info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
    color_bgr = (255, 0, 0) # 青

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
        # [MODIFIED] リカバリーメッセージを修正 ('j'キー)
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