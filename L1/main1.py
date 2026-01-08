import cv2
import numpy as np
import random
import time
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import ctypes
import gomoku_board_recognition as gbr

# ===============================================================
# ★重要★ 高DPI設定 (Surfaceの解像度ズレを防止)
# ===============================================================
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# Surface Pro解像度
SCREEN_W = 2880
SCREEN_H = 1920

# --- レイアウト設定 (3段構成) ---
DISPLAY_SCALE = 2.0  

LIVE_W_DISP = int(640 * DISPLAY_SCALE)   
LIVE_H_DISP = int(480 * DISPLAY_SCALE)   
RESULT_W_DISP = int(560 * DISPLAY_SCALE) 
RESULT_H_DISP = int(560 * DISPLAY_SCALE) 

MSG_W = 2400
MSG_H = 300 
BTN_W = 2400
BTN_H = 300

content_total_w = LIVE_W_DISP + RESULT_W_DISP + 80 
start_x = (SCREEN_W - content_total_w) // 2
margin_top = 50

POS_MSG = ((SCREEN_W - MSG_W)//2, margin_top)
content_y = margin_top + MSG_H + 50
POS_LIVE = (start_x, content_y)
POS_RESULT = (start_x + LIVE_W_DISP + 80, content_y)
max_content_h = max(LIVE_H_DISP, RESULT_H_DISP)
btn_y = content_y + max_content_h + 50
POS_BTN = ((SCREEN_W - BTN_W)//2, btn_y)


# ===============================================================
# 設定・定数
# ===============================================================
BOARD_SIZE = 13
AI = 1      # 黒（先手）
PLAYER = 2  # 白（後手）
EMPTY = 0

CAM_INDEX = 0  # ★環境に合わせて変更してください
FIXED_CORNER_POINTS = [(171, 58), (512, 58), (508, 406), (161, 398)]
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"

# ===============================================================
# 盤面・評価ロジック
# ===============================================================
class Board:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.grid = [[0]*size for _ in range(size)]
    def play(self, r, c, player):
        if self.is_empty(r, c):
            self.grid[r][c] = player
            return True
        return False
    def copy(self):
        b = Board(self.size)
        b.grid = [row[:] for row in self.grid]
        return b
    def inside(self, r, c): return 0 <= r < self.size and 0 <= c < self.size
    def is_empty(self, r, c): return self.inside(r, c) and self.grid[r][c] == 0
    def check_win(self, x, y, player):
        if not self.inside(y, x): return False 
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if not (self.inside(ny, nx) and self.grid[ny][nx] == player): break
                count += 1
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if not (self.inside(ny, nx) and self.grid[ny][nx] == player): break
                count += 1
            if count >= 5: return True
        return False
    def get_empty_spots(self):
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r][c] == 0]

def gen_moves(board, radius=2):
    occupied = [(r,c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] != 0]
    if not occupied:
        center = board.size // 2
        return [(center, center)]
    cand = set()
    for (r0, c0) in occupied:
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                r, c = r0+dr, c0+dc
                if board.inside(r, c) and board.grid[r][c] == 0:
                    cand.add((r,c))
    return list(cand)

# --- 評価関数群 ---
def evaluate_line(line, player):
    opponent = PLAYER if player == AI else AI
    if opponent in line: return 0 
    player_stones = line.count(player)
    if player_stones == 5: return 100000
    if player_stones == 4: return 5000 
    if player_stones == 3: return 100
    if player_stones == 2: return 10
    return 0

def count_patterns_for_player(board_grid, player):
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 4):
            score += evaluate_line([board_grid[r][c+i] for i in range(5)], player)
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - 4):
            score += evaluate_line([board_grid[r+i][c] for i in range(5)], player)
    for r in range(BOARD_SIZE - 4):
        for c in range(BOARD_SIZE - 4):
            score += evaluate_line([board_grid[r+i][c+i] for i in range(5)], player)
    for r in range(4, BOARD_SIZE):
        for c in range(BOARD_SIZE - 4):
            score += evaluate_line([board_grid[r-i][c+i] for i in range(5)], player)
    return score

def get_score(board_grid, player):
    opponent = PLAYER if player == AI else AI
    return count_patterns_for_player(board_grid, player) - count_patterns_for_player(board_grid, opponent) * 0.8

# --- AI思考ロジック ---
def pick_move_reception_final(board, ai_player, turn_count):
    if random.random() < 0.03:
        empties = board.get_empty_spots()
        if empties:
            print("AI: (思考放棄) なんとなくここに置きます...")
            return random.choice(empties)

    candidates = gen_moves(board, radius=2)
    if not candidates: return None

    scored_moves = []
    for (r, c) in candidates:
        b_copy = board.copy()
        b_copy.play(r, c, ai_player)
        score = get_score(b_copy.grid, ai_player)
        scored_moves.append({'move': (r, c), 'score': score})
    
    scored_moves.sort(key=lambda x: x['score'], reverse=True)

    winning_threshold = 4000 
    miss_prob = 0.05 + (turn_count * 0.005)
    
    filtered_moves = []
    for item in scored_moves:
        if item['score'] >= winning_threshold:
            if random.random() < miss_prob:
                print(f"AI: 勝ち手{item['move']}を見逃しました (確率 {miss_prob*100:.1f}%)")
                continue 
        filtered_moves.append(item)
    
    if not filtered_moves:
        filtered_moves = scored_moves[-1:] 

    filtered_moves.sort(key=lambda x: x['score'], reverse=True)

    target_weights = [5, 10, 20, 30, 20, 15]
    
    num_candidates = len(filtered_moves)
    if num_candidates == 0:
        return random.choice(candidates)

    current_weights = target_weights[:num_candidates]
    moves_to_consider = filtered_moves[:len(current_weights)]
    
    chosen = random.choices(moves_to_consider, weights=current_weights, k=1)[0]
    return chosen['move']


# ===============================================================
# UI & 画面描画
# ===============================================================

try:
    font_msg = ImageFont.truetype(FONT_PATH, 60)
    font_btn = ImageFont.truetype(FONT_PATH, 50)
    font_warning = ImageFont.truetype(FONT_PATH, 80)
    font_gameover = ImageFont.truetype(FONT_PATH, 120)
except IOError:
    print(f"フォントファイルが見つかりません: {FONT_PATH}")
    exit()

# グローバル変数
background_frame, latest_board_ui, saved_intersections = None, None, None
confirmed_discs, intersection_map = [], {}
game_over, winner = False, None
recovery_mode = False
game_started = False 
ai_turn_count = 0 

ui_message_line1 = "「開始」ボタンで背景をセット"
ui_message_line2 = ""
ui_message_line3 = ""

last_clicked_command = None 
current_active_buttons = [] 

def update_buttons_layout():
    """下段ボタンパネルのレイアウト設定"""
    global current_active_buttons
    current_active_buttons = []
    
    main_btn = {}
    rect_main = (100, 50, 1600, 200) 
    
    if recovery_mode:
        main_btn = {"label": "修正完了(j)", "cmd": "j", "rect": rect_main, "color": (100, 100, 200)}
    elif game_over:
        main_btn = {"label": "もう一度(r)", "cmd": "r", "rect": rect_main, "color": (50, 200, 100)}
    elif not game_started:
        main_btn = {"label": "開始(s)", "cmd": "s", "rect": rect_main, "color": (50, 200, 100)}
    else:
        main_btn = {"label": "決定(n)", "cmd": "n", "rect": rect_main, "color": (200, 150, 50)}
        
    current_active_buttons.append(main_btn)
    
    quit_btn = {"label": "終了(q)", "cmd": "q", "rect": (1800, 50, 500, 200), "color": (80, 80, 80)}
    current_active_buttons.append(quit_btn)

def on_mouse_click_unified(event, x, y, flags, param):
    global last_clicked_command
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = POS_BTN
        if ix <= x <= ix + BTN_W and iy <= y <= iy + BTN_H:
            rel_x = x - ix
            rel_y = y - iy
            for btn in current_active_buttons:
                bx, by, bw, bh = btn["rect"]
                if bx <= rel_x <= bx + bw and by <= rel_y <= by + bh:
                    last_clicked_command = btn["cmd"]

def draw_japanese_text(image, text, position, font, color_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_button_panel():
    """下段ボタン描画 (黒背景・枠線なし)"""
    panel = np.full((BTN_H, BTN_W, 3), (0, 0, 0), dtype=np.uint8) 
    img_pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for btn in current_active_buttons:
        bx, by, bw, bh = btn["rect"]
        r, g, b = btn["color"]
        draw.rectangle((bx, by, bx+bw, by+bh), fill=(r, g, b), outline=None)
        
        text = btn["label"]
        bbox = font_btn.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = bx + (bw - tw) // 2
        ty = by + (bh - th) // 2
        draw.text((tx, ty), text, font=font_btn, fill=(255, 255, 255))
        
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ★修正ポイント: display_hand_warning を引数で受け取るように変更
def draw_message_panel(display_hand_warning):
    """上段メッセージ描画 (黒背景・白文字)"""
    panel = np.full((MSG_H, MSG_W, 3), (0, 0, 0), dtype=np.uint8)
    if display_hand_warning:
        message = "⚠ 手をどけてください"
        bbox = font_warning.getbbox(message)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (MSG_W - w) // 2
        y = (MSG_H - h) // 2
        panel = draw_japanese_text(panel, message, (x, y), font_warning, (0, 0, 255))
    elif recovery_mode:
        msg1 = "⚠ 今置いた石を一度どけて"
        msg2 = "下の「修正完了」ボタンを押してください"
        panel = draw_japanese_text(panel, msg1, (50, 40), font_msg, (0, 0, 255))
        panel = draw_japanese_text(panel, msg2, (50, 140), font_msg, (0, 0, 255))
    else:
        panel = draw_japanese_text(panel, ui_message_line1, (50, 30), font_msg, (255, 255, 0)) 
        if ui_message_line2:
            panel = draw_japanese_text(panel, ui_message_line2, (50, 110), font_msg, (255, 255, 255))
        if ui_message_line3:
            panel = draw_japanese_text(panel, ui_message_line3, (50, 190), font_msg, (255, 255, 255))
    return panel

def calculate_grid_points(points):
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
    margin = cell_size
    grid_size = cell_size * (board_size-1)
    img_size = grid_size + margin*2
    board_img = np.full((img_size, img_size, 3), (218, 179, 125), dtype=np.uint8)
    for i in range(board_size):
        pos = margin + i * cell_size
        cv2.line(board_img, (pos, margin), (pos, margin + grid_size), (0,0,0), 2)
        cv2.line(board_img, (margin, pos), (margin + grid_size, pos), (0,0,0), 2)
    return board_img

def convert_discs_to_board_obj(confirmed_discs):
    b = Board(BOARD_SIZE)
    for d in confirmed_discs:
        r, c = d.cell
        if b.inside(r, c):
            b.grid[r][c] = AI if d.color == gbr.DiscColor.BLACK else PLAYER
    return b

def create_unified_view(live_img, result_img, msg_img, btn_img):
    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    
    if msg_img is not None:
        h, w = msg_img.shape[:2]
        x, y = POS_MSG
        canvas[y:y+h, x:x+w] = msg_img
    
    if live_img is not None:
        live_scaled = cv2.resize(live_img, (LIVE_W_DISP, LIVE_H_DISP))
        h, w = live_scaled.shape[:2]
        x, y = POS_LIVE
        canvas[y:y+h, x:x+w] = live_scaled
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 2)
        cv2.putText(canvas, "Camera", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,200,200), 3)

    if result_img is not None:
        result_scaled = cv2.resize(result_img, (RESULT_W_DISP, RESULT_H_DISP))
        h, w = result_scaled.shape[:2]
        x, y = POS_RESULT
        canvas[y:y+h, x:x+w] = result_scaled
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 2)
        cv2.putText(canvas, "AI Board", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,200,200), 3)

    if btn_img is not None:
        h, w = btn_img.shape[:2]
        x, y = POS_BTN
        canvas[y:y+h, x:x+w] = btn_img

    return canvas


# ===============================================================
# メインループ
# ===============================================================
def main():
    global background_frame, latest_board_ui, saved_intersections, intersection_map
    global confirmed_discs, game_over, winner, recovery_mode
    global game_started, ai_turn_count
    global ui_message_line1, ui_message_line2, ui_message_line3
    global last_clicked_command

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): print(f"❌ カメラ({CAM_INDEX})起動失敗"); exit()

    saved_intersections, intersection_map = calculate_grid_points(FIXED_CORNER_POINTS)

    cv2.namedWindow("MainGame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("MainGame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("MainGame", on_mouse_click_unified)

    latest_board_ui = draw_board_ui(BOARD_SIZE, 40)

    while True:
        update_buttons_layout()

        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()
        
        key_input = cv2.waitKey(1) & 0xFF
        
        if last_clicked_command:
            command = last_clicked_command
            last_clicked_command = None
        elif key_input != 255:
            command = chr(key_input)
        else:
            command = None

        if command == 'q': break

        # ★ここで変数を定義
        display_hand_warning = False
        if background_frame is not None and not game_over and not recovery_mode:
            real_time_changes = [d for d in detect_discs_on_frame(background_frame, frame, intersection_map) if d.cell not in [cd.cell for cd in confirmed_discs]]
            if len(real_time_changes) > 2:
                display_hand_warning = True

        if saved_intersections:
            for point in saved_intersections:
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

        if command == 's' and not game_started:
            background_frame = frame.copy()
            latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
            ui_message_line1 = "背景を記憶。AI(黒)の初手を思考中です..."
            ui_message_line2 = "お待ちください..."
            ui_message_line3 = ""
            
            # ★引数を渡す
            msg_img = draw_message_panel(display_hand_warning)
            btn_img = draw_button_panel()
            combined = create_unified_view(display_frame, latest_board_ui, msg_img, btn_img)
            cv2.imshow("MainGame", combined)
            cv2.waitKey(1)

            current_board = convert_discs_to_board_obj(confirmed_discs)
            move = pick_move_reception_final(current_board, AI, ai_turn_count)
            
            if move:
                r, c = move
                ai_turn_count += 1
                print(f"{c:02d}{r:02d}")
                
                new_disc = gbr.Disc(); new_disc.color = gbr.DiscColor.BLACK; new_disc.cell = (r, c)
                confirmed_discs.append(new_disc)
                
                color = (10,10,10)
                cv2.circle(latest_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)
                overlay = latest_board_ui.copy()
                cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
                latest_board_ui = cv2.addWeighted(overlay, 0.5, latest_board_ui, 0.5, 0)
                
                ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
                ui_message_line2 = "あなたの番(白)です。"
                ui_message_line3 = "白石を置いて「決定」ボタンを押してください。"
                game_started = True 

        if command == 'n' and background_frame is not None and not game_over and game_started and not recovery_mode:
            newly_found_discs = [d for d in detect_discs_on_frame(background_frame, frame, intersection_map) if d.cell not in [cd.cell for cd in confirmed_discs]]
            
            if len(newly_found_discs) == 1:
                new_disc = newly_found_discs[0]
                black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
                white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
                
                is_player_turn = (black_count > white_count) 
                is_correct_color = (is_player_turn and new_disc.color == gbr.DiscColor.WHITE) 

                if is_correct_color:
                    confirmed_discs.append(new_disc)
                    y_idx, x_idx = new_disc.cell
                    
                    current_board = convert_discs_to_board_obj(confirmed_discs)
                    if current_board.check_win(x_idx, y_idx, PLAYER): 
                        game_over, winner = True, PLAYER
                    
                    board_ui_img = draw_board_ui(BOARD_SIZE, 40)
                    for d in confirmed_discs:
                        r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)

                    ai_next_move = None
                    current_black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
                    current_white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
                    
                    if not game_over and current_black_count == current_white_count:
                        ui_message_line1 = "AI(黒)が思考中です..."
                        ui_message_line2 = "しばらくお待ちください。"
                        ui_message_line3 = ""
                        
                        # ★引数を渡す
                        msg_img = draw_message_panel(display_hand_warning)
                        btn_img = draw_button_panel()
                        combined = create_unified_view(display_frame, board_ui_img, msg_img, btn_img)
                        cv2.imshow("MainGame", combined)
                        cv2.waitKey(1)

                        current_board = convert_discs_to_board_obj(confirmed_discs)
                        move = pick_move_reception_final(current_board, AI, ai_turn_count)
                        if move:
                             ai_next_move = move
                             ai_turn_count += 1
                    
                    if ai_next_move:
                        r, c = ai_next_move
                        print(f"{c:02d}{r:02d}")
                        new_ai_disc = gbr.Disc(); new_ai_disc.color = gbr.DiscColor.BLACK; new_ai_disc.cell = (r, c)
                        confirmed_discs.append(new_ai_disc)
                        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, (10,10,10), -1)

                        current_board = convert_discs_to_board_obj(confirmed_discs)
                        if current_board.check_win(c, r, AI): 
                            game_over, winner = True, AI
                        
                        overlay = board_ui_img.copy()
                        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
                        board_ui_img = cv2.addWeighted(overlay, 0.5, board_ui_img, 0.5, 0)
                        
                        if not game_over:
                            ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
                            ui_message_line2 = "あなたの番(白)です。"
                            ui_message_line3 = "白石を置いて「決定」ボタンを押してください。"
                    else: 
                        if not game_over and winner is None:
                           pass
                        elif not game_over:
                           ui_message_line1 = "あなたの番です(白)。"
                           ui_message_line2 = "石を置いて「決定」ボタンを押してください。"
                           ui_message_line3 = ""
                    
                    if game_over:
                        message = "あなたの勝ち！" if winner == PLAYER else "AIの勝ち！"
                        bbox = font_gameover.getbbox(message)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        x_pos = (board_ui_img.shape[1] - w) // 2
                        y_pos = (board_ui_img.shape[0] - h) // 2
                        board_ui_img = draw_japanese_text(board_ui_img, message, (x_pos, y_pos), font_gameover, (0, 0, 255))
                        ui_message_line1 = "勝負あり！"
                        ui_message_line2 = "もう一度遊ぶ場合は「もう一度」を、"
                        ui_message_line3 = "終了する場合は「終了」を押してください。"

                    latest_board_ui = board_ui_img
                    if not game_over: background_frame = frame.copy()
                else:
                    if is_player_turn:
                        ui_message_line1 = "エラー：白の番です。白石を置いてください。"
                    else:
                        ui_message_line1 = "エラー：AIの番です。"
                    ui_message_line2 = ""
                    ui_message_line3 = ""
            
            elif len(newly_found_discs) > 1:
                recovery_mode = True
                ui_message_line1 = "" 
                ui_message_line2 = ""
                ui_message_line3 = ""

        if command == 'j' and recovery_mode:
            background_frame = frame.copy()
            recovery_mode = False
            ui_message_line1 = "背景を再設定しました。"
            ui_message_line2 = "あなたの番(白)です。"
            ui_message_line3 = "石を置いて「決定」ボタンを押してください。"

        if command == 'r':
            background_frame, confirmed_discs = None, []
            latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
            game_over, recovery_mode = False, False
            winner = None
            game_started = False 
            ai_turn_count = 0 
            ui_message_line1 = "リセットしました。「開始」ボタンを押してください。"
            ui_message_line2 = ""
            ui_message_line3 = ""

        # ★引数を渡す
        msg_img = draw_message_panel(display_hand_warning)
        btn_img = draw_button_panel()
        combined_screen = create_unified_view(display_frame, latest_board_ui, msg_img, btn_img)
        cv2.imshow("MainGame", combined_screen)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()