import cv2
import numpy as np 
import time
import threading
import gomoku_board_recognition as gbr
from PIL import Image, ImageDraw, ImageFont
import copy 
import multiprocessing # C++スレッドのハング防止に必須
import sys
import os
import ctypes
import serial

# --- Import Central Config ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    import config
except ImportError:
    print("❌ Config not found. Using defaults.")
    class config:
        SERIAL_PORT_GANTRY = 'COM9'
        SERIAL_PORT_SORTER = None
        CAMERA_INDEX = 0
        BOARD_SIZE_AI = 13
        SEARCH_DEPTH = 5
        FIXED_CORNER_POINTS = [(306, 216), (394, 213), (408, 326), (305, 314)]

# --- Add src path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from hardware.coin_sorter import CoinSorter
from hardware.magnet_control import MagnetControl

# ===============================================================
# Robot Control Settings
# ===============================================================
SERIAL_PORT_GANTRY = config.SERIAL_PORT_GANTRY 
SERIAL_PORT_MAGNET = config.SERIAL_PORT_MAGNET
SERIAL_PORT_SORTER = config.SERIAL_PORT_SORTER
BAUD_RATE = 9600

ser_gantry = None
magnet = None
sorter = None

# Pickup ROI
PICKUP_ROI = (50, 50, 60, 60) 

def setup_robot_connection():
    global ser_gantry, magnet
    # Gantry
    try:
        ser_gantry = serial.Serial(SERIAL_PORT_GANTRY, BAUD_RATE, timeout=1)
        print(f"✅ Gantry Connected: {SERIAL_PORT_GANTRY}")
    except Exception as e:
        print(f"❌ Gantry Connection Failed: {e}")
    
    # Magnet
    magnet = MagnetControl(SERIAL_PORT_MAGNET, BAUD_RATE)
    if magnet.connect():
        pass
    else:
        print("⚠️ Magnet Not Connected (Simulation Mode?)")

    # Sorter (Disabled)
    print("ℹ️ Coin Sorter: Disabled")

def _send_gantry_cmd_wait(cmd):
    """Helper to send command and wait for READY"""
    if not ser_gantry or not ser_gantry.is_open:
        return
    
    print(f"🤖 Gantry CMD: {cmd.strip()}")
    ser_gantry.write(cmd.encode('utf-8'))
    
    # Blocking wait for 'READY'
    while True:
        if ser_gantry.in_waiting > 0:
            line = ser_gantry.readline().decode('utf-8', errors='ignore').strip()
            if line == "READY":
                break
            # print(f"  [Gantry]: {line}") # Debug output

def send_home_command():
    """Send H:ALL command to return Gantry to home"""
    if ser_gantry and ser_gantry.is_open:
        _send_gantry_cmd_wait("H:ALL\n")

def execute_pick_and_place(x, y):
    """
    Coordinator Function:
    1. Move to Supply
    2. Magnet ON -> Down -> Up (Pick)
    3. Move to Target(x,y)
    4. Down -> Magnet OFF -> Up (Place)
    5. Return Home
    """
    if not ser_gantry or not ser_gantry.is_open:
        print("❌ Gantry not ready")
        return

    gx = x + 1
    gy = y + 1
    
    print(f"\n--- Sequence Start: Grid ({gx}, {gy}) ---")

    # 1. Move to Supply
    _send_gantry_cmd_wait("M:SPLY\n")

    # 2. Pick Sequence
    if magnet: magnet.on()
    time.sleep(0.5) # Wait for magnetization
    
    _send_gantry_cmd_wait("Z:PICK\n") # Down
    _send_gantry_cmd_wait("Z:HOME\n") # Up

    # 3. Move to Target
    cmd_move = f"M:{gx:02}{gy:02}\n"
    _send_gantry_cmd_wait(cmd_move)

    # 4. Place Sequence
    _send_gantry_cmd_wait("Z:PICK\n") # Down
    
    if magnet: magnet.off()
    time.sleep(0.5) # Wait for drop
    
    _send_gantry_cmd_wait("Z:HOME\n") # Up

    # 5. Return Home
    _send_gantry_cmd_wait("M:ZERO\n")

    print("--- Sequence Complete ---\n")

def check_pickup_point_and_feed(frame):
    """Monitor pickup point and control sorter"""
    # Sorter Disabled
    return

    if not sorter.is_connected: return

    x, y, w, h = PICKUP_ROI
    if y+h >= frame.shape[0] or x+w >= frame.shape[1]: return

    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    
    # Thresholds (Adjust as needed)
    has_stone = (mean_val < 80) or (mean_val > 180)
    
    if not has_stone:
        sorter.start_all()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        sorter.stop_all()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


# ===============================================================
# C++モジュールの読み込み
# ===============================================================
# Strict import removed to allow fallback logic below

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
# 設定・ロジック
# ===============================================================
BOARD_SIZE_AI = config.BOARD_SIZE_AI
PLAYER, AI = 2, 1 # AI先手 
SEARCH_DEPTH = config.SEARCH_DEPTH 

CAM_INDEX = config.CAMERA_INDEX
FIXED_CORNER_POINTS = config.FIXED_CORNER_POINTS
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"

def check_win(board, x, y, player):
    """勝利判定 (Python側で実施)"""
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
        if count >= 5: return True
    return False

# ===============================================================
# C++モジュールの読み込み (失敗時はPython版AIを使用)
# ===============================================================
ENABLE_CPP = False
try:
    import cpp_gomoku_ai
    ENABLE_CPP = True
    print("✅ C++ AIモジュール読み込み成功")
except ImportError:
    print("⚠️ C++ AIモジュールが見つかりません。Python版(低速)を使用します。")
    print("   (Visual C++ Build Toolsをインストールして再構築すると高速化できます)")

# --- Python版 Minimax AI (Fallback) ---
def evaluate_line(line, player, ai_player):
    opponent = player if ai_player != player else (3 - player) # Assuming 1 and 2
    if opponent in line: return 0
    player_stones = line.count(player)
    if player_stones == 5: return 100000
    if player_stones == 4: return 1000
    if player_stones == 3: return 100
    if player_stones == 2: return 10
    return 0

def count_patterns(board, player):
    score = 0
    bs = len(board)
    # Horizontal
    for r in range(bs):
        for c in range(bs - 4):
            score += evaluate_line([board[r][c+i] for i in range(5)], player, player)
    # Vertical
    for c in range(bs):
        for r in range(bs - 4):
             score += evaluate_line([board[r+i][c] for i in range(5)], player, player)
    # Diagonal \
    for r in range(bs - 4):
        for c in range(bs - 4):
            score += evaluate_line([board[r+i][c+i] for i in range(5)], player, player)
    # Diagonal /
    for r in range(4, bs):
        for c in range(bs - 4):
            score += evaluate_line([board[r-i][c+i] for i in range(5)], player, player)
    return score

def evaluate_board(board, ai_player, human_player):
    return count_patterns(board, ai_player) - count_patterns(board, human_player) * 1.5

def get_moves(board):
    candidates = set()
    bs = len(board)
    for y in range(bs):
        for x in range(bs):
            if board[y][x] != 0:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < bs and 0 <= ny < bs and board[ny][nx] == 0:
                            candidates.add((nx, ny))
    if not candidates: return [(bs//2, bs//2)]
    return list(candidates)

def minimax_py(board, depth, player, ai_player, human_player, alpha, beta):
    if depth == 0: return evaluate_board(board, ai_player, human_player), None
    moves = get_moves(board)
    best_move = moves[0] if moves else None
    
    if player == ai_player:
        max_eval = -float("inf")
        for x, y in moves:
            board[y][x] = ai_player
            if check_win(board, x, y, ai_player): 
                board[y][x] = 0
                return 1000000, (x, y)
            eval_val, _ = minimax_py(board, depth - 1, human_player, ai_player, human_player, alpha, beta)
            board[y][x] = 0
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = (x, y)
            alpha = max(alpha, eval_val)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for x, y in moves:
            board[y][x] = human_player
            if check_win(board, x, y, human_player): 
                 board[y][x] = 0
                 return -1000000, (x, y)
            eval_val, _ = minimax_py(board, depth - 1, ai_player, ai_player, human_player, alpha, beta)
            board[y][x] = 0
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = (x, y)
            beta = min(beta, eval_val)
            if beta <= alpha: break
        return min_eval, best_move

def find_best_move_parallel(board_np, depth):
    """C++エンジン または Python Fallback を使用"""
    # 2=Player, 1=AI (Assuming global constants)
    AI_ID = 1
    PLAYER_ID = 2
    
    start_time = time.time()
    
    if ENABLE_CPP:
        # C++に渡すためにint32型に変換
        board_for_cpp = board_np.astype(np.int32)
        move_tuple = cpp_gomoku_ai.find_best_move(board_for_cpp, depth)
        best_x, best_y = move_tuple[0], move_tuple[1]
        print(f"[C++ AI] Time: {time.time() - start_time:.4f}s (Depth: {depth})")
        return 99999, (best_x, best_y)
    else:
        # Python Fallback (Depth reduced for speed)
        py_depth = 2 # Force shallow search for speed
        board_list = board_np.tolist()
        _, move = minimax_py(board_list, py_depth, AI_ID, AI_ID, PLAYER_ID, -float('inf'), float('inf'))
        print(f"[Python AI] Time: {time.time() - start_time:.4f}s (Depth: {py_depth}) - C++Unavailable")
        return 99999, move

def convert_discs_to_ai_board(confirmed_discs):
    new_board = np.zeros((BOARD_SIZE_AI, BOARD_SIZE_AI), dtype=np.int32) 
    for d in confirmed_discs:
        row, col = d.cell
        if 0 <= row < BOARD_SIZE_AI and 0 <= col < BOARD_SIZE_AI:
            new_board[row][col] = AI if d.color == gbr.DiscColor.BLACK else PLAYER
    return new_board


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
    """マウスクリックイベント"""
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
    side_length = (BOARD_SIZE_AI - 1) * 40
    dst_pts = np.array([[0,0], [side_length,0], [side_length,side_length], [0,side_length]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
    ideal_grid_points = [[c * 40, r * 40] for r in range(BOARD_SIZE_AI) for c in range(BOARD_SIZE_AI)]
    real_grid_points_np = cv2.perspectiveTransform(np.array([ideal_grid_points], dtype=np.float32), M)
    final_intersections, final_intersection_map, idx = [], {}, 0
    for r in range(BOARD_SIZE_AI):
        for c in range(BOARD_SIZE_AI):
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
    margin = cell_size; grid_size = cell_size * (board_size-1); img_size = grid_size + margin*2
    board_img = np.full((img_size, img_size, 3), (218, 179, 125), dtype=np.uint8)
    for i in range(board_size):
        pos = margin + i * cell_size
        cv2.line(board_img, (pos, margin), (pos, margin + grid_size), (0,0,0), 2)
        cv2.line(board_img, (margin, pos), (margin + grid_size, pos), (0,0,0), 2)
    return board_img

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
def main_loop():
    global background_frame, latest_board_ui, saved_intersections, intersection_map
    global confirmed_discs, game_over, winner, recovery_mode
    global game_started
    global ui_message_line1, ui_message_line2, ui_message_line3
    global last_clicked_command

    setup_robot_connection()  # <--- Connect to Robots

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): print(f"❌ カメラ({CAM_INDEX})起動失敗"); exit()

    saved_intersections, intersection_map = calculate_grid_points(FIXED_CORNER_POINTS)

    cv2.namedWindow("MainGame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("MainGame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("MainGame", on_mouse_click_unified)

    latest_board_ui = draw_board_ui(BOARD_SIZE_AI, 40)

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

        display_hand_warning = False
        if background_frame is not None and not game_over and not recovery_mode:
            real_time_changes = [d for d in detect_discs_on_frame(background_frame, frame, intersection_map) if d.cell not in [cd.cell for cd in confirmed_discs]]
            if len(real_time_changes) > 2:
                display_hand_warning = True

        if saved_intersections:
            for point in saved_intersections:
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

        # ★ Sorter Logic
        check_pickup_point_and_feed(display_frame)

        # 's' (Start)
        if command == 's' and not game_started:
            ui_message_line1 = "ガントリーを退避中..."
            # 画面更新してメッセージを表示させるためのウェイト
            msg_img = draw_message_panel(display_hand_warning)
            combined = create_unified_view(display_frame, latest_board_ui, msg_img, None)
            cv2.imshow("MainGame", combined)
            cv2.waitKey(1)

            # ガントリー退避 (カメラ視野確保)
            send_home_command()
            
            # 退避後のフレームを少し待ってから取得 (振動収まり待ち)
            time.sleep(0.5)
            ret, frame = cap.read() # 最新フレームを取得し直す
            background_frame = frame.copy()
            latest_board_ui = draw_board_ui(BOARD_SIZE_AI, 40)
            ui_message_line1 = "背景を記憶。AI(黒)の初手を思考中です..."
            ui_message_line2 = "お待ちください..."
            ui_message_line3 = ""
            
            msg_img = draw_message_panel(display_hand_warning)
            btn_img = draw_button_panel()
            combined = create_unified_view(display_frame, latest_board_ui, msg_img, btn_img)
            cv2.imshow("MainGame", combined)
            cv2.waitKey(1)

            # C++ AIの初手計算
            current_ai_board = convert_discs_to_ai_board(confirmed_discs)
            _, ai_next_move = find_best_move_parallel(current_ai_board, SEARCH_DEPTH)
            
            r, c = ai_next_move[1], ai_next_move[0] 
            print(f"{c:02d}{r:02d}")
            
            # ★ Send to Gantry (Async)
            t = threading.Thread(target=execute_pick_and_place, args=(c, r))
            t.start()

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

        # 'n' (Next)
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
                    last_player = AI if new_disc.color == gbr.DiscColor.BLACK else PLAYER
                    
                    current_ai_board = convert_discs_to_ai_board(confirmed_discs)
                    if check_win(current_ai_board, x_idx, y_idx, last_player): game_over, winner = True, last_player
                    
                    board_ui_img = draw_board_ui(BOARD_SIZE_AI, 40)
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
                        
                        msg_img = draw_message_panel(display_hand_warning)
                        btn_img = draw_button_panel()
                        combined = create_unified_view(display_frame, board_ui_img, msg_img, btn_img)
                        cv2.imshow("MainGame", combined)
                        cv2.waitKey(1)

                        # C++ AIの思考
                        current_ai_board = convert_discs_to_ai_board(confirmed_discs)
                        _, ai_next_move = find_best_move_parallel(current_ai_board, SEARCH_DEPTH)
                    
                    if ai_next_move:
                        r, c = ai_next_move[1], ai_next_move[0]
                        print(f"{c:02d}{r:02d}")
                        
                        # ★ Send to Gantry (Async)
                        t = threading.Thread(target=execute_pick_and_place, args=(c, r))
                        t.start()

                        new_ai_disc = gbr.Disc(); new_ai_disc.color = gbr.DiscColor.BLACK; new_ai_disc.cell = (r, c)
                        confirmed_discs.append(new_ai_disc)
                        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, (10,10,10), -1)

                        current_ai_board = convert_discs_to_ai_board(confirmed_discs)
                        if check_win(current_ai_board, c, r, AI): game_over, winner = True, AI
                        
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
                        ui_message_line1 = "エラー：白の番です。"
                        ui_message_line2 = "白石を置いてください。"
                    else:
                        ui_message_line1 = "エラー：AIの番です。"
                    ui_message_line3 = ""
            
            elif len(newly_found_discs) > 1:
                recovery_mode = True
                ui_message_line1 = "" 
                ui_message_line2 = ""
                ui_message_line3 = ""

        # 'j' (Recovery)
        if command == 'j' and recovery_mode:
            background_frame = frame.copy()
            recovery_mode = False
            ui_message_line1 = "背景を再設定しました。"
            ui_message_line2 = "あなたの番(白)です。"
            ui_message_line3 = "石を置いて「決定」ボタンを押してください。"

        # 'r' (Reset)
        if command == 'r':
            background_frame, confirmed_discs = None, []
            latest_board_ui = draw_board_ui(BOARD_SIZE_AI, 40)
            game_over, recovery_mode = False, False
            winner = None
            game_started = False 
            ui_message_line1 = "リセットしました。「開始」ボタンを押してください。"
            ui_message_line2 = ""
            ui_message_line3 = ""

        msg_img = draw_message_panel(display_hand_warning)
        btn_img = draw_button_panel()
        combined_screen = create_unified_view(display_frame, latest_board_ui, msg_img, btn_img)
        cv2.imshow("MainGame", combined_screen)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main_loop()