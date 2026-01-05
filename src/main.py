import cv2
import numpy as np
import time
import serial # シリアル通信用ライブラリ
import sys
import os
# srcディレクトリをパスに追加して、同階層のモジュールをインポートできるようにする
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gomoku_board_recognition as gbr
from PIL import Image, ImageDraw, ImageFont
from hardware.coin_sorter import CoinSorter

# ===============================================================
# シリアル通信設定 (環境に合わせて変更してください)
# ===============================================================
SERIAL_PORT = 'COM10'  # ガントリー用
SERIAL_PORT_SORTER = 'COM11' # コインソーター用 (★要設定)
BAUD_RATE = 9600
ser = None  # シリアルポートオブジェクト
sorter = CoinSorter(SERIAL_PORT_SORTER) # コインソーターオブジェクト

# コピックアップ地点のROI (x, y, w, h) - 画像認識で監視する領域
# ※ カメラの画角に合わせて調整が必要です
PICKUP_ROI = (50, 50, 60, 60) 

# ===============================================================
# AI思考ロジック
# ===============================================================
BOARD_SIZE_AI = 13
PLAYER, AI = 2, 1 
board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]

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
    global board_for_ai
    board_for_ai = [[0] * BOARD_SIZE_AI for _ in range(BOARD_SIZE_AI)]
    for d in confirmed_discs:
        row, col = d.cell
        if 0 <= row < BOARD_SIZE_AI and 0 <= col < BOARD_SIZE_AI:
            board_for_ai[row][col] = AI if d.color == gbr.DiscColor.BLACK else PLAYER

# ===============================================================
# ガントリー制御用関数
# ===============================================================
def wait_for_arduino_ready():
    """Arduinoから 'READY' が返ってくるまで待機する"""
    if ser is None or not ser.is_open: return

    print("ガントリーの動作完了を待っています...")
    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line == "READY":
                    print("-> Arduino Ready!")
                    break
                elif line:
                    print(f"Arduino: {line}")
            except:
                pass
        # ウィンドウがフリーズしないようにイベント処理
        cv2.waitKey(1)

def send_command_to_gantry(x, y):
    """座標(x=col, y=row)をArduinoへ送信し、動作完了を待つ"""
    if ser is None or not ser.is_open:
        print("エラー: シリアルポートが開いていません")
        return

    # コマンド作成 (例: x=4, y=8 -> "0408\n")
    # Arduinoコードに合わせて x, y の順で送信
    command = f"{x:02}{y:02}\n"
    print(f"ガントリーへ送信: {command.strip()}")
    
    ser.write(command.encode('utf-8'))
    
    # 動作完了まで待機 (この間、ゲーム進行はブロックされる)
    wait_for_arduino_ready()

# ===============================================================
# メインプログラム
# ===============================================================
BOARD_SIZE = 13
CAM_INDEX = 2
FIXED_CORNER_POINTS = [(276, 127), (571, 127), (573, 429), (269, 424)]
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"

# --- シリアルポート接続 ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"{SERIAL_PORT} に接続しました。初期化(原点復帰)を待っています...")
    # 起動時の原点復帰完了を待つ
    wait_for_arduino_ready()
except Exception as e:
    print(f"❌ ガントリー接続エラー: {e}")
    print("Arduinoが接続されているか、PORT設定が正しいか確認してください。")
    ser = None

# --- コインソーター接続 ---
if sorter.connect():
    print("✅ コインソーターに接続しました")
else:
    print(f"⚠️ コインソーター接続失敗: {SERIAL_PORT_SORTER} を確認してください(動作に影響はありません)")

# ... (rest of init code) ...

def check_pickup_point_and_feed(frame):
    """ピックアップ地点を監視し、コマがなければ供給する"""
    if not sorter.is_connected: return

    x, y, w, h = PICKUP_ROI
    
    # 範囲外チェック
    if y+h >= frame.shape[0] or x+w >= frame.shape[1]: 
        return

    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 簡単な判定ロジック:
    # ROIの平均輝度や分散を見る、あるいは「変化」を見る
    # ここでは「黒っぽい石があるか」を判定する仮ロジック
    # (実際には、トレイの色と石の色に合わせて閾値を調整する必要があります)
    
    # 例: 平均輝度が低い(黒石) または 高い(白石) -> 石がある
    #     中間ぐらい(トレイの色) -> 石がない
    
    # ★仮実装: 石がない(Empty)とみなす条件★
    # ここは実環境に合わせて必ず調整してください
    mean_val = np.mean(gray_roi)
    
    # トレイが灰色(100-150)で、石が白(>200)か黒(<50)の場合
    has_stone = (mean_val < 80) or (mean_val > 180) 
    
    if not has_stone:
        # 石がない -> 供給ON
        sorter.start_all()
        # 画面に描画（デバッグ用）
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # 赤枠 = 供給中
    else:
        # 石がある -> 停止
        sorter.stop_all()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # 緑枠 = 準備OK

try:
    font_ui = ImageFont.truetype(FONT_PATH, 24)
    font_warning = ImageFont.truetype(FONT_PATH, 40)
    font_gameover = ImageFont.truetype(FONT_PATH, 80)
except IOError:
    print(f"フォントファイルが見つかりません: {FONT_PATH}")
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
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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

    # ★ コインソーター制御 (毎フレーム確認) ★
    check_pickup_point_and_feed(display_frame)

    # 's'キー (AIの初手実行)
    if key == ord('s') and not game_started:
        background_frame = frame.copy()
        latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
        
        ui_message_line1 = "背景を記憶。AI(黒)の初手を思考中です..."
        temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
        temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
        cv2.imshow("Info", temp_info_panel)
        cv2.imshow("Result", latest_board_ui) 
        cv2.waitKey(1)

        convert_discs_to_ai_board(confirmed_discs)
        _, ai_next_move = minimax(2, AI, -float("inf"), float("inf"))
        
        r, c = ai_next_move[1], ai_next_move[0] 
        print(f"AI Move: {c:02d}{r:02d}")

        # ★★★ ここでガントリーへ送信 ★★★
        send_command_to_gantry(c, r) 
        
        new_disc = gbr.Disc(); new_disc.color = gbr.DiscColor.BLACK; new_disc.cell = (r, c)
        confirmed_discs.append(new_disc)
        
        # ロボットが石を置いたはずなので、背景画像を更新して手の映り込み誤検知などを防ぐ
        # (ロボットのアームが完全退避した後なので、今のフレームを新背景にするのが安全)
        # 念のため数フレーム読み飛ばす処理を入れても良いが、今回はシンプルに更新
        ret, frame = cap.read()
        if ret: background_frame = frame.copy()

        convert_discs_to_ai_board(confirmed_discs)
        
        color = (10,10,10)
        cv2.circle(latest_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)
        overlay = latest_board_ui.copy()
        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
        latest_board_ui = cv2.addWeighted(overlay, 0.5, latest_board_ui, 0.5, 0)
        
        ui_message_line1 = "AIが緑の円に(黒を)打ちました。"
        ui_message_line2 = "あなたの番(白)です。"
        ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
        game_started = True 

    # 'n'キー (プレイヤーの手番終了 -> AIの手番)
    if key == ord('n') and background_frame is not None and not game_over and game_started and not recovery_mode:
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
                
                convert_discs_to_ai_board(confirmed_discs)
                if check_win(x_idx, y_idx, last_player): game_over, winner = True, last_player
                
                board_ui_img = draw_board_ui(BOARD_SIZE, 40)
                ai_next_move = None
                current_black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
                current_white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
                
                if not game_over and current_black_count == current_white_count:
                    ui_message_line1 = "AI(黒)が思考中です..."
                    ui_message_line2 = ""
                    ui_message_line3 = ""
                    
                    temp_board_ui = draw_board_ui(BOARD_SIZE, 40)
                    for d in confirmed_discs: 
                        r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                        cv2.circle(temp_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)

                    temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
                    temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
                    cv2.imshow("Info", temp_info_panel)
                    cv2.imshow("Result", temp_board_ui) 
                    cv2.waitKey(1)

                    _, ai_next_move = minimax(2, AI, -float("inf"), float("inf")) # AI(黒)の手番
                
                for d in confirmed_discs:
                    r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                    cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)
                
                if ai_next_move:
                    r, c = ai_next_move[1], ai_next_move[0]
                    print(f"AI Move: {c:02d}{r:02d}")

                    # ★★★ ここでガントリーへ送信 ★★★
                    send_command_to_gantry(c, r)
                    
                    new_ai_disc = gbr.Disc(); new_ai_disc.color = gbr.DiscColor.BLACK; new_ai_disc.cell = (r, c)
                    confirmed_discs.append(new_ai_disc)

                    # ロボットが石を置いたので、背景更新
                    ret, frame = cap.read()
                    if ret: background_frame = frame.copy()

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
                else: 
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
                # [MODIFIED] ロボット動作後の背景更新は上記で行っているのでここは削除しても良いが、念の為
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

    # 'j'キー (リカバリーモード)
    if key == ord('j') and recovery_mode:
        background_frame = frame.copy()
        recovery_mode = False
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
    
    if latest_board_ui is not None:
        cv2.imshow("Result", latest_board_ui)

    info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
    color_bgr = (255, 0, 0) 

    if display_hand_warning:
        message_to_draw = "手をどけてください"
        font_to_use = font_warning
        color_bgr = (0, 0, 255) 
        bbox = font_to_use.getbbox(message_to_draw)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x_pos = (info_panel.shape[1] - w) // 2
        y_pos = (info_panel.shape[0] - h) // 2
        info_panel = draw_japanese_text(info_panel, message_to_draw, (x_pos, y_pos), font_to_use, color_bgr)
    
    elif recovery_mode:
        msg_l1 = "今置いた石を一度どけて"
        msg_l2 = "「j」キーを押してください"
        font_to_use = font_warning
        color_bgr = (0, 0, 255) 
        
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
        info_panel = draw_japanese_text(info_panel, ui_message_line1, (15, 10), font_ui, color_bgr)
        info_panel = draw_japanese_text(info_panel, ui_message_line2, (15, 45), font_ui, color_bgr) 
        info_panel = draw_japanese_text(info_panel, ui_message_line3, (15, 80), font_ui, color_bgr)

    cv2.imshow("Info", info_panel)


cap.release()
if ser is not None and ser.is_open:
    ser.close()
cv2.destroyAllWindows()