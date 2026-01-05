# webcam_ai_vs_ai.py (デバッグ表示機能付き・UI分離版)
import cv2
import numpy as np
import time
import math
from PIL import Image, ImageDraw, ImageFont

# --- AIロジックのインポート ---
from battle_board import Board
from renju_ai_logic import RenjuAI
import gomoku_ai_logic as gomoku_ai

# ===============================================================
# メインプログラム (AI vs AI カメラ対局版)
# ===============================================================
BOARD_SIZE = 13
CAM_INDEX = 2 # ご自身のカメラ番号に合わせてください

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ ここに corner_finder.py で取得した座標を貼り付けます ★
FIXED_CORNER_POINTS = [(237, 116), (578, 98), (580, 425), (260, 436)]
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# [MODIFIED] AIの色定義
RENJU_COLOR_CODE = 1   # 連珠AI (先手・黒)
GOMOKU_COLOR_CODE = -1 # 五目並べAI (後手・白)

# [NEW] AIの推奨手 表示色
COLOR_RENJU_HINT = (50, 50, 50)     # 薄い黒
COLOR_GOMOKU_HINT = (200, 200, 200) # 薄い白

# [MODIFIED] 日本語フォントのパスとフォントサイズを複数設定
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"
try:
    font_ui = ImageFont.truetype(FONT_PATH, 24)
    font_warning = ImageFont.truetype(FONT_PATH, 40)
    font_gameover = ImageFont.truetype(FONT_PATH, 80)
except IOError:
    print(f"フォントファイルが見つかりません: {FONT_PATH}")
    print("FONT_PATHをPCに存在する日本語フォントファイルのパスに修正してください。")
    # フォントが見つからなくても続行
    font_ui, font_warning, font_gameover = None, None, None


# --- グローバル変数 ---
background_frame, latest_board_ui, saved_intersections = None, None, None
bg_gray = None # ★[NEW] 背景のグレースケール画像を保持
intersection_map = {}
game_over, winner = False, None
recovery_mode = False
# [MODIFIED] UIメッセージを3行に変更
ui_message_line1 = "'s'キーで背景をセットしてください"
ui_message_line2 = ""
ui_message_line3 = ""

# [NEW] AIと盤面のインスタンス化
board = Board(size=BOARD_SIZE)
renju_ai = RenjuAI(board, max_depth=4, max_time=5.0) # 連珠AI (先手)
gomoku_ai_depth = 2                                 # 五目並べAI (後手) の深さ

current_turn = RENJU_COLOR_CODE # 現在の手番
ai_next_move = None             # AIが推奨する次の手 (x, y)
confirmed_discs = []            # (cell, color_code) のタプルを格納


def draw_japanese_text(image, text, position, font, color_bgr):
    """OpenCVの画像(NumPy配列)に日本語を描画する"""
    if font is None:
        # フォントがない場合、OpenCVの英字フォントで代用
        (x, y) = position
        cv2.putText(image, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return image
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(position, text, font=font, fill=color_rgb)
        
        # タイポ修正 (COLOR_RGB_BGR -> COLOR_RGB2BGR)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        # print(f"テキスト描画エラー: {e}") # デバッグ用に残しても良い
        (x, y) = position
        cv2.putText(image, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return image

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

# ▼▼▼ [修正] グレースケール画像と差分画像を引数で受け取る ▼▼▼
def detect_discs_on_frame(curr_gray, diff_frame, intersection_map):
    """
    フレームの差分から石を検出する
    """
    newly_found, ROI_SIZE, CHANGE_THRESHOLD = [], 12, 35
    
    confirmed_cells = [d[0] for d in confirmed_discs]
    
    for point, cell_coord in intersection_map.items():
        if cell_coord in confirmed_cells:
            continue
            
        x, y = point
        x_s, x_e, y_s, y_e = x-ROI_SIZE//2, x+ROI_SIZE//2, y-ROI_SIZE//2, y+ROI_SIZE//2
        # curr_gray (現在のグレースケール画像) のサイズでチェック
        if 0 <= y_s and y_e < curr_gray.shape[0] and 0 <= x_s and x_e < curr_gray.shape[1]:
            # diff_frame (差分画像) で変化を検知
            if np.mean(diff_frame[y_s:y_e, x_s:x_e]) > CHANGE_THRESHOLD:
                # curr_gray (現在のグレースケール画像) で色を判別
                color_code = GOMOKU_COLOR_CODE if np.mean(curr_gray[y_s:y_e, x_s:x_e]) > 128 else RENJU_COLOR_CODE
                newly_found.append((cell_coord, color_code))
    return newly_found
# ▲▲▲ [修正] ▲▲▲

def draw_board_ui(board_size, cell_size):
    """UI用の碁盤を描画する (変更なし)"""
    margin = cell_size; grid_size = cell_size * (board_size-1); img_size = grid_size + margin*2
    board_img = np.full((img_size, img_size, 3), (218, 179, 125), dtype=np.uint8)
    for i in range(board_size):
        pos = margin + i * cell_size
        cv2.line(board_img, (pos, margin), (pos, margin + grid_size), (0,0,0), 2)
        cv2.line(board_img, (margin, pos), (margin + grid_size, pos), (0,0,0), 2)
    return board_img

# --- [NEW] UI更新処理を関数化 ---
def update_ui_board():
    """現在の状態に基づいてUIボードを生成して返す"""
    # [MODIFIED] ui_message変数をグローバル宣言
    global ui_message_line1, ui_message_line2, ui_message_line3
    
    board_ui_img = draw_board_ui(BOARD_SIZE, 40)
    
    # 確定した石を描画
    for d_cell, d_color in confirmed_discs:
        r, c = d_cell
        color = (10,10,10) if d_color == RENJU_COLOR_CODE else (245,245,245)
        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)
    
    # AIの推奨手を描画
    if ai_next_move:
        r, c = ai_next_move[1], ai_next_move[0]
        
        # ヒントの色は「現在の手番 (current_turn)」のAIの色
        hint_color_bgr = COLOR_RENJU_HINT if current_turn == RENJU_COLOR_CODE else COLOR_GOMOKU_HINT
        
        overlay = board_ui_img.copy()
        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, hint_color_bgr, -1)
        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,0,0), 1) # 枠線
        board_ui_img = cv2.addWeighted(overlay, 0.7, board_ui_img, 0.3, 0)

    if game_over:
        message = "連珠AI(黒)の勝ち！" if winner == RENJU_COLOR_CODE else "五目並べAI(白)の勝ち！"
        if font_gameover:
            bbox = font_gameover.getbbox(message)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x_pos = (board_ui_img.shape[1] - w) // 2
            y_pos = (board_ui_img.shape[0] - h) // 2
        else:
            x_pos, y_pos = 50, board_ui_img.shape[0] // 2
        
        board_ui_img = draw_japanese_text(board_ui_img, message, (x_pos, y_pos), font_gameover, (0, 0, 255))
        # [MODIFIED] 3行メッセージを更新
        ui_message_line1 = "'r'キーでリセット、'q'キーで終了"
        ui_message_line2 = ""
        ui_message_line3 = ""

    return board_ui_img


# --- AIの思考を実行する関数 (変更なし) ---
def get_next_ai_move():
    """現在の手番 (current_turn) に基づいて思考し、推奨手 (ai_next_move) とメッセージ (ui_message) を更新する"""
    # [MODIFIED] 3行メッセージをグローバル宣言
    global ui_message_line1, ui_message_line2, ui_message_line3, ai_next_move
    
    # AI思考中にUIが固まらないよう、先にUIを更新
    temp_ui = update_ui_board() # 現状の盤面でUI更新
    thinking_message = "連珠AI(黒) 思考中..." if current_turn == RENJU_COLOR_CODE else "五目並べAI(白) 思考中..."
    
    # [MODIFIED] メッセージを "Result" から "Info" ウィンドウに表示
    # temp_ui = draw_japanese_text(temp_ui, thinking_message, (10, 5), font_ui, (255, 0, 0))
    cv2.imshow("Result", temp_ui)
    
    # [NEW] Infoウィンドウを作成して思考中メッセージを表示
    info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
    info_panel = draw_japanese_text(info_panel, thinking_message, (15, 10), font_ui, (0, 0, 255))
    cv2.imshow("Info", info_panel)
    cv2.waitKey(1)
        
    if current_turn == RENJU_COLOR_CODE:
        # --- 先手 (連珠AI) のターン ---
        print("--- Renju AI (先手・黒) 思考中 ---")
        x, y = renju_ai.get_best_move()
        ai_next_move = (x, y)
        
        if x is not None:
            print(f"Renju AI (黒) の推奨手: {x:02d}{y:02d}")
            # [MODIFIED] 3行メッセージを更新
            ui_message_line1 = "連珠AI(黒)の手です。"
            ui_message_line2 = "推奨位置(薄い黒)に石を置き、"
            ui_message_line3 = "「n」キーを押してください。"
        else:
            print("Renju AI (黒): パス")
            ui_message_line1 = "連珠AI(黒)がパスしました。"
            ui_message_line2 = "「n」キーを押して次へ"
            ui_message_line3 = ""

    else:
        # --- 後手 (五目並べAI) のターン ---
        print("--- Gomoku AI (後手・白) 思考中 ---")
        _, move = gomoku_ai.minimax(board, gomoku_ai_depth, GOMOKU_COLOR_CODE, -math.inf, math.inf)
        ai_next_move = move # (x, y)
        
        if ai_next_move:
            # ★要求仕様確認: 後手(白)は座標を出力しない
            # print(f"Gomoku AI (白) の推奨手: [{ai_next_move[0]}, {ai_next_move[1]}]") 
            # [MODIFIED] 3行メッセージを更新
            ui_message_line1 = "五目並べAI(白)の手です。"
            ui_message_line2 = "推奨位置(薄い白)に石を置き、"
            ui_message_line3 = "「n」キーを押してください。"
        else:
            print("Gomoku AI (白): パス")
            ui_message_line1 = "五目並べAI(白)がパスしました。"
            ui_message_line2 = "「n」キーを押して次へ"
            ui_message_line3 = ""


# --- メインループ ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened(): print(f"❌ カメラ({CAM_INDEX})起動失敗"); exit()

saved_intersections, intersection_map = calculate_grid_points(FIXED_CORNER_POINTS)

cv2.namedWindow("Live")
cv2.namedWindow("Result") # Resultウィンドウも最初に定義
# [NEW] Infoウィンドウを追加
cv2.namedWindow("Info")
# ▼▼▼ [NEW] デバッグ用ウィンドウを定義 ▼▼▼
cv2.namedWindow("Debug - Grayscale")
cv2.namedWindow("Debug - Difference")
# ▲▲▲ [NEW] ▲▲▲

while True:
    ret, frame = cap.read()
    if not ret: break
    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    # ▼▼▼ [NEW] 毎フレーム、差分画像とグレースケール画像を計算・表示 ▼▼▼
    curr_gray, diff_frame = None, None # 初期化
    if background_frame is not None:
        # 1. 現在のフレームをグレースケールに
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. 背景のグレースケール画像と差分を計算
        diff_frame = cv2.absdiff(bg_gray, curr_gray)
        
        # 3. デバッグウィンドウに表示
        cv2.imshow("Debug - Grayscale", curr_gray)
        cv2.imshow("Debug - Difference", diff_frame)
    # ▲▲▲ [NEW] ▲▲▲

    # リアルタイムの警告
    display_hand_warning = False
    # `diff_frame` が None でない（＝背景が設定済）場合のみ実行
    if diff_frame is not None and not game_over and not recovery_mode: # [MODIFIED] recovery_mode中は警告しない
        # `detect_discs_on_frame` に計算済みの画像を渡す
        real_time_changes = detect_discs_on_frame(curr_gray, diff_frame, intersection_map)
        if len(real_time_changes) > 2:
            display_hand_warning = True

    # カメラ映像の交点表示
    if saved_intersections:
        for point in saved_intersections:
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

    # [MODIFIED] 'b' -> 's' に変更
    if key == ord('s') and not game_over: # [MODIFIED] 試合中でも押せないように (game_started フラグは無いので game_over で代用)
        background_frame = frame.copy()
        # ▼▼▼ [NEW] 背景のグレースケール画像を保持 ▼▼▼
        bg_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        # ▲▲▲ [NEW] ▲▲▲
        
        # [MODIFIED] 3行メッセージを更新
        ui_message_line1 = "背景を記憶しました。"
        ui_message_line2 = "「n」キーを押して"
        ui_message_line3 = "連珠AI(黒)の思考を開始してください。"
        
        ai_next_move = None
        current_turn = RENJU_COLOR_CODE
        latest_board_ui = update_ui_board() # 空の盤面を描画

    # [MODIFIED] nキーのロジック (リカバリーを分離)
    if key == ord('n') and background_frame is not None and not game_over and not recovery_mode:
        
        is_first_action = (ai_next_move is None) 
        
        if is_first_action:
             # --- 1. 初手の処理 (石の認識は行わない) ---
            print("初手：連珠AIの思考を開始します。")
            get_next_ai_move() # 黒AIが思考し、ai_next_move が設定される
            # current_turn は 黒(1) のまま
            
        else:
            # --- 2. 2手目以降の処理 (リカバリーロジックは 'j' に分離) ---
            if curr_gray is not None and diff_frame is not None:
                newly_found_discs = detect_discs_on_frame(curr_gray, diff_frame, intersection_map)
            else:
                newly_found_discs = [] # 背景設定直後など
            
            if len(newly_found_discs) == 1:
                new_disc_cell, new_disc_color = newly_found_discs[0]
                
                # 期待される色 (現在の手番の色)
                expected_color = current_turn
                
                if new_disc_color == expected_color:
                    # 認識成功
                    confirmed_discs.append((new_disc_cell, new_disc_color))
                    y_idx, x_idx = new_disc_cell
                    
                    # 論理盤面に反映
                    board.place_stone(x_idx, y_idx, new_disc_color)
                    
                    # 勝利判定
                    if board.check_win(x_idx, y_idx, new_disc_color):
                        game_over = True
                        winner = new_disc_color
                        ai_next_move = None
                    
                    # 背景画像の更新 (要求仕様)
                    if not game_over:
                        background_frame = frame.copy()
                        # ▼▼▼ [NEW] 背景グレースケールを更新 ▼▼▼
                        bg_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
                        # ▲▲▲ [NEW] ▲▲▲
                        
                        # ★ここでターン交代
                        current_turn *= -1 
                        # ★交代後のAIが思考
                        get_next_ai_move()
                    
                else:
                    # 色が違うエラー
                    expected_color_name = "黒" if expected_color == RENJU_COLOR_CODE else "白"
                    # [MODIFIED] 3行メッセージを更新
                    ui_message_line1 = f"エラー：{expected_color_name}石を置いてください。"
                    ui_message_line2 = ""
                    ui_message_line3 = ""
            
            elif len(newly_found_discs) > 1:
                recovery_mode = True
                # [MODIFIED] 3行メッセージを更新 (リカバリーキーを 'j' に変更)
                ui_message_line1 = f"エラー：石を複数({len(newly_found_discs)})検知。"
                ui_message_line2 = "今置いた石を一度どけて、"
                ui_message_line3 = "「j」キーを押してください。"
            
            else: # len == 0
                # [MODIFIED] 3行メッセージを更新
                ui_message_line1 = "エラー：石を認識できません。"
                ui_message_line2 = "推奨位置に石を置き、"
                ui_message_line3 = "「n」キーを押してください。"
        
        # --- 3. UIボードの更新 (nキーが押されたら必ず実行) ---
        latest_board_ui = update_ui_board()
        
    # [NEW] リカバリーキー 'j' を新設
    if key == ord('j') and recovery_mode:
        background_frame = frame.copy()
        bg_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        recovery_mode = False
        # [MODIFIED] 3行メッセージを更新
        ui_message_line1 = "背景を再設定しました。"
        ui_message_line2 = "推奨位置に石を置き、"
        ui_message_line3 = "「n」キーを押してください。"
        # AIの思考は行わず、UIの更新は次のループに任せる
        latest_board_ui = update_ui_board()


    if key == ord('r'):
        background_frame, latest_board_ui, confirmed_discs = None, None, []
        bg_gray = None # ★[NEW] 背景グレースケールもリセット
        game_over, recovery_mode = False, False
        winner = None
        ai_next_move = None
        current_turn = RENJU_COLOR_CODE
        # [NEW] 盤面とAIの状態をリセット
        board = Board(size=BOARD_SIZE)
        renju_ai.board = board # AIが参照する盤面オブジェクトも更新
        
        # [MODIFIED] 3行メッセージを更新 ('b' -> 's')
        ui_message_line1 = "リセットしました。"
        ui_message_line2 = "'s'キーで背景をセットしてください。"
        ui_message_line3 = ""
        latest_board_ui = None # リセットしたらUIも消す

    # --- 毎フレームの描画処理 ---
    cv2.imshow("Live", display_frame)
    
    # [MODIFIED] 描画ロジックを "Result" と "Info" に完全分離
    
    # 1. 盤面ウィンドウ ("Result") の表示
    if latest_board_ui is not None:
        cv2.imshow("Result", latest_board_ui)

    # 2. 情報ウィンドウ ("Info") の作成と表示
    # (横幅700px, 高さ130px の白紙)
    info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
    color_bgr = (255, 0, 0) # 青
    font_to_use = font_ui

    if display_hand_warning:
        message_to_draw = "手をどけてください"
        font_to_use = font_warning
        color_bgr = (0, 0, 255) # 赤
        if font_warning:
            bbox = font_to_use.getbbox(message_to_draw)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x_pos = (info_panel.shape[1] - w) // 2
            y_pos = (info_panel.shape[0] - h) // 2
        else:
            x_pos, y_pos = 50, info_panel.shape[0] // 2
        
        info_panel = draw_japanese_text(info_panel, message_to_draw, (x_pos, y_pos), font_to_use, color_bgr)
    
    elif recovery_mode:
        # [NEW] リカバリーモード専用メッセージ ('j'キー)
        msg_l1 = "今置いた石を一度どけて"
        msg_l2 = "「j」キーを押してください"
        font_to_use = font_warning # 40pt
        color_bgr = (0, 0, 255) # 赤
        
        if font_warning:
            bbox1 = font_to_use.getbbox(msg_l1)
            w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
            x_pos1 = (info_panel.shape[1] - w1) // 2
            y_pos1 = (info_panel.shape[0] - h1*2 - 10) // 2 
            
            bbox2 = font_to_use.getbbox(msg_l2)
            w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox1[1]
            x_pos2 = (info_panel.shape[1] - w2) // 2
            y_pos2 = y_pos1 + h1 + 10 
        else:
            x_pos1, y_pos1 = 15, 10
            x_pos2, y_pos2 = 15, 45

        info_panel = draw_japanese_text(info_panel, msg_l1, (x_pos1, y_pos1), font_to_use, color_bgr)
        info_panel = draw_japanese_text(info_panel, msg_l2, (x_pos2, y_pos2), font_to_use, color_bgr)

    else:
        # 通常/ゲームオーバーのメッセージ
        info_panel = draw_japanese_text(info_panel, ui_message_line1, (15, 10), font_ui, color_bgr)
        info_panel = draw_japanese_text(info_panel, ui_message_line2, (15, 45), font_ui, color_bgr) 
        info_panel = draw_japanese_text(info_panel, ui_message_line3, (15, 80), font_ui, color_bgr) # NEW

    cv2.imshow("Info", info_panel)

cap.release()
cv2.destroyAllWindows()