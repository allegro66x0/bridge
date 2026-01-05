import cv2
import numpy as np 
import time
import gomoku_board_recognition as gbr
from PIL import Image, ImageDraw, ImageFont
import copy 
import multiprocessing # C++スレッドのハング防止に必須

try:
    import cpp_gomoku_ai
except ImportError:
    print("="*50)
    print("エラー: C++ AIモジュール (cpp_gomoku_ai.pyd) が見つかりません。")
    print("py setup.py build_ext --inplace を実行しましたか？")
    print("="*50)
    exit()


# ===============================================================
# AI思考ロジック (C++に完全移行)
# ===============================================================
BOARD_SIZE_AI = 13
PLAYER, AI = 2, 1 # AI先手 
SEARCH_DEPTH = 5 

def check_win(board, x, y, player):
    # (変更なし)
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

def find_best_move_parallel(board_np, depth):
    # (変更なし)
    board_for_cpp = board_np.astype(np.int32)
    start_time = time.time()
    move_tuple = cpp_gomoku_ai.find_best_move(board_for_cpp, depth)
    end_time = time.time()
    print(f"[C++ AI] 思考時間: {end_time - start_time:.4f}秒 (探索深度: {depth})")
    return 99999, (move_tuple[0], move_tuple[1])


def convert_discs_to_ai_board(confirmed_discs):
    # (変更なし)
    new_board = np.zeros((BOARD_SIZE_AI, BOARD_SIZE_AI), dtype=np.int32) 
    for d in confirmed_discs:
        row, col = d.cell
        if 0 <= row < BOARD_SIZE_AI and 0 <= col < BOARD_SIZE_AI:
            new_board[row][col] = AI if d.color == gbr.DiscColor.BLACK else PLAYER
    return new_board

# ===============================================================
# メインプログラム (v11.4: 3行表示UI)
# ===============================================================
BOARD_SIZE = 13
CAM_INDEX = 2 # あなたのカメラ番号

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

background_frame, latest_board_ui, saved_intersections = None, None, None
confirmed_discs, intersection_map = [], {}
game_over, winner = False, None
recovery_mode = False 
game_started = False

# [MODIFIED] 変更点 1: UIメッセージを3行に
ui_message_line1 = "'s'キーで背景をセットしてください"
ui_message_line2 = "" 
ui_message_line3 = "" # NEW

def draw_japanese_text(image, text, position, font, color_bgr):
    # (変更なし)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_grid_points(points):
    # (変更なし)
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
    # (変更なし)
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
    # (変更なし)
    margin = cell_size; grid_size = cell_size * (board_size-1); img_size = grid_size + margin*2
    board_img = np.full((img_size, img_size, 3), (218, 179, 125), dtype=np.uint8)
    for i in range(board_size):
        pos = margin + i * cell_size
        cv2.line(board_img, (pos, margin), (pos, margin + grid_size), (0,0,0), 2)
        cv2.line(board_img, (margin, pos), (margin + grid_size, pos), (0,0,0), 2)
    return board_img

# --- メインループ ---
def main_loop():
    global background_frame, latest_board_ui, saved_intersections, intersection_map
    global confirmed_discs, game_over, winner, recovery_mode
    global game_started 
    # [MODIFIED] 変更点 2: 3行メッセージをグローバル宣言
    global ui_message_line1, ui_message_line2, ui_message_line3

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): print(f"❌ カメラ({CAM_INDEX})起動失敗"); exit()

    saved_intersections, intersection_map = calculate_grid_points(FIXED_CORNER_POINTS)
    cv2.namedWindow("Live")
    cv2.namedWindow("Info")
    
    print("C++ AIエンジンをロードしました。カメラを起動します。")

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

        # 's'キー (AI先手)
        if key == ord('s') and not game_started and not game_over:
            background_frame = frame.copy()
            latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
            ui_message_line1 = f"背景を記憶。AI(黒)の初手を思考中です..."
            ui_message_line2 = ""
            ui_message_line3 = ""
            
            # [MODIFIED] 変更点 3: Infoウィンドウの高さを 130px に
            temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
            temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
            cv2.imshow("Info", temp_info_panel)
            cv2.imshow("Result", latest_board_ui) 
            cv2.waitKey(1)

            current_ai_board = convert_discs_to_ai_board(confirmed_discs) 
            _, ai_next_move = find_best_move_parallel(current_ai_board, SEARCH_DEPTH)
            
            r, c = ai_next_move[1], ai_next_move[0] 
            print(f"{c:02d}{r:02d}")
            
            new_disc = gbr.Disc(); new_disc.color = gbr.DiscColor.BLACK; new_disc.cell = (r, c)
            confirmed_discs.append(new_disc)
            
            color = (10,10,10) 
            cv2.circle(latest_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)
            overlay = latest_board_ui.copy()
            cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
            latest_board_ui = cv2.addWeighted(overlay, 0.5, latest_board_ui, 0.5, 0)
            
            # [MODIFIED] 変更点 4: 3行に分割
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
                
                is_player_turn = (black_count > white_count) 
                is_correct_color = (is_player_turn and new_disc.color == gbr.DiscColor.WHITE) or \
                                   (not is_player_turn and new_disc.color == gbr.DiscColor.BLACK) 

                if is_correct_color:
                    confirmed_discs.append(new_disc)
                    y_idx, x_idx = new_disc.cell
                    last_player = AI if new_disc.color == gbr.DiscColor.BLACK else PLAYER
                    
                    current_ai_board = convert_discs_to_ai_board(confirmed_discs)
                    
                    if check_win(current_ai_board, x_idx, y_idx, last_player): game_over, winner = True, last_player
                    
                    board_ui_img = draw_board_ui(BOARD_SIZE, 40)
                    ai_next_move = None
                    current_black_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.BLACK)
                    current_white_count = sum(1 for d in confirmed_discs if d.color == gbr.DiscColor.WHITE)
                    
                    # (AIの思考)
                    if not game_over and current_black_count == current_white_count:
                        ui_message_line1 = f"AI(黒)が思考中です (C++, 深さ{SEARCH_DEPTH})..."
                        ui_message_line2 = ""
                        ui_message_line3 = ""
                        
                        temp_board_ui = latest_board_ui.copy() if latest_board_ui is not None else draw_board_ui(BOARD_SIZE, 40)
                        for d in confirmed_discs: 
                            r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                            cv2.circle(temp_board_ui, (40 + c*40, 40 + r*40), 18, color, -1)

                        # [MODIFIED] Infoウィンドウの高さを 130px に
                        temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
                        temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
                        cv2.imshow("Info", temp_info_panel)
                        cv2.imshow("Result", temp_board_ui) 
                        cv2.waitKey(1)
                        
                        _, ai_next_move = find_best_move_parallel(current_ai_board, SEARCH_DEPTH)
                    
                    for d in confirmed_discs:
                        r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)
                    
                    if ai_next_move:
                        r, c = ai_next_move[1], ai_next_move[0] 
                        print(f"{c:02d}{r:02d}")

                        new_ai_disc = gbr.Disc(); new_ai_disc.color = gbr.DiscColor.BLACK; new_ai_disc.cell = (r, c)
                        confirmed_discs.append(new_ai_disc)
                        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, (10,10,10), -1)

                        current_ai_board = convert_discs_to_ai_board(confirmed_discs)
                        if check_win(current_ai_board, c, r, AI): game_over, winner = True, AI

                        overlay = board_ui_img.copy()
                        cv2.circle(overlay, (40 + c*40, 40 + r*40), 16, (0,255,0), -1)
                        board_ui_img = cv2.addWeighted(overlay, 0.5, board_ui_img, 0.5, 0)
                        if not game_over:
                            # [MODIFIED] 3行に分割
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
                    if not game_over: background_frame = frame.copy()
                else:
                    if is_player_turn:
                        ui_message_line1 = "エラー：白石を置いてください"
                        ui_message_line2 = ""
                        ui_message_line3 = ""
                    else:
                        ui_message_line1 = "エラー：黒石を置いてください"
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
            ui_message_line1 = "背景を再設定。あなたの番(白)です。"
            ui_message_line2 = "好きな場所に白石を置き、「n」キーを押してください。"
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
        
        # [MODIFIED] 変更点 5: 最終的な描画ロジック (3行対応)
        
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
            bbox = font_to_use.getbbox(message_to_draw)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x_pos = (info_panel.shape[1] - w) // 2
            y_pos = (info_panel.shape[0] - h) // 2
            info_panel = draw_japanese_text(info_panel, message_to_draw, (x_pos, y_pos), font_to_use, color_bgr)
        
        elif recovery_mode:
            msg_l1 = "今置いた白石を一度どけて"
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
            # 通常/ゲームオーバーのメッセージ (左上・青)
            # [MODIFIED] 3行描画
            info_panel = draw_japanese_text(info_panel, ui_message_line1, (15, 10), font_ui, color_bgr)
            info_panel = draw_japanese_text(info_panel, ui_message_line2, (15, 45), font_ui, color_bgr) 
            info_panel = draw_japanese_text(info_panel, ui_message_line3, (15, 80), font_ui, color_bgr) # NEW

        cv2.imshow("Info", info_panel)

    cap.release()
    cv2.destroyAllWindows()

# C++スレッドのハング防止に必須
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main_loop()