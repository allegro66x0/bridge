import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# ===============================================================
# 設定・定数
# ===============================================================
BOARD_SIZE = 13
AI = 1      # 黒（先手）
PLAYER = 2  # 白（後手）
EMPTY = 0

# UI設定
CELL_SIZE = 40
MARGIN = 40
GRID_SIZE = CELL_SIZE * (BOARD_SIZE - 1)
WINDOW_SIZE = GRID_SIZE + MARGIN * 2
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc" 

# ===============================================================
# AI評価ロジック（脳みそを復活）
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

# --- 評価関数群 (盤面の点数を計算する) ---
def evaluate_line(line, player):
    opponent = PLAYER if player == AI else AI
    if opponent in line: return 0 # 相手の石があれば価値なし
    player_stones = line.count(player)
    # 点数配分
    if player_stones == 5: return 100000
    if player_stones == 4: return 1000
    if player_stones == 3: return 100
    if player_stones == 2: return 10
    return 0

def count_patterns_for_player(board_grid, player):
    score = 0
    # 横・縦・斜めのすべての5マスの並びをチェックして点数を足す
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
    # 自分の有利さ - 相手の有利さ
    opponent = PLAYER if player == AI else AI
    return count_patterns_for_player(board_grid, player) - count_patterns_for_player(board_grid, opponent) * 1.2

# ===============================================================
# ★改良版: 人間らしい接待AIロジック
# ===============================================================
def pick_move_reception_smart(board, ai_player):
    human_player = PLAYER if ai_player == AI else AI
    candidates = gen_moves(board, radius=2)
    if not candidates: return None

    # --- 1. [防衛] プレイヤーの必勝手(リーチ)を防ぐ ---
    # ここは「接待」として試合を長引かせるために本気で防ぐ
    threat_moves = []
    for (r, c) in candidates:
        board.grid[r][c] = human_player # 仮にプレイヤーが置いたとする
        
        # 簡易脅威判定: 置いた場所を含んで4つ以上並ぶなら危険
        is_dangerous = False
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = c + i*dx, r + i*dy
                if board.inside(ny, nx) and board.grid[ny][nx] == human_player: count += 1
                else: break
            for i in range(1, 5):
                nx, ny = c - i*dx, r - i*dy
                if board.inside(ny, nx) and board.grid[ny][nx] == human_player: count += 1
                else: break
            if count >= 4: 
                is_dangerous = True
                break
        
        board.grid[r][c] = 0 # 元に戻す
        if is_dangerous:
            threat_moves.append((r, c))

    if threat_moves:
        print(f"AI: プレイヤーの攻撃を検知！防衛します。")
        return random.choice(threat_moves)

    # --- 2. [手選び] すべての候補手を評価してランク付けする ---
    scored_moves = []
    for (r, c) in candidates:
        b_copy = board.copy()
        b_copy.play(r, c, ai_player)
        score = get_score(b_copy.grid, ai_player)
        scored_moves.append(((r, c), score))
    
    # 点数が高い順に並び替え
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    # --- 3. [接待フィルター] ---
    # 方針: 1位（最強の手）は避けて、2位〜5位くらいの手から選ぶ
    
    # 候補が少なすぎる場合はランダム
    if len(scored_moves) <= 1:
        return scored_moves[0][0]
    
    # 上位の手を抽出
    top_moves = scored_moves[:min(6, len(scored_moves))]
    
    # ここで「AIが勝ち確（4連以上）」になる手が含まれていたら、それは除外したい
    # (AIが勝って終わらせないため)
    filtered_moves = []
    for move, score in top_moves:
        # scoreが異常に高い(10000以上)場合は5連ができているので避ける
        # scoreが1000以上なら4連ができているので避ける
        if score < 900: 
            filtered_moves.append(move)
    
    if not filtered_moves:
        # もし全部勝ち手なら、仕方ないので一番点数の低い勝ち手を選ぶ（またはランダム）
        return random.choice(candidates)

    # 残った「そこそこ良い手」の中から選ぶ
    # 1番良い手(インデックス0)は、30%くらいの確率で除外して、2番目以降を選びやすくする
    if len(filtered_moves) >= 2:
        # インデックス: 0=1位, 1=2位, 2=3位...
        # 1位, 2位, 3位, 4位 の中からランダム（少し下の順位が出やすいように重みづけしてもいいが、単純ランダムでOK）
        
        # 例: 上位5個のうち、1位を外して 2,3,4位から選ぶ
        # ただし、候補が少なければあるものから選ぶ
        pick_idx = random.choice(range(len(filtered_moves)))
        
        # 「1位の手」をあえて避けるロジック (7割の確率で2位以下にする)
        if pick_idx == 0 and len(filtered_moves) > 1 and random.random() < 0.7:
            pick_idx = random.randint(1, len(filtered_moves)-1)
            
        choice = filtered_moves[pick_idx]
        print(f"AI: {len(scored_moves)}手の中で {pick_idx+1}番目に良さそうな手を選びました")
        return choice
    else:
        return filtered_moves[0]


# ===============================================================
# UI & ゲーム制御 (Webcam版)
# ===============================================================
import gomoku_board_recognition as gbr

# UI設定
CAM_INDEX = 2
FIXED_CORNER_POINTS = [(171, 58), (512, 58), (508, 406), (161, 398)]

background_frame, latest_board_ui, saved_intersections = None, None, None
confirmed_discs, intersection_map = [], {}
game_over, winner = False, None
recovery_mode = False
game_started = False 
ai_turn_count = 0 

ui_message_line1 = "'s'キーで背景をセットしてください"
ui_message_line2 = ""
ui_message_line3 = ""

def calculate_grid_points(points):
    """4点の座標から盤上の全ての交点座標を計算する"""
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
    """フレームの差分から石を検出する"""
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
    """UI用の碁盤を描画する"""
    margin = cell_size; grid_size = cell_size * (board_size-1); img_size = grid_size + margin*2
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

def draw_japanese_text(image, text, position, font, color_bgr):
    if font is None: return image
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    global background_frame, latest_board_ui, saved_intersections, intersection_map
    global confirmed_discs, game_over, winner, recovery_mode
    global game_started, ai_turn_count
    global ui_message_line1, ui_message_line2, ui_message_line3

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): print(f"❌ カメラ({CAM_INDEX})起動失敗"); exit()

    saved_intersections, intersection_map = calculate_grid_points(FIXED_CORNER_POINTS)

    cv2.namedWindow("Live")
    cv2.namedWindow("Info") 
    cv2.namedWindow("Result")

    try:
        font_ui = ImageFont.truetype(FONT_PATH, 24)
        font_warning = ImageFont.truetype(FONT_PATH, 40)
        font_gameover = ImageFont.truetype(FONT_PATH, 80)
    except IOError:
        font_ui = None

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

        # 's'キー (AIの初手実行)
        if key == ord('s') and not game_started:
            background_frame = frame.copy()
            latest_board_ui = draw_board_ui(BOARD_SIZE, 40)
            
            ui_message_line1 = "背景を記憶。AI(黒)の初手を思考中です..."
            ui_message_line2 = ""
            ui_message_line3 = ""
            
            temp_info_panel = np.full((130, 700, 3), (255, 255, 255), dtype=np.uint8)
            temp_info_panel = draw_japanese_text(temp_info_panel, ui_message_line1, (15, 10), font_ui, (0,0,255))
            cv2.imshow("Info", temp_info_panel)
            cv2.imshow("Result", latest_board_ui) 
            cv2.waitKey(1)

            current_board = convert_discs_to_board_obj(confirmed_discs)
            # Level 1 calling convention: (board, ai_player)
            move = pick_move_reception_smart(current_board, AI)
            
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
                ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
                game_started = True 
            else:
                pass

        # 'n'キー (プレイヤーの手番)
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
                    
                    current_board = convert_discs_to_board_obj(confirmed_discs)
                    if current_board.check_win(x_idx, y_idx, PLAYER): 
                        game_over, winner = True, PLAYER
                    
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

                        current_board = convert_discs_to_board_obj(confirmed_discs)
                        # Level 1 calling convention
                        move = pick_move_reception_smart(current_board, AI)
                        if move:
                             ai_next_move = move
                             ai_turn_count += 1
                    
                    for d in confirmed_discs:
                        r, c = d.cell; color = (10,10,10) if d.color == gbr.DiscColor.BLACK else (245,245,245)
                        cv2.circle(board_ui_img, (40 + c*40, 40 + r*40), 18, color, -1)
                    
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
                            ui_message_line3 = "好きな場所に白石を置き、「n」キーを押してください。"
                    else: 
                        if not game_over and winner is None:
                           pass
                        elif not game_over:
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

        # 'j'キー
        if key == ord('j') and recovery_mode:
            background_frame = frame.copy()
            recovery_mode = False
            ui_message_line1 = "背景を再設定しました。"
            ui_message_line2 = "あなたの番(白)です。石を置いて「n」キーを押してください。"
            ui_message_line3 = ""

        # 'r'キー
        if key == ord('r'):
            background_frame, latest_board_ui, confirmed_discs = None, None, []
            game_over, recovery_mode = False, False
            winner = None
            game_started = False 
            ai_turn_count = 0 
            ui_message_line1 = "リセットしました。's'キーで背景をセットしてください。"
            ui_message_line2 = ""
            ui_message_line3 = ""

        cv2.imshow("Live", display_frame)
        
        if latest_board_ui is not None:
            cv2.imshow("Result", latest_board_ui)

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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()