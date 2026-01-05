import numpy as np
import cv2
from enum import IntEnum

# --- クラス定義 ---
class DiscColor(IntEnum):
    BLACK = 0
    WHITE = 1

class RecognizerType(IntEnum):
    REALBOARD = 0

class Hint():
    def __init__(self):
        pass

class Disc():
    def __init__(self):
        self.color = None
        self.position = None
        self.cell = None

class Result():
    def __init__(self):
        self.disc = []
        self.vertex = []
        self.recognizerType = None

    def clearDiscInfo(self):
        self.disc = []

def to_gray_safe(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# --- 認識処理の基底クラス ---
class BaseRecognizer:
    _CELL_SIZE = 30
    _BOARD_MARGIN = 15
    
    def __init__(self, board_size=14):
        self.BOARD_SIZE = board_size
        # 1ピクセルの誤差を補正するために +1 を追加
        self._EXTRACT_IMG_SIZE = self._CELL_SIZE * (self.BOARD_SIZE - 1) + 1 + self._BOARD_MARGIN * 2

    def extractBoard(self, image, vertex, size):
        height, width = size[1], size[0]
        src = np.array(vertex, dtype=np.float32)
        margin = self._BOARD_MARGIN
        dst = np.array([
            [margin, margin], [width - 1 - margin, margin],
            [width - 1 - margin, height - 1 - margin], [margin, height - 1 - margin]
        ], dtype=np.float32)
        
        trans = cv2.getPerspectiveTransform(src, dst)
        board = cv2.warpPerspective(image, trans, (int(width), int(height)), flags=cv2.INTER_LINEAR)
        return board

# --- 実際の盤を認識するクラス ---
class RealBoardRecognizer(BaseRecognizer):

    def _detectConvexHull(self, image):
        debug_images = {}
        height, width, _ = image.shape

        # --- ステップ1：色の情報で盤のおおまかな領域を特定 ---
        # BGR画像をHSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 木の色（黄色〜茶色）の範囲を定義 ※環境によって要調整
        lower_wood = np.array([15, 50, 50])
        upper_wood = np.array([30, 255, 255])
        
        # マスクを作成
        color_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        
        # マスクのノイズを除去し、領域を閉じる
        kernel = np.ones((5,5),np.uint8)
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        debug_images["1_ColorMask"] = closed_mask

        # --- ステップ2：特定した盤の領域内だけで二値化を実行 ---
        gray = to_gray_safe(image)
        
        # 盤の領域外を黒く塗りつぶし、背景ノイズを消す
        masked_gray = cv2.bitwise_and(gray, gray, mask=closed_mask)

        # 木目のような細かいノイズを除去するために、画像をぼかす
        blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        # ★★★★★★★★★★★★★★★★★★★★★

        # ぼかした画像に対して適応的二値化を実行
        binary_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 15, 3)
        
        # マスクの外側で発生する不要な線を消す
        binary_img = cv2.bitwise_and(binary_img, binary_img, mask=closed_mask)
        debug_images["2_BinaryImage_Hybrid"] = binary_img

        # --- ステップ3：二値化画像から輪郭を検出し、盤を特定（これ以降は前回と同じ）---
        contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return False, None, debug_images

        found_board_contour = None
        max_area = 0
        min_area_threshold = width * height * 0.1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_threshold: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                if area > max_area:
                    found_board_contour = approx
                    max_area = area

        if found_board_contour is None: return False, None, debug_images

        points = found_board_contour.reshape(4, 2)
        box = sorted(points, key=lambda p: p[1])
        tl, tr = sorted(box[:2], key=lambda p: p[0])
        bl, br = sorted(box[2:], key=lambda p: p[0])

        expand_ratio = 0.5 / (self.BOARD_SIZE - 1)
        center_x = (tl[0] + tr[0] + br[0] + bl[0]) / 4
        center_y = (tl[1] + tr[1] + br[1] + bl[1]) / 4
        center = np.array([center_x, center_y])
        
        tl_expanded = tl - (center - tl) * expand_ratio
        tr_expanded = tr - (center - tr) * expand_ratio
        br_expanded = br - (center - br) * expand_ratio
        bl_expanded = bl - (center - bl) * expand_ratio
        
        hull = np.array([tl_expanded, tr_expanded, br_expanded, bl_expanded], dtype=np.float32)

        debug_contour_img = image.copy()
        cv2.drawContours(debug_contour_img, [found_board_contour], -1, (0, 255, 0), 2)
        cv2.polylines(debug_contour_img, [np.int32(hull)], True, (0, 0, 255), 2)
        debug_images["3_DetectedBoardContour_Hybrid"] = debug_contour_img

        return True, hull, debug_images

    def detectDisc(self, background_image, original_board, intersection_points):
        result = Result()
        result.clearDiscInfo()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        background_gray = to_gray_safe(background_image)
        background_gray = clahe.apply(background_gray)
        current_gray = to_gray_safe(original_board)
        current_gray = clahe.apply(current_gray)
        
        diff_image = cv2.absdiff(current_gray, background_gray)
        
        ROI_SIZE = 10
        CHANGE_THRESHOLD = 30

        for idx, point in enumerate(intersection_points):
            x, y = int(point[0]), int(point[1])
            x_start, x_end = x - ROI_SIZE // 2, x + ROI_SIZE // 2
            y_start, y_end = y - ROI_SIZE // 2, y + ROI_SIZE // 2

            diff_roi = diff_image[y_start:y_end, x_start:x_end]
            avg_diff = np.mean(diff_roi)
            
            if avg_diff > CHANGE_THRESHOLD:
                color_roi = current_gray[y_start:y_end, x_start:x_end]
                avg_color = np.mean(color_roi)
                
                disc = Disc()
                disc.color = DiscColor.WHITE if avg_color > 128 else DiscColor.BLACK
                
                row_idx, col_idx = idx // self.BOARD_SIZE, idx % self.BOARD_SIZE
                disc.cell = (row_idx, col_idx)
                result.disc.append(disc)

        return True, result, None

# --- 認識処理を自動で振り分けるクラス ---
class AutomaticRecognizer(BaseRecognizer):
    def __init__(self, board_size=13):
        super().__init__(board_size)
        self._REAL_RECOGNIZER = RealBoardRecognizer(board_size)

    def detectBoard(self, image, hint):
        ret, result_hull, debug_images = self._REAL_RECOGNIZER._detectConvexHull(image)
        if ret:
            result_obj = Result()
            result_obj.vertex = result_hull
            return ret, result_obj, debug_images
        else:
            return False, None, debug_images

    def detectDisc(self, background_image, original_board, intersection_points):
        return self._REAL_RECOGNIZER.detectDisc(background_image, original_board, intersection_points)