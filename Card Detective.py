import cv2
import numpy as np
import os
import random
import time

# --- KONFIGURASI TAMPILAN ---
WINDOW_NAME = "Card Detective"
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 600
CAM_W, CAM_H = 640, 480

# Warna (B G R format di OpenCV)
COLOR_BG = (30, 30, 30)       # Abu Gelap
COLOR_PANEL = (50, 50, 50)    # Abu Terang
COLOR_TEXT = (255, 255, 255)  # Putih
COLOR_ACCENT = (0, 255, 255)  # Kuning
COLOR_SUCCESS = (0, 200, 0)   # Hijau
COLOR_FAIL = (0, 0, 200)      # Merah

# --- 1. SETUP DATASET ---
dataset_path = 'dataset_kartu'
database = {}
card_names = []

if not os.path.exists(dataset_path):
    print("Error: Dataset tidak ditemukan!")
    exit()

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 300))
        name = os.path.splitext(filename)[0]
        database[name] = img
        card_names.append(name)

if not card_names:
    print("Dataset kosong!")
    exit()

# --- FUNGSI BANTUAN UI ---
def draw_centered_text(img, text, x, y, w, h, font_scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

def create_ui_layout(frame, sidebar_content):
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = COLOR_BG 
    y_cam = (CANVAS_HEIGHT - CAM_H) // 2
    x_cam = 20 
    frame_resized = cv2.resize(frame, (CAM_W, CAM_H))
    canvas[y_cam:y_cam+CAM_H, x_cam:x_cam+CAM_W] = frame_resized
    cv2.rectangle(canvas, (x_cam-2, y_cam-2), (x_cam+CAM_W+2, y_cam+CAM_H+2), (200,200,200), 2)
    return canvas, x_cam + CAM_W + 20 

# --- FUNGSI LOGIKA DETEKSI ---
def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    dst = np.array([[0, 0], [199, 0], [199, 299], [0, 299]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (200, 300))

def match_card(warped_img, database):
    best_match = None
    min_diff = float('inf')
    for name, db_img in database.items():
        score = np.mean(cv2.absdiff(warped_img, db_img))
        if score < min_diff:
            min_diff = score
            best_match = name
    return best_match if min_diff < 75 else None

# --- MAIN LOOP ---
def main():
    cap = cv2.VideoCapture(0)
    
    score = 0
    lives = 3
    TIME_LIMIT = 10.0
    
    target_card = random.choice(card_names)
    state = "MENU" # MENU, PLAYING, FEEDBACK, GAMEOVER (READY dihapus)
    
    round_start_time = 0
    feedback_start_time = 0
    feedback_msg = ""
    feedback_color = COLOR_TEXT

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. PROSES DETEKSI ---
        detected_name = None
        warped_display = None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and cv2.contourArea(c) > 1000:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                pts = approx.reshape(4, 2)
                warped = four_point_transform(gray, pts)
                warped_display = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                
                if state == "PLAYING":
                    detected_name = match_card(warped, database)
                    if detected_name == target_card:
                        state = "FEEDBACK"
                        score += 1
                        feedback_msg = "BENAR!"
                        feedback_color = COLOR_SUCCESS
                        feedback_start_time = time.time()
                break

        # --- 2. BUAT TAMPILAN UI ---
        canvas, side_x = create_ui_layout(frame, None)
        side_w = CANVAS_WIDTH - side_x - 20
        y_cursor = 50 

        # A. JUDUL GAME
        draw_centered_text(canvas, "CARD DETECTIVE", side_x, 10, side_w, 40, 0.8, COLOR_ACCENT, 2)
        cv2.line(canvas, (side_x, 50), (side_x + side_w, 50), (100,100,100), 1)
        y_cursor += 30

        # B. LOGIKA TAMPILAN PER STATE
        if state == "MENU":
            y_cursor += 100
            draw_centered_text(canvas, "TEKAN [ENTER]", side_x, y_cursor, side_w, 50, 0.8, COLOR_TEXT)
            draw_centered_text(canvas, "UNTUK MULAI", side_x, y_cursor+40, side_w, 50, 0.8, COLOR_TEXT)
        
        elif state == "GAMEOVER":
            y_cursor += 80
            draw_centered_text(canvas, "GAME OVER", side_x, y_cursor, side_w, 60, 1.2, COLOR_FAIL, 3)
            draw_centered_text(canvas, f"SKOR AKHIR: {score}", side_x, y_cursor+60, side_w, 40, 0.8, COLOR_TEXT)
            draw_centered_text(canvas, "[R] RESTART", side_x, y_cursor+150, side_w, 30, 0.6, (150,150,150))

        else: # PLAYING, FEEDBACK
            # 1. INFO PANEL
            y_cursor += 20
            # Skor
            cv2.rectangle(canvas, (side_x, y_cursor), (side_x + side_w//2 - 5, y_cursor+60), COLOR_PANEL, -1)
            draw_centered_text(canvas, "SKOR", side_x, y_cursor+5, side_w//2 - 5, 20, 0.5, (200,200,200), 1)
            draw_centered_text(canvas, str(score), side_x, y_cursor+25, side_w//2 - 5, 30, 0.8, COLOR_TEXT, 2)
            
            # Nyawa
            cv2.rectangle(canvas, (side_x + side_w//2 + 5, y_cursor), (side_x + side_w, y_cursor+60), COLOR_PANEL, -1)
            draw_centered_text(canvas, "NYAWA", side_x + side_w//2 + 5, y_cursor+5, side_w//2 - 5, 20, 0.5, (200,200,200), 1)
            heart_str = " <3 " * lives
            draw_centered_text(canvas, heart_str, side_x + side_w//2 + 5, y_cursor+25, side_w//2 - 5, 30, 0.6, COLOR_FAIL, 2)
            
            y_cursor += 80

            # 2. TARGET KARTU
            cv2.rectangle(canvas, (side_x, y_cursor), (side_x + side_w, y_cursor+100), COLOR_PANEL, -1)
            draw_centered_text(canvas, "CARI KARTU:", side_x, y_cursor+10, side_w, 30, 0.6, (200,200,200), 1)
            
            # Tampilkan nama kartu
            card_display = target_card.replace("_", " ").upper()
            f_scale = 1.0 if len(card_display) < 10 else 0.7
            draw_centered_text(canvas, card_display, side_x, y_cursor+40, side_w, 50, f_scale, COLOR_ACCENT, 2)
            
            # 3. TIMER BAR
            if state == "PLAYING":
                elapsed = time.time() - round_start_time
                time_left = max(0.0, TIME_LIMIT - elapsed)
                if time_left <= 0:
                    state = "FEEDBACK"
                    lives -= 1
                    feedback_msg = "WAKTU HABIS"
                    feedback_color = COLOR_FAIL
                    feedback_start_time = time.time()
            else:
                time_left = 0 # Saat feedback, timer dianggap 0
            
            bar_bg_y = y_cursor + 110
            cv2.rectangle(canvas, (side_x, bar_bg_y), (side_x + side_w, bar_bg_y+15), (20,20,20), -1)
            fill_w = int((time_left / TIME_LIMIT) * side_w)
            col_bar = COLOR_SUCCESS if time_left > 4 else COLOR_FAIL
            cv2.rectangle(canvas, (side_x, bar_bg_y), (side_x + fill_w, bar_bg_y+15), col_bar, -1)
            draw_centered_text(canvas, f"{time_left:.1f}s", side_x, bar_bg_y+20, side_w, 20, 0.5, (200,200,200), 1)

            # 4. AREA SCAN
            y_cursor += 160
            scan_h = 150
            scan_w = 100
            scan_x = side_x + (side_w - scan_w) // 2
            
            cv2.rectangle(canvas, (scan_x-2, y_cursor-2), (scan_x+scan_w+2, y_cursor+scan_h+2), (100,100,100), 1)
            if warped_display is not None:
                warped_resized = cv2.resize(warped_display, (scan_w, scan_h))
                canvas[y_cursor:y_cursor+scan_h, scan_x:scan_x+scan_w] = warped_resized
            else:
                cv2.rectangle(canvas, (scan_x, y_cursor), (scan_x+scan_w, y_cursor+scan_h), (20,20,20), -1)
                draw_centered_text(canvas, "NO SCAN", scan_x, y_cursor, scan_w, scan_h, 0.4, (80,80,80), 1)

        # --- POPUP FEEDBACK (Overlay) ---
        if state == "FEEDBACK":
            center_x, center_y = CANVAS_WIDTH//2, CANVAS_HEIGHT//2
            cv2.rectangle(canvas, (center_x-200, center_y-60), (center_x+200, center_y+60), (0,0,0), -1)
            cv2.rectangle(canvas, (center_x-200, center_y-60), (center_x+200, center_y+60), feedback_color, 3)
            draw_centered_text(canvas, feedback_msg, center_x-200, center_y-60, 400, 120, 1.5, feedback_color, 3)
            
            # --- LOGIKA BARU: LANGSUNG RESET WAKTU OTOMATIS ---
            if time.time() - feedback_start_time > 1.5: # Tunggu 1.5 detik lihat tulisan BENAR
                if lives > 0:
                    state = "PLAYING"                  # Langsung main lagi
                    target_card = random.choice(card_names) # Ganti kartu
                    round_start_time = time.time()     # Reset Timer ke 10 detik
                else:
                    state = "GAMEOVER"

        # TAMPILKAN
        cv2.imshow(WINDOW_NAME, canvas)

        # INPUT KEYBOARD
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        
        if state == "MENU" and key == 13: # Start Game
            score = 0
            lives = 3
            state = "PLAYING" # Langsung main
            target_card = random.choice(card_names)
            round_start_time = time.time()
            
        if state == "GAMEOVER" and key == ord('r'):
            state = "MENU"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()