import sys
import time
import cv2
import numpy as np
import torch

# ====== GPIO hanya tersedia di Jetson ======
try:
    import Jetson.GPIO as GPIO
    JETSON_GPIO_AVAILABLE = True
except Exception as e:
    print(f"  Jetson.GPIO tidak tersedia: {e}")
    JETSON_GPIO_AVAILABLE = False

# ====== Path YOLOv5 & SORT ======
sys.path.append('/home/farhan/Downloads/yolov5-legacy')  # YOLOv5
sys.path.append('/home/farhan/Downloads/sort')            # SORT

from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from models.experimental import attempt_load
from sort import Sort

# =========================
# Inisialisasi YOLOv5
# =========================
device = select_device('0' if torch.cuda.is_available() else 'cpu')
half = device.type != 'cpu'  # FP16 hanya jika bukan CPU
model_path = 'finetuning_100s.pt'

model = attempt_load(model_path, map_location=device)
if half:
    model.half()
model.eval()
print(f"✅ Model loaded on {device} (FP16: {half})")

# =========================
# Buka Kamera
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera!")
    sys.exit(1)

# Info kamera
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback 30 FPS
print(f" Camera: {width}x{height} @ ~{fps_input:.1f} FPS")

# Posisi garis deteksi (60% tinggi frame → area bawah)
line_position = int(height * 0.60)

# =========================
# Inisialisasi SORT Tracker
# =========================
tracker = Sort()  # default params: max_age=1, min_hits=3, iou_threshold=0.3

# =========================
# GPIO & PWM
# =========================
L_PWM = 32  # Pin PWM untuk motor kiri (BOARD numbering)
LED   = 12  # LED indikator (opsional)

if JETSON_GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(L_PWM, GPIO.OUT)
    GPIO.setup(LED, GPIO.OUT)
    nilaiPWM = GPIO.PWM(L_PWM, 500)  # 500 Hz
    nilaiPWM.start(0)                # duty cycle awal 0%
else:
    nilaiPWM = None  # placeholder agar tidak error di referensi

# =========================
# Fuzzy Sugeno sesuai spesifikasi
# =========================
def sugeno(x_in):
    """
    x_in: jumlah noda (deteksi) dalam jendela 3 detik (0..8)
    Output: duty cycle PWM (0..100)
    Spesifikasi:
      μ_Sedikit: [0–4], 1 di [0–1], turun linier ke 0 di 4
      μ_Sedang : [1–7], segitiga puncak di 4
      μ_Banyak : [4–8], naik 4→6, lalu 1 untuk x>6
      z1 = 40*μ_Sedikit          (0–40)
      z2 = 40*μ_Sedang + 30      (30–70)
      z3 = 40*μ_Banyak + 60      (60–100)
      Sugeno: (Σ μ_i * z_i) / (Σ μ_i)
    """
    # Guard domain ke [0,8]
    x = max(0.0, min(float(x_in), 8.0))

    # μ_Sedikit
    if 0 <= x <= 1:
        mu_sedikit = 1.0
    elif 1 < x <= 4:
        mu_sedikit = (4.0 - x) / 3.0
    else:
        mu_sedikit = 0.0

    # μ_Sedang
    if x <= 1 or x >= 7:
        mu_sedang = 0.0
    elif 1 < x <= 4:
        mu_sedang = (x - 1.0) / 3.0
    elif 4 < x < 7:
        mu_sedang = (7.0 - x) / 3.0
    else:  # x == 4
        mu_sedang = 1.0

    # μ_Banyak
    if x < 4:
        mu_banyak = 0.0
    elif 4 < x <= 6:
        mu_banyak = (x - 4.0) / 2.0
    elif x > 6:
        mu_banyak = 1.0
    else:  # x == 4
        mu_banyak = 0.0

    # Konsekuen Sugeno (berbasis μ sesuai spesifikasi)
    z1 = 40.0 * mu_sedikit
    z2 = 40.0 * mu_sedang + 30.0
    z3 = 40.0 * mu_banyak + 60.0

    w1, w2, w3 = mu_sedikit, mu_sedang, mu_banyak
    denom = w1 + w2 + w3
    if denom == 0.0:
        return 0.0

    pwm = (w1 * z1 + w2 * z2 + w3 * z3) / denom
    # Clamp ke 0..100
    return max(0.0, min(pwm, 100.0))

# =========================
# Variabel proses
# =========================
last_update_time = time.time()
window_max = 0                    # menyimpan max objek dalam area selama 3 detik
crossed_ids = set()               # ID unik yang pernah melewati garis
total_crossed = 0                 # informasi (opsional)

# =========================
# Proses Frame
# =========================
def process_frames():
    global last_update_time, window_max, total_crossed, crossed_ids

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame kosong dari kamera, menghentikan loop.")
            break

        t = time.time()

        # ----- Preprocess (YOLOv5 letterbox ke 640) -----
        img = letterbox(frame, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).to(device)
        if half:
            img = img.half()
        img = img.unsqueeze(0)  # add batch dim

        # ----- Inference YOLOv5 -----
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45)

        # ----- Kumpulkan deteksi dan hitung current_in_box -----
        det_list = []
        current_in_box = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    det_list.append([x1, y1, x2, y2, float(conf)])
                    # Hitung jika kotak menyentuh area bawah garis (y2 > line_position)
                    if y2 > line_position:
                        current_in_box += 1

        # Ambil maksimum dalam jendela 3 detik
        window_max = max(window_max, current_in_box)

        # LED indikator ON ketika ada proses
        if JETSON_GPIO_AVAILABLE:
            GPIO.output(LED, GPIO.HIGH)

        # ----- Setiap 3 detik, update tampilan dan PWM -----
        if t - last_update_time >= 3.0:
            display_count = window_max
            display_rpm = sugeno(display_count)

            # Fail-safe: jika count > 8, paksa 100%
            if display_count > 8:
                display_rpm = 100.0

            print(f"[{time.strftime('%H:%M:%S')}] 3s Update → Noda max: {display_count}, PWM fuzzy: {display_rpm:.2f}%")

            # Terapkan PWM
            if JETSON_GPIO_AVAILABLE and nilaiPWM is not None:
                try:
                    nilaiPWM.ChangeDutyCycle(display_rpm)
                except Exception as e:
                    print(f"Gagal set PWM: {e}")

            # Reset jendela
            window_max = 0
            last_update_time = t

        # ----- Update tracker SORT & hitung unik melewati garis -----
        dets_np = np.array(det_list, dtype=np.float32)
        tracks = tracker.update(dets_np) if dets_np.size else np.empty((0, 5), dtype=np.float32)

        for tr in tracks:
            # track format: x1, y1, x2, y2, track_id
            tid = int(tr[4])
            y2 = int(tr[3])
            if y2 > line_position and tid not in crossed_ids:
                crossed_ids.add(tid)

        total_crossed = len(crossed_ids)

# =========================
# Main
# =========================
if __name__ == '__main__':
    try:
        process_frames()
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna (Ctrl+C).")
    except Exception as e:
        print(f"❌ Terjadi error: {e}")
    finally:
        # Cleanup resource
        try:
            cap.release()
        except Exception:
            pass

        # Matikan LED & PWM, cleanup GPIO
        if JETSON_GPIO_AVAILABLE:
            try:
                GPIO.output(LED, GPIO.LOW)
            except Exception:
                pass
            try:
                if nilaiPWM is not None:
                    nilaiPWM.ChangeDutyCycle(0)
                    nilaiPWM.stop()
            except Exception:
                pass
            try:
                GPIO.cleanup()
            except Exception:
                pass

        print("Resource dibersihkan. Program selesai.")
