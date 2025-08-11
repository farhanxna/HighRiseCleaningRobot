import sys
import torch
import cv2
import numpy as np
import time
from flask import Flask, Response
import Jetson.GPIO as GPIO

# Menambahkan path folder 'yolov5' dan 'sort' ke sys.path
sys.path.append('/home/farhan/Downloads/yolov5-legacy')  # Path untuk YOLOv5
sys.path.append('/home/farhan/Downloads/sort')  # Path untuk SORT

from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from models.experimental import attempt_load
from sort import Sort

# Inisialisasi device dan model YOLOv5
device = select_device('0' if torch.cuda.is_available() else 'cpu')
half = device.type != 'cpu'
model_path = 'finetuning_100s.pt'
model = attempt_load(model_path, map_location=device)
if half:
    model.half()
model.eval()
print(f"✅ Model loaded on {device} (FP16: {half})")

# ---------------------------------------------------
# Buka kamera (index 0). Ganti 0 jika butuh kamera lain.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Gagal membuka kamera!")
    exit(1)

# Ambil resolusi frame dari kamera
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS) or 30  # default 30 jika FPS tidak tersedia

# (Optional) Siapkan video writer jika ingin merekam output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_camera.mp4', fourcc, fps_input, (width, height))
print(f"📼 Output camera akan disimpan di: output_camera.mp4")
# ---------------------------------------------------

# Posisi garis merah (75% tinggi frame)
line_position = int(height * 0.60)

# Inisialisasi tracker SORT
tracker = Sort()

# Variabel penghitung melewati garis merah
total_crossed = 0
crossed_ids = set()

# Timer dan window max untuk kotak biru setiap 3 detik
last_update_time = time.time()
window_max = 0

# Variabel untuk menyimpan nilai yang ditampilkan setiap 3 detik
display_count = 0
display_rpm = 0.0

# Font untuk overlay teks
font = cv2.FONT_HERSHEY_SIMPLEX

# Inisialisasi pin PWM untuk motor
L_PWM = 32  # Pin PWM untuk motor kiri
LED = 12
# Setup GPIO
GPIO.setmode(GPIO.BOARD)  # Bisa menggunakan BOARD atau BCM sesuai preferensi Anda
GPIO.setup(L_PWM, GPIO.OUT)
GPIO.setup(LED, GPIO.OUT)

# Membuat objek PWM pada pin L_PWM dengan frekuensi 500 Hz
nilaiPWM = GPIO.PWM(L_PWM, 500)

# Mulai dengan duty cycle 0%
nilaiPWM.start(0)

# Menyimpan nilai duty cycle saat ini
current_pwm_value = 0

# Definisi fungsi Fuzzy Sugeno
def sugeno(x):
    mu1 = max(min((3 - x) / 3, 1), 0)  # Sedikit
    if x < 1 or x > 7:
        mu2 = 0
    elif x <= 4:
        mu2 = (x - 1) / 3
    else:
        mu2 = (7 - x) / 3             # Sedang
    mu3 = max(min((x - 5) / 3, 1), 0)  # Banyak

    z1 = 5 * x + 40
    z2 = 7.5 * x + 40
    z3 = 10 * x + 20

    denom = mu1 + mu2 + mu3
    if denom == 0:
        return 0.0
    rpm = (mu1 * z1 + mu2 * z2 + mu3 * z3) / denom
    return max(0.0, min(rpm, 100.0))

# Setup Flask app
app = Flask(__name__)

# Function to generate video frames
def generate_frames():
    global last_update_time, window_max
    crossed_ids = set()  # Variabel untuk melacak objek yang melewati garis
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = time.time()

        # Preprocess frame untuk YOLOv5
        img = letterbox(frame, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).to(device)
        if half:
            img = img.half()
        img = img.unsqueeze(0)

        # Inference YOLOv5
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45)

        # Hitung current_in_box dan update window_max
        det_list = []
        current_in_box = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    det_list.append([x1, y1, x2, y2, conf])
                    if y2 > line_position:
                        current_in_box += 1
                    # Gambar bounding box untuk noda yang terdeteksi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}',
                                (x1, y1-10), font, 0.5, (0,255,0), 2)

        window_max = max(window_max, current_in_box)

        # Setiap 3 detik, perbarui display_count & display_rpm
        if t - last_update_time >= 3.0:
            display_count = window_max
            display_rpm = sugeno(display_count)  # Menghitung nilai RPM berdasarkan deteksi objek
            print(f"[3s Update] Noda max: {display_count}, RPM fuzzy: {display_rpm:.2f}%")
            window_max = 0
            last_update_time = t

            # Langsung set nilai PWM motor sesuai dengan display_rpm
            nilaiPWM.ChangeDutyCycle(display_rpm)  # Mengubah duty cycle PWM langsung sesuai display_rpm
            GPIO.output(LED, GPIO.HIGH)

        # Update tracker dan hitung total unik melewati garis
        dets_np = np.array(det_list, dtype=np.float32)
        tracks = tracker.update(dets_np) if dets_np.size else np.empty((0,5))
        for tr in tracks:
            tid = int(tr[4]); y2 = int(tr[3])
            if y2 > line_position and tid not in crossed_ids:
                crossed_ids.add(tid)
        total_crossed = len(crossed_ids)

        # Gambar garis merah dan kotak biru
        cv2.line(frame, (0, line_position), (width, line_position), (0,0,255), 2)
        cv2.rectangle(frame, (0, line_position), (width, height), (255,0,0), 2)

        # Overlay teks
        cv2.putText(frame, f'Noda melewati garis: {total_crossed}',
                    (10, 30), font, 1, (0,255,255), 2)
        cv2.putText(frame, f'Noda di dalam kotak: {display_count}',
                    (10, line_position + 30), font, 0.7, (255,255,255), 2)
        cv2.putText(frame, f'RPM (fuzzy): {display_rpm:.2f}%',
                    (10, line_position + 60), font, 0.7, (255,255,255), 2)

        # Encode frame sebagai JPEG dan kirim ke browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define the route for video streaming
@app.route('/')
def video():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask app
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Program dihentikan oleh pengguna.")
    finally:
        # Pastikan untuk menghentikan PWM dan membersihkan GPIO saat program dihentikan
        nilaiPWM.stop()  # Menghentikan PWM
        GPIO.cleanup()  # Membersihkan GPIO untuk memastikan tidak ada konfigurasi yang tersisa

