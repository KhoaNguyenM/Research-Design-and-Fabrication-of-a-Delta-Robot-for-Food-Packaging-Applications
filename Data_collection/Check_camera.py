import cv2
import os
import time
import random
import string
from datetime import datetime

# Tạo thư mục lưu ảnh nếu chưa có
save_folder = "Data_ver4_banhnho"
os.makedirs(save_folder, exist_ok=True)

# Hàm tạo hậu tố ngẫu nhiên cho tên file
def random_suffix(length=2):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Khởi động camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

print("🎥 Mở camera thành công.")
print("⏎ Nhấn SPACE để bắt đầu/dừng chụp liên tục (10 ảnh/giây).")
print("⌨️ Nhấn Q để thoát.")

shooting = False          # Trạng thái chụp ảnh
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể nhận frame từ camera.")
        break

    # Hiển thị hình ảnh từ camera
    cv2.imshow('Camera', frame)

    # Đọc phím nhấn
    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        print(f"🔑 Phím nhấn: {key}")

    if key == ord('q'):
        print("👋 Thoát chương trình.")
        break

    elif key == 32:  # SPACE
        shooting = not shooting
        if shooting:
            print("📸 → Bắt đầu chụp liên tục 10 ảnh/giây...")
        else:
            print("⏸️ → Dừng chụp.")

    # Nếu đang ở chế độ chụp liên tục
    if shooting:
        current_time = time.time()
        # Nếu đã đến thời điểm chụp ảnh tiếp theo
        if current_time - last_capture_time >= 0.2:  # 0.1 giây = 10 ảnh/giây
            timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]
            suffix = random_suffix()
            filename = f"{timestamp}_{suffix}.jpg"
            filepath = os.path.join(save_folder, filename)

            success = cv2.imwrite(filepath, frame)
            if success:
                print(f"✅ Đã lưu ảnh: {filepath}")
            else:
                print(f"❌ Ghi ảnh thất bại: {filepath}")

            last_capture_time = current_time

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
