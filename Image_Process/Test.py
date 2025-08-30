import cv2

# Mở camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    # Đọc từng frame
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể nhận frame")
        break

    # Lấy kích thước khung hình
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Vẽ tâm: một đường ngang và một đường dọc cắt nhau
    color = (0, 255, 0)  # Màu xanh lá
    thickness = 2
    length = 20  # Độ dài của mỗi đoạn tâm

    # Vẽ đường ngang
    cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), color, thickness)
    # Vẽ đường dọc
    cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), color, thickness)

    # Hiển thị frame
    cv2.imshow('Camera', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()







