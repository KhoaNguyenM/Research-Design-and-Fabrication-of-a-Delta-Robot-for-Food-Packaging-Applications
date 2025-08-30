import cv2
import numpy as np

def undistort_realtime_camera(calibration_file, camera_id=0):
    """
    Khử méo hình cho camera theo thời gian thực
    
    Parameters:
    calibration_file (str): Đường dẫn đến file chứa thông số hiệu chuẩn
    camera_id (int): ID của camera (0 cho camera mặc định)
    
    Returns:
    None
    """
    # Tải thông số hiệu chuẩn
    try:
        data = np.load(calibration_file)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Đã tải thông số hiệu chuẩn camera thành công")
    except Exception as e:
        print(f"Lỗi: Không thể đọc file hiệu chuẩn: {e}")
        return
    
    # Khởi tạo camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera {camera_id}")
        return
    
    # Lấy kích thước khung hình
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc khung hình từ camera")
        cap.release()
        return
    
    h, w = frame.shape[:2]
    
    # Tính ma trận camera mới để tối ưu hóa kết quả
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Tính sẵn bản đồ méo hình để tăng tốc độ xử lý
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, newcameramtx, (w, h), cv2.CV_32FC1)
    
    print("Đang chạy camera với khử méo hình theo thời gian thực...")
    print("Nhấn 'q' để thoát, 's' để lưu ảnh, 'd' để chuyển đổi chế độ hiển thị")
    
    # Biến để theo dõi chế độ hiển thị
    show_comparison = True
    
    while True:
        # Đọc khung hình
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ camera")
            break
        
        # Khử méo hình sử dụng bản đồ đã tính sẵn (nhanh hơn)
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # Cắt ROI (tùy chọn)
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            undistorted_cropped = undistorted[y:y+h, x:x+w]
        else:
            undistorted_cropped = undistorted
        
        # Hiển thị kết quả
        if show_comparison:
            # Hiển thị so sánh cạnh nhau
            comparison = np.hstack((frame, undistorted))
            cv2.putText(comparison, "Goc (Original)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(comparison, "Da khu meo (Undistorted)", (frame.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Camera - Original vs Undistorted", comparison)
        else:
            # Hiển thị chỉ ảnh đã khử méo
            cv2.putText(undistorted_cropped, "Da khu meo (Undistorted)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Camera - Undistorted", undistorted_cropped)
        
        # Xử lý phím nhấn
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' để thoát
        if key == ord('q'):
            break
        # 's' để lưu ảnh
        elif key == ord('s'):
            timestamp = cv2.getTickCount()
            cv2.imwrite(f"undistorted_{timestamp}.jpg", undistorted_cropped)
            cv2.imwrite(f"original_{timestamp}.jpg", frame)
            print(f"Đã lưu ảnh gốc và ảnh đã khử méo với timestamp {timestamp}")
        # 'd' để chuyển đổi chế độ hiển thị
        elif key == ord('d'):
            show_comparison = not show_comparison
            cv2.destroyAllWindows()
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng camera")

# Sử dụng hàm
if __name__ == "__main__":
    calibration_file = "camera_calibration.npz"  # Đường dẫn đến file thông số hiệu chuẩn
    camera_id = 0  # Camera mặc định, thay đổi nếu cần
    
    undistort_realtime_camera(calibration_file, camera_id)