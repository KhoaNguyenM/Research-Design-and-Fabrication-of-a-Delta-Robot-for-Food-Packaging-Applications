import cv2
import numpy as np
import os
from functools import lru_cache
from typing import Optional, Tuple

#########################################################################
# HÀM SỬ DỤNG ĐỂ HIỆU CHỈNH BIẾN DẠNG CAMERA TRONG THỜI GIAN THỰC

class UndistortedCamera:
    """
    Lớp camera có hiệu chỉnh biến dạng, với giao diện sử dụng tương tự như cv2.VideoCapture
    """
    def __init__(self, camera_id=0, calibration_file=None):
        """
        Khởi tạo camera với khả năng hiệu chỉnh biến dạng
        
        Parameters:
        camera_id (int): ID của camera (mặc định: 0 - camera tích hợp)
        calibration_file (str): Đường dẫn đến file calibration (bắt buộc)
        """
        # Mở camera
        self.cap = cv2.VideoCapture(camera_id)
        
        # Kiểm tra camera có mở được không
        if not self.cap.isOpened():
            print("Không thể mở camera!")
            return None
        
        # Đọc tham số hiệu chỉnh biến dạng từ file (bắt buộc)
        if not calibration_file:
            print("Lỗi: Thiếu file hiệu chuẩn camera.")
            print("Cần cung cấp đường dẫn đến file hiệu chuẩn chứa camera_matrix và dist_coeffs.")
            return None
        
        try:
            calib_data = np.load(calibration_file)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            print("Đã đọc thành công các tham số hiệu chuẩn từ file.")
        except Exception as e:
            print(f"Lỗi khi đọc file hiệu chuẩn: {e}")
            print("Vui lòng cung cấp file hiệu chuẩn đúng định dạng, có chứa 'camera_matrix' và 'dist_coeffs'.")
            return None
        
        # Lấy kích thước frame
        ret, frame = self.cap.read()
        if ret:
            self.height, self.width = frame.shape[:2]
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (self.width, self.height), 1, (self.width, self.height)
            )
            # Đặt lại con trỏ đầu video stream
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Không thể đọc frame từ camera để khởi tạo ma trận hiệu chỉnh.")
            return None
    
    def read(self):
        """
        Đọc một frame từ camera và áp dụng hiệu chỉnh biến dạng
        
        Returns:
        tuple: (ret, frame) - ret là True nếu đọc thành công, frame là khung hình đã hiệu chỉnh
        """
        # Đọc frame từ camera
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        # Áp dụng hiệu chỉnh biến dạng
        undistorted = cv2.undistort(
            frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix
        )
        
        # Cắt ROI nếu cần
        x, y, w, h = self.roi
        if all(val > 0 for val in [x, y, w, h]):
            undistorted = undistorted[y:y+h, x:x+w]
        
        return True, undistorted
    
    def isOpened(self):
        """
        Kiểm tra camera có đang mở không
        
        Returns:
        bool: True nếu camera đang mở
        """
        return self.cap.isOpened()
    
    def release(self):
        """Giải phóng tài nguyên camera"""
        self.cap.release()
    
    def get(self, propId):
        """
        Lấy thuộc tính của camera
        
        Parameters:
        propId: ID thuộc tính (từ cv2.CAP_PROP_*)
        
        Returns:
        Giá trị thuộc tính
        """
        return self.cap.get(propId)
    
    def set(self, propId, value):
        """
        Thiết lập thuộc tính cho camera
        
        Parameters:
        propId: ID thuộc tính (từ cv2.CAP_PROP_*)
        value: Giá trị muốn thiết lập
        
        Returns:
        bool: True nếu thiết lập thành công
        """
        return self.cap.set(propId, value)


class CameraCalibrationLib:
    """
    Thư viện xử lý camera calibration với memoization tối ưu
    """
    
    def __init__(self, calibration_file: str = 'camera_calibration.npz'):
        """
        Khởi tạo thư viện calibration
        
        Args:
            calibration_file: Đường dẫn đến file calibration .npz
        """
        self.calibration_file = calibration_file
        self._calibration_data = None
        self._load_calibration()
    
    def _load_calibration(self) -> None:
        """
        Load dữ liệu calibration từ file npz với caching
        """
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Không tìm thấy file calibration: {self.calibration_file}")
        
        try:
            data = np.load(self.calibration_file)
            self._calibration_data = {
                'camera_matrix': data['camera_matrix'],
                'dist_coeffs': data['dist_coeffs'],
                'rvecs': data.get('rvecs', None),
                'tvecs': data.get('tvecs', None),
                'reprojection_error': data.get('reprojection_error', None)
            }
        except Exception as e:
            raise ValueError(f"Lỗi khi load file calibration: {e}")
    
    @lru_cache(maxsize=1)
    def _get_optimal_camera_matrix(self, img_width: int, img_height: int, 
                                   alpha: float = 0.0) -> np.ndarray:
        """
        Tính toán optimal camera matrix với memoization
        
        Args:
            img_width: Chiều rộng ảnh
            img_height: Chiều cao ảnh  
            alpha: Tham số free scaling (0.0 = crop, 1.0 = no crop)
        
        Returns:
            Optimal camera matrix
        """
        return cv2.getOptimalNewCameraMatrix(
            self._calibration_data['camera_matrix'],
            self._calibration_data['dist_coeffs'],
            (img_width, img_height),
            alpha
        )[0]
    
    @lru_cache(maxsize=1)
    def _get_undistort_maps(self, img_width: int, img_height: int, 
                           alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo undistortion maps với memoization
        
        Args:
            img_width: Chiều rộng ảnh
            img_height: Chiều cao ảnh
            alpha: Tham số free scaling
            
        Returns:
            Tuple (map1, map2) cho remap
        """
        optimal_matrix = self._get_optimal_camera_matrix(img_width, img_height, alpha)
        
        map1, map2 = cv2.initUndistortRectifyMap(
            self._calibration_data['camera_matrix'],
            self._calibration_data['dist_coeffs'],
            None,
            optimal_matrix,
            (img_width, img_height),
            cv2.CV_16SC2
        )
        
        return map1, map2
    
    def undistort_frame(self, frame: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """
        Hàm chính để undistort frame với tối ưu hóa tối đa
        
        Args:
            frame: Frame ảnh từ camera (BGR format)
            alpha: Tham số free scaling (0.0 = crop black pixels, 1.0 = keep all pixels)
                  - 0.0: Cắt bỏ các pixel đen, ảnh nhỏ hơn nhưng không có vùng đen
                  - 1.0: Giữ tất cả pixel, ảnh đầy đủ nhưng có thể có vùng đen ở góc
        
        Returns:
            Frame đã được undistort
        """
        if self._calibration_data is None:
            raise ValueError("Chưa load được dữ liệu calibration")
        
        if frame is None or frame.size == 0:
            raise ValueError("Frame không hợp lệ")
        
        img_height, img_width = frame.shape[:2]
        
        # Sử dụng cached maps để remap (tối ưu nhất)
        map1, map2 = self._get_undistort_maps(img_width, img_height, alpha)
        
        # Remap frame với interpolation tối ưu
        undistorted_frame = cv2.remap(
            frame, map1, map2, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return undistorted_frame
    
    def get_calibration_info(self) -> dict:
        """
        Lấy thông tin calibration
        
        Returns:
            Dictionary chứa thông tin calibration
        """
        if self._calibration_data is None:
            return {}
        
        return {
            'camera_matrix': self._calibration_data['camera_matrix'].tolist(),
            'distortion_coefficients': self._calibration_data['dist_coeffs'].tolist(),
            'reprojection_error': float(self._calibration_data['reprojection_error']) if self._calibration_data['reprojection_error'] is not None else None
        }
    
    def clear_cache(self) -> None:
        """
        Xóa cache để tiết kiệm bộ nhớ khi cần
        """
        self._get_optimal_camera_matrix.cache_clear()
        self._get_undistort_maps.cache_clear()

# Tạo instance global để sử dụng như singleton
_calibration_lib = None

def initialize_calibration(calibration_file: str = 'camera_calibration.npz') -> None:
    """
    Khởi tạo thư viện calibration
    
    Args:
        calibration_file: Đường dẫn đến file calibration
    """
    global _calibration_lib
    _calibration_lib = CameraCalibrationLib(calibration_file)

def undistort_frame(frame: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """
    Hàm chính để undistort frame - Interface đơn giản để import
    
    Args:
        frame: Frame ảnh từ camera
        alpha: Tham số free scaling (0.0-1.0)
        
    Returns:
        Frame đã được undistort
        
    Example:
        from camera_calibration_lib import undistort_frame, initialize_calibration
        
        # Khởi tạo một lần duy nhất
        initialize_calibration('camera_calibration.npz')
        
        # Sử dụng trong loop
        ret, frame = cap.read()
        if ret:
            corrected_frame = undistort_frame(frame)
    """
    global _calibration_lib
    
    if _calibration_lib is None:
        # Auto-initialize với file default
        initialize_calibration()
    
    return _calibration_lib.undistort_frame(frame, alpha)

def get_calibration_info() -> dict:
    """
    Lấy thông tin calibration
    
    Returns:
        Dictionary chứa thông tin calibration
    """
    global _calibration_lib
    
    if _calibration_lib is None:
        initialize_calibration()
    
    return _calibration_lib.get_calibration_info()

def clear_calibration_cache() -> None:
    """
    Xóa cache calibration
    """
    global _calibration_lib
    
    if _calibration_lib is not None:
        _calibration_lib.clear_cache()


# Ví dụ sử dụng:
if __name__ == "__main__":
    # Hàm này yêu cầu file hiệu chuẩn
    camera = UndistortedCamera(camera_id=0, calibration_file='Tools\\camera_calibration.npz')
    
    if camera and camera.isOpened():
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("Không thể nhận frame từ camera. Đang thoát...")
                break
            
            # Hiển thị frame đã hiệu chỉnh
            cv2.imshow('Camera da hieu chuan bien dang', frame)
            
            # Thoát nếu nhấn phím 'q'
            if cv2.waitKey(1) == ord('q'):
                break
        
        # Giải phóng tài nguyên
        camera.release()
        cv2.destroyAllWindows()



################################################################################################################################