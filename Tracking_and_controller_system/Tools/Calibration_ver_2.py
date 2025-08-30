import cv2
import numpy as np

def load_calibration(npz_path):
    """
    Đọc file calibration lưu dưới dạng .npz và trả về các thông số quan trọng.
    
    Trả về:
        mtx: ma trận nội tại gốc của camera
        dist: hệ số méo
    """
    data = np.load(npz_path, allow_pickle=True)
    mtx = data['camera_matrix']
    dist = data['dist_coeffs']
    return mtx, dist

def get_optimal_new_camera_matrix(mtx, dist, image_size, alpha=1):
    """
    Tính ma trận camera tối ưu để hiệu chỉnh ảnh.
    
    Tham số:
        mtx: ma trận camera gốc
        dist: hệ số méo
        image_size: tuple (w, h)
        alpha: thông số điều chỉnh vùng nhìn (0: cắt nhiều, 1: giữ toàn bộ)
    
    Trả về:
        new_mtx: ma trận camera tối ưu
        roi: vùng crop ảnh hợp lệ (x, y, w, h)
    """
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, alpha, image_size)
    return new_mtx, roi

def get_undistort_map(mtx, dist, image_size, alpha=1):
    """
    Tạo bản đồ hiệu chỉnh ảnh méo sử dụng ma trận camera tối ưu.
    
    Trả về:
        map1, map2: bản đồ ánh xạ dùng với cv2.remap
        roi: vùng nên crop ảnh sau hiệu chỉnh
    """
    new_mtx, roi = get_optimal_new_camera_matrix(mtx, dist, image_size, alpha)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, image_size, cv2.CV_16SC2)
    return map1, map2, roi

def undistort_with_map(frame, map1, map2):
    """
    Áp dụng bản đồ hiệu chỉnh ảnh.
    """
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)