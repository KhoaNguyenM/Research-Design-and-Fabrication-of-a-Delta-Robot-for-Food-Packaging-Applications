import numpy as np

def pixel_to_world(x_pixel, y_pixel, H):
    # Đảo ngược ma trận H để có được H_inverse (thế giới thực -> ảnh)
    H_inverse = np.linalg.inv(H)
    
    # Tạo điểm pixel dạng homogeneous
    point_pixel = np.array([x_pixel, y_pixel, 1], dtype=np.float32)
    
    # Áp dụng H_inverse để có tọa độ thực
    point_world = H_inverse @ point_pixel
    
    # Chuẩn hóa tọa độ
    point_world /= point_world[2]
    
    # Trả về tọa độ x, y trong thế giới thực
    return point_world[:2]