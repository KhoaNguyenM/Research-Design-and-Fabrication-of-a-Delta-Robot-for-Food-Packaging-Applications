import os
import cv2
import numpy as np
from PIL import Image
from Tools import crop_and_concatenate

# Hàm xử lý thư mục
def process_images_in_folder(input_folder, output_folder, rect1, rect2, axis='vertical'):
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_folder, exist_ok=True)

    # Duyệt qua tất cả file trong thư mục
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Đọc ảnh
            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                # Cắt và ghép ảnh
                result = crop_and_concatenate(image, rect1, rect2, axis)

                # Chuyển sang PIL để lưu
                result_img = Image.fromarray(result)
                result_img.save(output_path)

                print(f"✓ Đã xử lý: {filename}")
            except Exception as e:
                print(f"✗ Lỗi khi xử lý {filename}: {e}")

# Ví dụ sử dụng:
if __name__ == '__main__':
    input_folder = 'Data'    # Thư mục chứa ảnh gốc
    output_folder = 'Data_after_crop'  # Thư mục lưu ảnh đã xử lý
    os.makedirs(output_folder, exist_ok=True)

    # Định nghĩa rect1 và rect2 (4 điểm theo thứ tự np.array([[x1, y1], ..., [x4, y4]]))
    crop1 =  np.array([[46, 3], [271, 3], [271, 355], [46, 355]])
    crop2 = np.array([[425, 6], [562, 6], [562, 358], [425, 358]])

    process_images_in_folder(input_folder, output_folder, crop1, crop2, axis='horizontal')

















