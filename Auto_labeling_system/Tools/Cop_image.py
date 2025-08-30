import numpy as np
import matplotlib.pyplot as plt

def crop_and_concatenate(image, rect1, rect2, axis='horizontal', show=False):
    """
    Cắt hai hình chữ nhật từ ảnh và ghép chúng lại.
    
    Parameters:
    - image_path: Đường dẫn đến ảnh
    - rect1, rect2: Mỗi hình là np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    - axis: 'horizontal' (mặc định) hoặc 'vertical' để ghép ảnh
    - show: nếu True sẽ hiển thị ảnh ghép

    Returns:
    - image_concat: mảng ảnh sau khi ghép
    """
    # Đọc ảnh và chuyển sang array
    # image = Image.open(image_path)
    # image = np.array(image)
    
    def crop_rectangle(img, rect):
        x_min, x_max = rect[:, 0].min(), rect[:, 0].max()
        y_min, y_max = rect[:, 1].min(), rect[:, 1].max()
        return img[y_min:y_max, x_min:x_max]

    # Cắt hai vùng
    crop1 = crop_rectangle(image, rect1)
    crop2 = crop_rectangle(image, rect2)

    # Ghép ảnh
    if axis == 'horizontal':
        image_concat = np.concatenate((crop1, crop2), axis=1)
    else:  # vertical
        image_concat = np.concatenate((crop1, crop2), axis=0)

    # Hiển thị nếu cần
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_concat)
        plt.axis('off')
        plt.title(f"Concatenated ({axis}) Image")
        plt.show()
    
    return image_concat