import os
import shutil
import random
from pathlib import Path
import yaml

def create_yolo_dataset(data_folder, labels_folder, dataset_folder, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    Tạo dataset YOLO với việc tách dữ liệu theo tỉ lệ mong muốn
    
    Args:
        data_folder (str): Đường dẫn đến folder chứa ảnh
        labels_folder (str): Đường dẫn đến folder chứa labels
        dataset_folder (str): Đường dẫn đến folder dataset đích
        train_ratio (float): Tỉ lệ dữ liệu train (mặc định 0.7)
        valid_ratio (float): Tỉ lệ dữ liệu validation (mặc định 0.2)
        test_ratio (float): Tỉ lệ dữ liệu test (mặc định 0.1)
    """
    
    # Kiểm tra tỉ lệ
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Tổng tỉ lệ train + valid + test phải bằng 1.0")
    
    # Tạo đường dẫn Path
    data_path = Path(data_folder)
    labels_path = Path(labels_folder)
    dataset_path = Path(dataset_folder)
    
    # Kiểm tra folder tồn tại
    if not data_path.exists():
        raise FileNotFoundError(f"Folder Data không tồn tại: {data_folder}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Folder Labels không tồn tại: {labels_folder}")
    
    # Tạo cấu trúc thư mục dataset
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    test_images = dataset_path / "test" / "images"
    test_labels = dataset_path / "test" / "labels"
    
    # Tạo các thư mục
    for folder in [train_images, train_labels, valid_images, valid_labels, test_images, test_labels]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Đọc danh sách classes từ classes.txt
    classes_file = labels_path / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"File classes.txt không tồn tại trong folder labels: {classes_file}")
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Đã đọc {len(classes)} classes: {classes}")
    
    # Lấy danh sách tất cả ảnh trong folder Data
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f"*{ext}")))
        image_files.extend(list(data_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        raise FileNotFoundError("Không tìm thấy ảnh nào trong folder Data")
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Lọc ảnh có label tương ứng
    valid_images_list = []
    for img_file in image_files:
        label_file = labels_path / (img_file.stem + '.txt')
        if label_file.exists():
            valid_images_list.append(img_file)
        else:
            print(f"Cảnh báo: Không tìm thấy label cho ảnh {img_file.name}")
    
    if not valid_images_list:
        raise FileNotFoundError("Không tìm thấy cặp ảnh-label nào hợp lệ")
    
    print(f"Có {len(valid_images_list)} cặp ảnh-label hợp lệ")
    
    # Trộn ngẫu nhiên danh sách
    random.shuffle(valid_images_list)
    
    # Tính toán số lượng cho mỗi tập
    total_count = len(valid_images_list)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count
    
    print(f"Phân chia: Train={train_count}, Valid={valid_count}, Test={test_count}")
    
    # Chia dữ liệu
    train_files = valid_images_list[:train_count]
    valid_files = valid_images_list[train_count:train_count + valid_count]
    test_files = valid_images_list[train_count + valid_count:]
    
    # Copy files
    def copy_files(file_list, img_dest, label_dest, set_name):
        print(f"Đang copy {len(file_list)} files cho {set_name}...")
        for img_file in file_list:
            # Copy ảnh
            shutil.copy2(img_file, img_dest / img_file.name)
            
            # Copy label
            label_file = labels_path / (img_file.stem + '.txt')
            shutil.copy2(label_file, label_dest / label_file.name)
        print(f"Hoàn thành copy {set_name}")
    
    # Copy files cho từng tập
    copy_files(train_files, train_images, train_labels, "Train")
    copy_files(valid_files, valid_images, valid_labels, "Valid")
    copy_files(test_files, test_images, test_labels, "Test")
    
    # Tạo file YAML với format tùy chỉnh
    yaml_file = dataset_path / 'data.yaml'
    
    with open(yaml_file, 'w', encoding='utf-8') as f:
        # f.write(f"path: {dataset_path.absolute()}\n")
        f.write("train: ./train/images # train images\n")
        f.write("val: ./valid/images # val images\n") 
        f.write("test: ./test/images # test images\n")
        f.write(f"\nnc: {len(classes)}\n")
        f.write("\n# Classes\n")
        f.write("names:\n")
        for i, class_name in enumerate(classes):
            f.write(f"  {i}: {class_name}\n")
    
    print(f"Đã tạo file YAML: {yaml_file}")
    print("Hoàn thành tạo dataset YOLO!")
    
    # In thống kê
    print("\n=== THỐNG KÊ DATASET ===")
    print(f"Tổng số ảnh: {total_count}")
    print(f"Train: {train_count} ảnh ({train_ratio*100:.1f}%)")
    print(f"Valid: {valid_count} ảnh ({valid_ratio*100:.1f}%)")
    print(f"Test: {test_count} ảnh ({test_ratio*100:.1f}%)")
    print(f"Số classes: {len(classes)}")
    print(f"Dataset được lưu tại: {dataset_path.absolute()}")

def main():
    """
    Hàm main để chạy chương trình
    """
    print("=== YOLO Dataset Splitter ===")
    
    # Nhập đường dẫn từ người dùng
    data_folder = input("Nhập đường dẫn đến folder Data: ").strip()
    labels_folder = input("Nhập đường dẫn đến folder Labels: ").strip()
    dataset_folder = input("Nhập đường dẫn đến folder Dataset (sẽ được tạo): ").strip()
    
    # Nhập tỉ lệ phân chia
    print("\nNhập tỉ lệ phân chia (ví dụ: 0.7 0.2 0.1):")
    ratios_input = input("Train Valid Test (cách nhau bởi dấu cách): ").strip()
    
    try:
        ratios = list(map(float, ratios_input.split()))
        if len(ratios) != 3:
            raise ValueError("Cần nhập đúng 3 giá trị")
        train_ratio, valid_ratio, test_ratio = ratios
    except:
        print("Sử dụng tỉ lệ mặc định: 0.7 0.2 0.1")
        train_ratio, valid_ratio, test_ratio = 0.7, 0.2, 0.1
    
    try:
        create_yolo_dataset(
            data_folder=data_folder,
            labels_folder=labels_folder,
            dataset_folder=dataset_folder,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()