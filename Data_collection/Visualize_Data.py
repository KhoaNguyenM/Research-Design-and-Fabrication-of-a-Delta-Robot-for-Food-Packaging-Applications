import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# IMAGE_WIDTH_SET = 230
# IMAGE_HEIGHT_SET = 640

class YoloImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Viewer")
        self.root.geometry("1200x800")
        
        self.image_folder = ""
        self.label_folder = ""
        self.image_files = []
        self.current_index = 0
        self.class_names = []  # Danh sách tên các class
        
        # Frame cho các nút điều khiển
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Nút chọn folder ảnh
        self.btn_image_folder = tk.Button(self.control_frame, text="Chọn folder ảnh", command=self.select_image_folder)
        self.btn_image_folder.pack(side=tk.LEFT, padx=5)
        
        # Nút chọn folder label
        self.btn_label_folder = tk.Button(self.control_frame, text="Chọn folder label", command=self.select_label_folder)
        self.btn_label_folder.pack(side=tk.LEFT, padx=5)
        
        # Hiển thị đường dẫn folder ảnh
        self.lbl_image_folder = tk.Label(self.control_frame, text="Folder ảnh: Chưa chọn")
        self.lbl_image_folder.pack(side=tk.LEFT, padx=20)
        
        # Hiển thị đường dẫn folder label
        self.lbl_label_folder = tk.Label(self.control_frame, text="Folder label: Chưa chọn")
        self.lbl_label_folder.pack(side=tk.LEFT, padx=20)
        
        # Frame cho hiển thị ảnh
        self.image_frame = tk.Frame(root, bg="gray")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label hiển thị ảnh
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Label hiển thị tên ảnh
        self.lbl_filename = tk.Label(self.image_frame, text="", bg="gray", fg="white", font=("Arial", 12, "bold"))
        self.lbl_filename.pack(side=tk.BOTTOM, pady=5)
                
        # Frame hiển thị thông tin classes đã tải
        self.class_info_frame = tk.Frame(root)
        self.class_info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Label hiển thị thông tin classes
        self.lbl_class_info = tk.Label(self.class_info_frame, text="Classes: Chưa tải")
        self.lbl_class_info.pack(side=tk.LEFT, padx=5)
        
        # Frame cho các nút điều hướng
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Nút điều hướng
        self.btn_prev = tk.Button(self.nav_frame, text="Ảnh trước", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        # Hiển thị thông tin ảnh hiện tại
        self.lbl_image_info = tk.Label(self.nav_frame, text="0/0")
        self.lbl_image_info.pack(side=tk.LEFT, padx=20)
        
        self.btn_next = tk.Button(self.nav_frame, text="Ảnh tiếp theo", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.RIGHT, padx=5)
        
        # Biến lưu trữ ảnh hiện tại để tránh bị garbage collection
        self.current_photo = None
    
    def select_image_folder(self):
        folder_path = filedialog.askdirectory(title="Chọn folder chứa ảnh")
        if folder_path:
            self.image_folder = folder_path
            self.lbl_image_folder.config(text=f"Folder ảnh: {folder_path}")
            self.load_image_files()
    
    def select_label_folder(self):
        folder_path = filedialog.askdirectory(title="Chọn folder chứa label")
        if folder_path:
            self.label_folder = folder_path
            self.lbl_label_folder.config(text=f"Folder label: {folder_path}")
            
            # Đọc file classes.txt nếu có
            self.load_class_names()
            
            # Nếu đã có folder ảnh thì load ảnh đầu tiên
            if self.image_folder and self.image_files:
                self.show_current_image()
    
    def load_class_names(self):
        classes_path = os.path.join(self.label_folder, "classes.txt")
        if os.path.exists(classes_path):
            try:
                with open(classes_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                
                # Cập nhật thông tin các class đã tải
                class_text = ", ".join(self.class_names)
                self.lbl_class_info.config(text=f"Classes: {class_text}")
                messagebox.showinfo("Thông báo", f"Đã tải {len(self.class_names)} classes từ file classes.txt")
            except Exception as e:
                messagebox.showwarning("Cảnh báo", f"Lỗi khi đọc file classes.txt: {str(e)}")
                self.class_names = []
        else:
            messagebox.showinfo("Thông báo", "Không tìm thấy file classes.txt trong folder label")
            self.class_names = []
    
    def load_image_files(self):
        # Chỉ lấy các file ảnh có định dạng phổ biến
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = [f for f in os.listdir(self.image_folder) 
                            if os.path.isfile(os.path.join(self.image_folder, f)) and 
                            os.path.splitext(f)[1].lower() in valid_extensions]
        
        self.image_files.sort()  # Sắp xếp theo tên file
        
        if self.image_files:
            self.current_index = 0
            self.lbl_image_info.config(text=f"1/{len(self.image_files)}")
            self.btn_next.config(state=tk.NORMAL if len(self.image_files) > 1 else tk.DISABLED)
            self.btn_prev.config(state=tk.DISABLED)
            
            # Nếu đã có folder label thì hiển thị ảnh đầu tiên
            if self.label_folder:
                self.show_current_image()
        else:
            messagebox.showinfo("Thông báo", "Không tìm thấy file ảnh trong folder.")
    
    def get_class_name(self, class_id):
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"Class {class_id}"
    
    def show_current_image(self):
        if not self.image_files:
            return
        
        # Lấy tên file ảnh hiện tại
        image_filename = self.image_files[self.current_index]
        image_path = os.path.join(self.image_folder, image_filename)
        
        # Đọc ảnh với OpenCV
        img = cv2.imread(image_path)
        # img = cv2.resize(img, (IMAGE_WIDTH_SET, IMAGE_HEIGHT_SET), interpolation=cv2.INTER_AREA)
        if img is None:
            messagebox.showerror("Lỗi", f"Không thể đọc ảnh: {image_path}")
            return
        
        # Lấy kích thước ảnh
        height, width = img.shape[:2]
        
        # Tìm file label tương ứng
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_filename)
        
        # Đọc bbox từ file label nếu tồn tại
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Vẽ các bbox
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # format YOLO: class x_center y_center width height
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height
                        
                        # Tính toán tọa độ góc trên bên trái
                        x1 = int(x_center - bbox_width / 2)
                        y1 = int(y_center - bbox_height / 2)
                        x2 = int(x_center + bbox_width / 2)
                        y2 = int(y_center + bbox_height / 2)
                        
                        # Lấy tên class từ danh sách class_names
                        class_name = self.get_class_name(class_id)
                        
                        # Vẽ bbox
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Tạo nền cho text
                        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
                        
                        # Viết tên class
                        cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            except Exception as e:
                messagebox.showwarning("Cảnh báo", f"Lỗi khi đọc file label: {str(e)}")
        else:
            messagebox.showinfo("Thông báo", f"Không tìm thấy file label cho ảnh {image_filename}")
        
        # Chuyển đổi ảnh từ BGR sang RGB (OpenCV đọc ảnh ở dạng BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize ảnh để vừa với cửa sổ
        max_height = self.root.winfo_height() - 150  # Trừ đi phần điều khiển
        max_width = self.root.winfo_width() - 40     # Padding
        
        scale_factor = min(max_width / width, max_height / height)
        
        if scale_factor < 1:  # Chỉ resize nếu ảnh lớn hơn cửa sổ
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height))
        
        # Chuyển đổi thành đối tượng hình ảnh Tkinter
        img_pil = Image.fromarray(img_rgb)
        self.current_photo = ImageTk.PhotoImage(image=img_pil)
        
        # Hiển thị ảnh
        self.image_label.config(image=self.current_photo)

        self.lbl_filename.config(text=f"Tên ảnh: {image_filename}")
        
        # Cập nhật thông tin ảnh hiện tại
        self.lbl_image_info.config(text=f"{self.current_index + 1}/{len(self.image_files)}")
    
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.btn_prev.config(state=tk.NORMAL)
            if self.current_index == len(self.image_files) - 1:
                self.btn_next.config(state=tk.DISABLED)
            self.show_current_image()
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.btn_next.config(state=tk.NORMAL)
            if self.current_index == 0:
                self.btn_prev.config(state=tk.DISABLED)
            self.show_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloImageViewer(root)
    root.mainloop()