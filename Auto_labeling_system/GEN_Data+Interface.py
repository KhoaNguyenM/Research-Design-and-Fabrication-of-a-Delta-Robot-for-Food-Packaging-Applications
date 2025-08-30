import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import argparse
from collections import Counter
from threading import Thread
from tkinter import PhotoImage
from PIL import Image, ImageTk
import colorsys

from Tools import remove_background_mask, filter_masks_by_squareness, process_masks
from Tools import label_masks_from_examples, compare_and_label_mask, extract_mask_features
from Tools import crop_and_concatenate

# Define global constants
MODEL_CFG = "D:/UTE/UTE_Nam_4_ki_2_DATN/Model&code_SAM/Model_SAM2_JinsuaFeito/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = "D:/UTE/UTE_Nam_4_ki_2_DATN/Model&code_SAM/Model_SAM2_JinsuaFeito/sam2.1_hiera_base_plus.pt"

# crop1 = np.array([[79, 4], [264, 4], [264, 477], [79, 477]])
# crop2 =   np.array([[443, 4], [574, 4], [574, 477], [443, 477]])

crop1 = np.array([[45, 8], [272, 8], [272, 359], [45, 359]])
crop2 =  np.array([[427, 8], [575, 8], [575, 359], [427, 359]])

# crop1 =  np.array([[28, 5], [256, 5], [256, 474], [28, 474]])
# crop2 =  np.array([[458, 5], [615, 5], [615, 474], [458, 474]])

BOX_NMS_THRESH_SET = 0.7
PRED_IOU_THRESH_SET = 0.7
STABILITY_SCORE_THRESH_SET = 0.7

MAX_AREA_THRESHOLD_SET = 11000 # 12000
MIN_AREA_THRESHOLD_SET = 4500 # 2485 

MIN_SQUARNESS_RATIO = 0.9  # Minimum squareness ratio for masks to be considered valid

ADD_AREA = 20000

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_white_background(image_path, bg_color=(231,244,255)):  # bg_color là tuple RGB của #E7F4FF
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        # Nếu pixel trắng gần tuyệt đối thì thay bằng màu nền
        if item[0] > 230 and item[1] > 230 and item[2] > 230:
            newData.append((bg_color[0], bg_color[1], bg_color[2], 255))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

class LabelingWindow:
    def __init__(self, parent, image, masks, callback):
        self.parent = parent
        self.image = image
        self.masks = masks
        self.callback = callback
        self.labels = {}
        self.current_mask_index = 0
        
        # Create new window
        self.window = tk.Toplevel(parent)
        self.window.title("Mask Labeling")
        self.window.geometry("900x700")
        self.window.configure(bg="#70B7FF")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.setup_ui()
        self.display_current_mask()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.window, bg="#70B7FF")
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Title
        title = tk.Label(
            main_frame,
            text="GÁN NHÃN CHO CÁC ĐỐI TƯỢNG ĐƯỢC PHÁT HIỆN",
            font=("Cambria", 22, "bold"),
            bg="#70B7FF",
            fg="#B21A1A"
        )
        title.pack(pady=(0, 20))
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg="#70B7FF")
        content_frame.pack(fill='both', expand=True)
        
        # Left panel for controls
        left_panel = tk.Frame(content_frame, bg="#70B7FF", width=250)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Right panel for image
        right_panel = tk.Frame(content_frame, bg="#70B7FF")
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Setup left panel
        self.setup_left_panel(left_panel)
        
        # Setup right panel
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        # Progress info
        self.progress_label = tk.Label(
            parent,
            text=f"Đối tượng 1 / {len(self.masks)}",
            font=("Cambria", 14, "bold"),
            bg="#70B7FF",
            fg="#222"
        )
        self.progress_label.pack(pady=(0, 20))
        
        # Current object info
        info_frame = tk.LabelFrame(
            parent,
            text="Thông tin đối tượng hiện tại",
            font=("Cambria", 12, "bold"),
            bg="#70B7FF",
            fg="#222",
            padx=10,
            pady=10
        )
        info_frame.pack(fill='x', pady=(0, 20))
        
        self.object_info = tk.Label(
            info_frame,
            text="",
            font=("Cambria", 10),
            bg="#70B7FF",
            fg="#444",
            justify='left'
        )
        self.object_info.pack()
        
        # Label input
        input_frame = tk.LabelFrame(
            parent,
            text="Nhập nhãn",
            font=("Cambria", 12, "bold"),
            bg="#70B7FF",
            fg="#222",
            padx=10,
            pady=10
        )
        input_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(
            input_frame,
            text="Tên nhãn:",
            font=("Cambria", 11),
            bg="#70B7FF",
            fg="#222"
        ).pack(anchor='w')
        
        self.label_entry = tk.Entry(
            input_frame,
            font=("Cambria", 12),
            width=25
        )
        self.label_entry.pack(fill='x', pady=(5, 10))
        self.label_entry.bind('<Return>', lambda e: self.next_mask())
        
        # Buttons
        button_frame = tk.Frame(input_frame, bg="#70B7FF")
        button_frame.pack(fill='x')
        
        self.prev_btn = tk.Button(
            button_frame,
            text="◀ Trước",
            font=("Arial", 10),
            command=self.prev_mask,
            width=8
        )
        self.prev_btn.pack(side='left', padx=(0, 5))
        
        self.next_btn = tk.Button(
            button_frame,
            text="Tiếp ▶",
            font=("Arial", 10),
            command=self.next_mask,
            width=8
        )
        self.next_btn.pack(side='right', padx=(5, 0))
        
        # Labels list
        list_frame = tk.LabelFrame(
            parent,
            text="Nhãn đã gán",
            font=("Cambria", 12, "bold"),
            bg="#70B7FF",
            fg="#222",
            padx=10,
            pady=10
        )
        list_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Scrollable listbox
        scroll_frame = tk.Frame(list_frame, bg="#70B7FF")
        scroll_frame.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.labels_listbox = tk.Listbox(
            scroll_frame,
            font=("Cambria", 10),
            yscrollcommand=scrollbar.set
        )
        self.labels_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.labels_listbox.yview)
        
        # Final buttons
        final_frame = tk.Frame(parent, bg="#70B7FF")
        final_frame.pack(fill='x', pady=(10, 0))
        
        tk.Button(
            final_frame,
            text="Hoàn thành",
            font=("Cambria", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.finish_labeling,
            width=12
        ).pack(side='right', pady=(0, 5))
        
        tk.Button(
            final_frame,
            text="Hủy",
            font=("Cambria", 12, "bold"),
            bg="#f44336",
            fg="white",
            command=self.cancel_labeling,
            width=12
        ).pack(side='left', pady=(0, 5))
        
    def setup_right_panel(self, parent):
        # Image display frame
        self.image_frame = tk.Frame(parent, bg="#70B7FF", relief='sunken', bd=2)
        self.image_frame.pack(fill='both', expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(
            self.image_frame,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_mask_display_image(self, highlight_index=None):
        """Create image with all masks displayed, highlighting current mask"""
        # Create a copy of the original image
        result_image = self.image.copy()
        
        # Draw all masks
        for i, mask_dict in enumerate(self.masks):
            mask = mask_dict['segmentation']
            
            # Create color for this mask
            if i == highlight_index:
                # Highlight current mask with bright color
                color = (255, 0, 0)  # Red for current mask
                thickness = 3
            else:
                # Other masks with muted colors
                hue = i / max(1, len(self.masks))
                rgb = colorsys.hsv_to_rgb(hue, 0.6, 0.6)
                color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                thickness = 2
            
            # Find and draw contours
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result_image, contours, -1, color, thickness)
            
            # Add mask number
            x, y, w, h = mask_dict['bbox']
            font_scale = 0.7 if i == highlight_index else 0.5
            font_thickness = 2 if i == highlight_index else 1
            
            cv2.putText(
                result_image, 
                str(i), 
                (int(x), int(y) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                color, 
                font_thickness
            )
        
        return result_image
        
    def display_current_mask(self):
        """Display image with current mask highlighted"""
        # Update progress label
        self.progress_label.config(text=f"Đối tượng {self.current_mask_index + 1} / {len(self.masks)}")
        
        # Update object info
        if self.current_mask_index < len(self.masks):
            mask_dict = self.masks[self.current_mask_index]
            x, y, w, h = mask_dict['bbox']
            area = mask_dict.get('area', w * h)
            info_text = f"Vị trí: ({int(x)}, {int(y)})\nKích thước: {int(w)} x {int(h)}\nDiện tích: {int(area)}"
            self.object_info.config(text=info_text)
        
        # Create display image
        display_image = self.create_mask_display_image(self.current_mask_index)
        
        # Convert to PIL Image and display on canvas
        pil_image = Image.fromarray(display_image)
        
        # Calculate size to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            # Calculate scaling to fit image in canvas while maintaining aspect ratio
            img_width, img_height = pil_image.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width()//2,
            self.canvas.winfo_height()//2,
            image=self.photo,
            anchor='center'
        )
        
        # Update entry field with existing label if any
        if self.current_mask_index in self.labels:
            self.label_entry.delete(0, tk.END)
            self.label_entry.insert(0, self.labels[self.current_mask_index])
        else:
            self.label_entry.delete(0, tk.END)
        
        # Focus on entry field
        self.label_entry.focus_set()
        
        # Update buttons state
        self.prev_btn.config(state='normal' if self.current_mask_index > 0 else 'disabled')
        self.next_btn.config(text="Tiếp ▶" if self.current_mask_index < len(self.masks) - 1 else "Hoàn thành")
        
        # Update window to ensure canvas size is correct
        self.window.update_idletasks()
        
    def save_current_label(self):
        """Save the current label"""
        label = self.label_entry.get().strip()
        if label:
            self.labels[self.current_mask_index] = label
            self.update_labels_list()
            
    def update_labels_list(self):
        """Update the labels listbox"""
        self.labels_listbox.delete(0, tk.END)
        for i in range(len(self.masks)):
            label = self.labels.get(i, "")
            status = f"✓" if label else "○"
            display_text = f"{status} Đối tượng {i}: {label if label else '(chưa gán)'}"
            self.labels_listbox.insert(tk.END, display_text)
            
    def prev_mask(self):
        """Go to previous mask"""
        self.save_current_label()
        if self.current_mask_index > 0:
            self.current_mask_index -= 1
            self.display_current_mask()
            
    def next_mask(self):
        """Go to next mask or finish if at the end"""
        self.save_current_label()
        if self.current_mask_index < len(self.masks) - 1:
            self.current_mask_index += 1
            self.display_current_mask()
        else:
            self.finish_labeling()
            
    def finish_labeling(self):
        """Finish labeling and return results"""
        self.save_current_label()
        
        # Check if all masks have labels
        unlabeled = []
        for i in range(len(self.masks)):
            if i not in self.labels or not self.labels[i].strip():
                unlabeled.append(i)
                
        if unlabeled:
            response = tk.messagebox.askyesno(
                "Cảnh báo",
                f"Có {len(unlabeled)} đối tượng chưa được gán nhãn.\n"
                "Bạn có muốn tiếp tục không?\n"
                "(Các đối tượng chưa gán nhãn sẽ được đặt tên mặc định)",
                parent=self.window
            )
            if not response:
                return
                
            # Assign default labels to unlabeled masks
            for i in unlabeled:
                self.labels[i] = f"object{i}"
        
        # Get unique class labels
        class_labels = set(self.labels.values())
        
        self.window.destroy()
        self.callback(self.labels, class_labels)
        
    def cancel_labeling(self):
        """Cancel labeling"""
        response = tk.messagebox.askyesno(
            "Xác nhận",
            "Bạn có chắc chắn muốn hủy? Tất cả nhãn đã gán sẽ bị mất.",
            parent=self.window
        )
        if response:
            self.window.destroy()
            self.callback(None, None)

def run_gui_app():
    def on_start():
        start_btn['state'] = 'disabled'
        status_var.set("Đang chọn thư mục...")
        root.update_idletasks()

        folder_path = select_folder()
        if not folder_path:
            status_var.set("Không có thư mục nào được chọn.")
            start_btn['state'] = 'normal'
            return

        status_var.set(f"Thư mục đã chọn: {folder_path}\nĐang kiểm tra tệp ảnh...")
        root.update_idletasks()

        image_files = get_image_files(folder_path)
        if not image_files:
            status_var.set("Không tìm thấy ảnh trong thư mục này.")
            start_btn['state'] = 'normal'
            return
        
        FOLDER_LABELS = folder_path

        labels_folder = os.path.join(FOLDER_LABELS, "labels")

        def process_thread():
            try:
                status_var.set("Bắt đầu xử lý ảnh...")
                process_images(image_files, labels_folder, root)  # Pass root to process_images
                status_var.set("Xử lý hoàn tất! Xem terminal để biết thêm chi tiết.")
            except Exception as e:
                status_var.set(f"Lỗi: {e}")
            finally:
                start_btn['state'] = 'normal'

        Thread(target=process_thread, daemon=True).start()

    # Tạo cửa sổ chính
    root = tk.Tk()
    root.title("Mask Generator - YOLO Label Tool")
    root.geometry("850x500")
    root.resizable(False, False)
    root.configure(bg="#CBE1FF")

    # THÊM DÒNG NÀY:
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy(), exit()))

    # ==== Header: logo + info bên trái, tiêu đề giữa trên cùng ====
    header_frame = tk.Frame(root, bg="#CBE1FF")
    header_frame.pack(fill='x', pady=(0, 10))

    # --- Cột trái: logo + info ---
    left_frame = tk.Frame(header_frame, bg="#CBE1FF")
    left_frame.grid(row=0, column=0, padx=(15, 10), sticky="nw")

    # ========== THÊM 2 ẢNH SONG SONG ==========
    logo_img = None
    new_img = None
    img_frame = tk.Frame(left_frame, bg="#CBE1FF")
    img_frame.pack(anchor="nw", pady=(5, 4))

    # Logo bên trái
    try:
        # pil_logo = Image.open("Interface\\download.png")
        pil_logo = replace_white_background("Interface\\download.png", bg_color=(203, 225, 255))  # Thay nền trắng bằng màu #CBE1FF
        width, height = pil_logo.size
        pil_logo = pil_logo.resize((int(width * 0.5), int(height * 0.5)), Image.Resampling.LANCZOS)
        logo_img = ImageTk.PhotoImage(pil_logo)
        # Căn giữa theo chiều dọc frame chứa 2 ảnh
        logo_label = tk.Label(img_frame, image=logo_img, bg="#CBE1FF")
        logo_label.pack(side="left", anchor="s")  # anchor="s" giúp căn đáy
    except Exception as e:
        logo_label = tk.Label(img_frame, text="LOGO", bg="#CBE1FF", font=("Cambria", 12, "bold"))
        logo_label.pack(side="left", anchor="s")

    # Ảnh mới bên phải
    try:
        # pil_new = Image.open("D:\\UTE\\UTE_Nam_4_ki_2_DATN\\Thu_Thap_Data\\Test_Final_Gen_Data\\Interface\\GetArticleImage.jpg")
        pil_new = replace_white_background("D:\\UTE\\UTE_Nam_4_ki_2_DATN\\Thu_Thap_Data\\Test_Final_Gen_Data\\Interface\\GetArticleImage.jpg", bg_color=(203, 225, 255))  # Thay nền trắng bằng màu #CBE1FF
        pil_new = pil_new.resize((120,143), Image.Resampling.LANCZOS)
        new_img = ImageTk.PhotoImage(pil_new)
        new_label = tk.Label(img_frame, image=new_img, bg="#CBE1FF")
        new_label.pack(side="left", anchor="s", padx=(10, 0))
    except Exception as e:
        new_label = tk.Label(img_frame, text="ẢNH MỚI", bg="#CBE1FF", font=("Cambria", 12, "bold"))
        new_label.pack(side="left", anchor="s", padx=(10, 0))

    # ========== Kết thúc phần ảnh song song ==========

    info_text = (
        "GVHD : TS. Võ Lâm Chương\n"
        "SVTH  : Nguyễn Minh Khoa     21146112\n"
        "               Lê Nhật Duy                   21146441\n"
        "               Lê Lý Tam                       21146145"
    )
    info_label = tk.Label(
        left_frame, text=info_text, font=("Cambria", 14),
        fg="#000000", bg="#CBE1FF", justify="left", anchor="w"
    )
    info_label.pack(anchor="nw", pady=(8,0))

    # --- Cột phải: tiêu đề trường ---
    right_frame = tk.Frame(header_frame, bg="#CBE1FF")
    right_frame.grid(row=0, column=1, padx=(30,0), sticky="n")
    header_frame.grid_columnconfigure(1, weight=1)

    header_text = (
        "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP.HCM\n"
        "KHOA CƠ KHÍ CHẾ TẠO MÁY\n"
        "NGÀNH CƠ ĐIỆN TỬ\n"
    )
    header_label = tk.Label(
        right_frame,
        text=header_text,
        font=("Cambria", 15, "bold"),
        fg="#000000",
        bg="#CBE1FF",
        justify="center"
    )
    header_label.pack(anchor="n", pady=(18, 2), padx=(0, 20))

    # ===== THÊM DÒNG "ĐỒ ÁN TỐT NGHIỆP" =====
    doan_label = tk.Label(
        right_frame,
        text="ĐỒ ÁN TỐT NGHIỆP",
        font=("Cambria", 36, "bold"),
        fg="#d13015",   # màu đỏ đô nổi bật, bạn có thể chỉnh
        bg="#CBE1FF",
        justify="center"
    )
    doan_label.pack(anchor="n", pady=(30, 10), padx=(0, 20))
    # ===== KẾT THÚC THÊM DÒNG =====

    # ===== Nội dung phần dưới (căn giữa) =====
    intro = tk.Label(
        root,
        text="HỆ THỐNG GÁN NHÃN THỰC PHẨM TỰ ĐỘNG",
        font=("Cambria", 22, "bold"),
        bg="#CBE1FF", fg="#222"
    )
    intro.pack(pady=(20, 10))


    # Nút Start
    start_btn = tk.Button(root, text="Start", font=("Arial", 14), width=12, command=on_start)
    start_btn.pack(pady=(20, 10))

    # Nhãn trạng thái
    status_var = tk.StringVar()
    status_var.set("Nhấn Start để bắt đầu chọn thư mục ảnh.")
    status_lbl = tk.Label(
        root,
        textvariable=status_var,
        font=("Cambria", 10),
        wraplength=600,
        fg="#444",
        bg="#CBE1FF"
    )
    status_lbl.pack(pady=8)

    root.mainloop()


################################################################################################################

def convert_cropped_to_original_coordinates(
    crop1_rect, crop2_rect,
    x, y,
):
    """
    Chuyển (x, y, w, h) từ tọa độ pixel trong ảnh đã crop
    sang tọa độ pixel trong ảnh gốc.

    Args:
        crop1_rect (np.ndarray): 4 điểm (x, y) tạo thành hình chữ nhật crop1.
        crop2_rect (np.ndarray): 4 điểm (x, y) tạo thành hình chữ nhật crop2.
        x, y (float): Tọa độ tâm bounding box trong ảnh đã crop.
        w, h (float): Kích thước bounding box trong ảnh đã crop.

    Returns:
        Tuple[float, float, float, float]: (x, y, w, h) trong tọa độ pixel của ảnh gốc.
    """
    # Xác định chiều rộng của crop1 (giúp phân biệt box nằm ở crop1 hay crop2)
    crop1_width = crop1_rect[1][0] - crop1_rect[0][0]
    
    if x >= crop1_width:
        # Box thuộc crop2
        offset_x = crop2_rect[0][0]
        offset_y = crop2_rect[0][1]
        x_orig = (x - crop1_width) + offset_x
        y_orig = y + offset_y
    else:
        # Box thuộc crop1
        offset_x = crop1_rect[0][0]
        offset_y = crop1_rect[0][1]
        x_orig = x + offset_x
        y_orig = y + offset_y

    # Kích thước không thay đổi vì không chịu ảnh hưởng bởi vị trí crop
    return x_orig, y_orig


def select_folder():
    """Open a dialog to select a folder containing images."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Show dialog on top
    folder_path = filedialog.askdirectory(title="Select folder containing images")
    return folder_path

def get_image_files(folder_path):
    """Get all image files from the folder."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(folder_path, file))
    
    return sorted(image_files)  # Sort to ensure consistent order

def select_reference_image(folder_path, image_files):
    """
    Open a file dialog to select a reference image from the folder.
    
    Args:
        folder_path: Path to the folder containing images
        image_files: List of image file paths
        
    Returns:
        Index of the selected image in image_files list
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Show dialog on top
    
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select reference image for labeling",
        initialdir=folder_path,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    
    # If user cancels, default to first image
    if not file_path:
        print("No reference image selected. Using the first image by default.")
        return 0
    
    # Find the index of the selected file in image_files
    try:
        file_name = os.path.basename(file_path)
        selected_idx = next((i for i, f in enumerate(image_files) if os.path.basename(f) == file_name), 0)
        print(f"Selected reference image: {os.path.basename(file_path)}")
        return selected_idx
    except ValueError:
        print(f"Selected file not in the processed image list. Using the first image by default.")
        return 0

def save_yolo_labels(masks, labels_map, img_width, img_height, output_path, class_mapping, crop1, crop2):
    """
    Save labels in YOLO format (normalized center_x, center_y, width, height)
    """
    with open(output_path, 'w') as f:
        for i, mask_dict in enumerate(masks):
            if i not in labels_map:
                continue

            # Get bounding box and label
            x_min, y_min, w, h = mask_dict['bbox']
            label = labels_map[i]
            class_idx = class_mapping[label]

            # Convert to center-based coords
            x_center = x_min + w / 2
            y_center = y_min + h / 2

            # Convert center to original image coordinate
            x_center_orig, y_center_orig = convert_cropped_to_original_coordinates(
                crop1, crop2, x_center, y_center
            )

            # Normalize
            center_x = x_center_orig / img_width
            center_y = y_center_orig / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            # Write to file
            f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def process_images(image_files, labels_folder, parent_window):
    """Process all images and create YOLO labels using improved color comparison."""
    # Import SAM2 model here to avoid circular imports
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("Error: SAM2 modules not found. Please ensure SAM2 is installed correctly.")
        return

    # Check if labels folder exists, if not create it
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    # Load SAM2 model
    model_cfg = MODEL_CFG
    sam2_checkpoint = SAM2_CHECKPOINT
    
    try:
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        print("SAM2 model loaded successfully")
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        return
    
    # Create mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        box_nms_thresh=BOX_NMS_THRESH_SET,
        multimask_output=True,
        pred_iou_thresh=PRED_IOU_THRESH_SET,
        stability_score_thresh=STABILITY_SCORE_THRESH_SET,
        crop_n_layers=0,
    )
    
    # Process first image for manual labeling
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Let user select reference image
    print("Please select a reference image for manual labeling...")
    
    # Extract folder path from the first image (assuming all images are in the same folder)
    folder_path = os.path.dirname(image_files[0])
    ref_img_idx = select_reference_image(folder_path, image_files)
    first_img_path = image_files[ref_img_idx]

    # Move the selected image to the beginning of the list for processing
    image_files.pop(ref_img_idx)
    image_files.insert(0, first_img_path)

    # print(f"Selected reference image: {os.path.basename(first_img_path)}")
    
    # Load first image using OpenCV instead of PIL
    try:
        first_image = cv2.imread(first_img_path)
        if first_image is None:
            raise Exception("Failed to load image")
        base_image = first_image.copy()
        base_img_height, base_img_width = first_image.shape[:2]
        first_image = crop_and_concatenate(first_image, crop1, crop2, axis='horizontal', show=False)
        # Convert BGR to RGB (OpenCV loads as BGR, but we need RGB for processing)
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {first_img_path}: {e}")
        return
    
    # Get image dimensions for YOLO format
    first_img_height, first_img_width = first_image.shape[:2]
    
    # Generate masks for first image
    try:
        first_masks = mask_generator.generate(first_image)
    except Exception as e:
        print(f"Error generating masks for {first_img_path}: {e}")
        return
        
    # Filter masks
    first_masks = remove_background_mask(first_masks, remove_larger_than_threshold=True, max_area_threshold=MAX_AREA_THRESHOLD_SET,
                                        remove_smallest=True, min_area_threshold=MIN_AREA_THRESHOLD_SET)
    
    # Further filter masks based on squareness
    first_masks = filter_masks_by_squareness(first_masks, min_squareness_ratio=MIN_SQUARNESS_RATIO)
    
    # Further filter masks based on bounding box overlap
    first_masks = process_masks(first_masks, (first_img_height, first_img_width),
                                overlap_threshold=0.6, fill_ring=True, add_area=ADD_AREA)
    
    if not first_masks:
        print(f"No valid masks found for {first_img_path}")
        return
    
    # Use new labeling window instead of matplotlib
    def labeling_callback(manual_labels, class_set):
        if manual_labels is None:  # User cancelled
            print("Labeling cancelled by user")
            return
            
        # Create class mapping
        class_list = sorted(list(class_set))
        class_mapping = {label: i for i, label in enumerate(class_list)}
        
        print(f"Class mapping: {class_mapping}")
        
        # Save YOLO labels for first image
        base_filename = os.path.splitext(os.path.basename(first_img_path))[0]
        label_file = os.path.join(labels_folder, f"{base_filename}.txt")
        save_yolo_labels(first_masks, manual_labels, base_img_width, base_img_height, label_file, class_mapping, crop1, crop2)
        
        # Process remaining images - AUTOMATIC LABELING using improved algorithm
        if len(image_files) > 1:
            print("\nProcessing remaining images with advanced color comparison...")
            
            # Prepare labeled indices and labels dictionary for reference
            labeled_indices = list(manual_labels.keys())
            base_labels = manual_labels  # This is already in the correct format {index: label}
            
            for img_path in image_files[1:]:
                print(f"Processing {os.path.basename(img_path)}...")
                
                # Load image using OpenCV instead of PIL
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        raise Exception("Failed to load image")
                    # Save original image dimensions before cropping
                    original_img_height, original_img_width = image.shape[:2]
                    
                    # Convert BGR to RGB (OpenCV loads as BGR, but we need RGB for processing)
                    image = crop_and_concatenate(image, crop1, crop2, axis='horizontal', show=False)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
                
                # Get image dimensions for YOLO format
                img_height, img_width = image.shape[:2]
                
                # Generate masks
                try:
                    masks = mask_generator.generate(image)
                except Exception as e:
                    print(f"Error generating masks for {img_path}: {e}")
                    continue
                    
                # Filter masks
                masks = remove_background_mask(masks, remove_larger_than_threshold=True, max_area_threshold=MAX_AREA_THRESHOLD_SET,
                                               remove_smallest=True, min_area_threshold=MIN_AREA_THRESHOLD_SET)
                
                # Further filter masks based on squareness
                masks = filter_masks_by_squareness(masks, min_squareness_ratio=MIN_SQUARNESS_RATIO)
                
                # Further filter masks based on bounding box overlap
                masks = process_masks(masks, (img_height, img_width),
                                      overlap_threshold=0.6, fill_ring=True, add_area=ADD_AREA)
                
                if not masks:
                    # print(f"No valid masks found for {img_path}")
                    # try:
                    #     os.remove(img_path)
                    #     print(f"Deleted image without valid masks: {os.path.basename(img_path)}")
                    # except Exception as e:
                    #     print(f"Error deleting {os.path.basename(img_path)}: {e}")
                    # continue
                    print(f"No valid masks found for {img_path}")
                    print(f"Keeping image without valid masks: {os.path.basename(img_path)}")
                    # Không tạo file txt cho ảnh này, chỉ giữ lại ảnh
                    continue

                # Automatically label masks using the improved compare_and_label_mask function
                try:
                    auto_labels = compare_and_label_mask(
                        reference_image=first_image,
                        reference_masks=first_masks,
                        labeled_indices=labeled_indices,
                        base_labels=base_labels,
                        new_image=image,
                        new_masks=masks,
                        area_weight=0.4,  # Can be adjusted based on needs
                        color_weight=0.6  # Can be adjusted based on needs
                    )
                except Exception as e:
                    print(f"Error in automatic labeling for {img_path}: {e}")
                    continue
                
                # Create YOLO format labels - Use original image dimensions
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                label_file = os.path.join(labels_folder, f"{base_filename}.txt")
                save_yolo_labels(masks, auto_labels, original_img_width, original_img_height, label_file, class_mapping, crop1, crop2)
        
        # Save class mapping to a file
        with open(os.path.join(labels_folder, "classes.txt"), 'w') as f:
            for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{label}\n")
        
        print(f"\nProcessing complete! Labels saved to {labels_folder}")
        print(f"Class mapping saved to {os.path.join(labels_folder, 'classes.txt')}")
        print("Advanced color comparison with LAB and HSV color spaces was used for automatic labeling.")
    
    # Open labeling window
    labeling_window = LabelingWindow(parent_window, first_image, first_masks, labeling_callback)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images and generate YOLO format labels with advanced color comparison')
    parser.add_argument('--folder', type=str, help='Path to folder containing images (optional)')
    
    args = parser.parse_args()
    
    # Select folder containing images
    if args.folder:
        folder_path = args.folder
    else:
        print("Please select a folder containing images...")
        folder_path = select_folder()
    
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    print(f"Selected folder: {folder_path}")
    
    # Get all image files in the folder
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print("No image files found in the selected folder.")
        return
    
    FOLDER_LABELS = folder_path
    
    # Create labels folder
    labels_folder = os.path.join(FOLDER_LABELS, "labels")
    
    # Process all images - need to pass a dummy parent window for command line mode
    root = tk.Tk()
    root.withdraw()
    process_images(image_files, labels_folder, root)

if __name__ == "__main__":
    # Run GUI app
    run_gui_app()
    
    # Uncomment below to run in command line mode
    # main()