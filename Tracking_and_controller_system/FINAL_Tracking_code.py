import cv2
import threading
import time
import numpy as np
import torch
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from queue import Queue
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
import os
import pymcprotocol
import struct

from Tools import load_calibration, get_undistort_map, undistort_with_map, pixel_to_world
from Tools import delta_calcInverse, replace_white_background

# ============ CÁC HÀM TÍNH TOÁN KINEMATICS ============
def write_dword_to_wordlist(value):
    if not (0 <= value <= 0xFFFFFFFF):
        raise ValueError("Giá trị vượt quá phạm vi 32-bit unsigned.")

    def to_signed16(val):
        return val - 0x10000 if val > 32767 else val

    low_word = to_signed16(value & 0xFFFF)
    high_word = to_signed16((value >> 16) & 0xFFFF)
    return [low_word, high_word]


def calculate_inverse_kinematics_pick(x, y, z, down):
    """Tính toán vị trí của robot Delta dựa trên tọa độ x, y, z"""
    a, b, c = delta_calcInverse(x, y, z)
    a1, b1, c1 = delta_calcInverse(x, y, z - down)

    a = int(a)*700
    b = int(b)*700
    c = int(c)*700
    a1 = int(a1)*700
    b1 = int(b1)*700
    c1 = int(c1)*700

    mc.batchwrite_wordunits(headdevice="D900", values=write_dword_to_wordlist(a))
    mc.batchwrite_wordunits(headdevice="D905", values=write_dword_to_wordlist(b))
    mc.batchwrite_wordunits(headdevice="D910", values=write_dword_to_wordlist(c))

    mc.batchwrite_wordunits(headdevice="D1100", values=write_dword_to_wordlist(a1))
    mc.batchwrite_wordunits(headdevice="D1105", values=write_dword_to_wordlist(b1))
    mc.batchwrite_wordunits(headdevice="D1110", values=write_dword_to_wordlist(c1))

    return a, b, c

def calculate_inverse_kinematics_place(x, y, z, down):
    """Tính toán vị trí của robot Delta dựa trên tọa độ x, y, z"""
    a, b, c = delta_calcInverse(x, y, z)
    a1, b1, c1 = delta_calcInverse(x, y, z - down)

    a = int(a)*700
    b = int(b)*700
    c = int(c)*700
    a1 = int(a1)*700
    b1 = int(b1)*700
    c1 = int(c1)*700

    mc.batchwrite_wordunits(headdevice="D1200", values=write_dword_to_wordlist(a))
    mc.batchwrite_wordunits(headdevice="D1205", values=write_dword_to_wordlist(b))
    mc.batchwrite_wordunits(headdevice="D1210", values=write_dword_to_wordlist(c))

    mc.batchwrite_wordunits(headdevice="D1300", values=write_dword_to_wordlist(a1))
    mc.batchwrite_wordunits(headdevice="D1305", values=write_dword_to_wordlist(b1))
    mc.batchwrite_wordunits(headdevice="D1310", values=write_dword_to_wordlist(c1))
    return a, b, c

def start_point(x,y,z):
    a, b, c = delta_calcInverse(x, y, z)
    a = int(a)*700
    b = int(b)*700
    c = int(c)*700

    mc.batchwrite_wordunits(headdevice="D1500", values=write_dword_to_wordlist(a))
    mc.batchwrite_wordunits(headdevice="D1505", values=write_dword_to_wordlist(b))
    mc.batchwrite_wordunits(headdevice="D1510", values=write_dword_to_wordlist(c))

    mc.batchwrite_bitunits(headdevice="M340", values=[1])

# ========== THIẾT LẬP ==========

class Args: 
    track_buffer = 120 # Số frame để giữ track
    min_hit = 5
    track_high_thresh = 0.75
    track_low_thresh = 0.6
    new_track_thresh = 0.75
    match_thresh = 0.8
    fuse_score = True

class FakeResults:
    def __init__(self, boxes_data):
        if len(boxes_data) == 0:
            self.conf = np.array([])
            self.xywh = np.array([]).reshape(0, 4)
            self.cls = np.array([])
        else:
            boxes_array = np.array(boxes_data)
            self.conf = boxes_array[:, 4]
            x1, y1, w, h = boxes_array[:, 0], boxes_array[:, 1], boxes_array[:, 2], boxes_array[:, 3]
            center_x = x1 + w / 2
            center_y = y1 + h / 2
            self.xywh = np.column_stack([center_x, center_y, w, h])
            self.cls = boxes_array[:, 5].astype(int)


# ========== CÁC HÀM DÀNH CHO GIAO TIẾP PLC==========
# Khởi tạo kết nối đến PLC
mc = pymcprotocol.Type3E()

# Kết nối đến IP của PLC
mc.connect('192.168.10.100', 3000)

def write_float_to_words(value):
    data_bytes = struct.pack('>f', value)
    high_word = int.from_bytes(data_bytes[:2], byteorder='big', signed=False)
    low_word  = int.from_bytes(data_bytes[2:], byteorder='big', signed=False)
    # Chuyển về signed 16 bit nếu cần
    def to_signed(w):
        return w if w < 32768 else w - 65536
    return [to_signed(low_word), to_signed(high_word)]

def get_and_remove_nearest_object(Class_label, x, class_type):
    """
    Tìm object gần x nhất chưa được xử lý, đánh dấu đã được xử lý và trả về position.
    class_type: 'class_0' hoặc 'class_1'
    """
    if not Class_label:
        return None
    
    nearest_position = None
    nearest_id = None
    min_dist = float('inf')
    
    for obj_id, obj in Class_label.items():
        # Bỏ qua nếu ID đã được xử lý
        if obj_id in processed_ids[class_type]:
            continue
            
        pos = obj.get('position')
        if pos is None:
            continue

        x_center, y_center = pos

        # Bỏ qua nếu y_center > 200
        if x_center*10 > 150:
            continue

        dist = abs(x_center - x)
        if dist < min_dist:
            min_dist = dist
            nearest_position = (x_center*10, y_center*10)
            nearest_id = obj_id
    
    # Đánh dấu ID đã được xử lý
    if nearest_id is not None:
        processed_ids[class_type].add(nearest_id)
    
    return nearest_position

# ========== BIẾN TOÀN CỤC ==========

# model_path = "D:\\UTE\\UTE_Nam_4_ki_2_DATN\\Tai_model_va_detect\\Model_all_of_Banhpia\\detect\\train2\\weights\\best.pt"
model_path = "D:\\UTE\\UTE_Nam_4_ki_2_DATN\\ZZZ_Data_banhnho\\detect\\train23\\weights\\best.pt"

camera_shape = (640, 480)

# Khởi tạo calibration
mtx, dist = load_calibration("Tools\\camera_calibration.npz")

H_loaded = np.load('Tools\\homography_matrix.npy')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)
class_names = model.names
print(f"Detected classes: {class_names}")

args = Args()
tracker = BYTETracker(args)

frame_lock = threading.Lock()
shared_frame = None
running = True
frame_count = 0 

cap = cv2.VideoCapture(0)
map1, map2, roi = get_undistort_map(mtx, dist, camera_shape, alpha=0.0)

# 2 Queue riêng biệt
frame_queue = Queue()
tracking_queue = Queue()

# Bảng theo dõi và đếm xuất hiện
track_count = defaultdict(int)  # Đếm số lần xuất hiện của mỗi ID
track_class = {}  # Lưu class của mỗi ID

# Hai bảng dict global cho 2 class (tùy chỉnh class ID)
TARGET_CLASS_1 = 2  # Thay đổi class ID muốn track
TARGET_CLASS_2 = 0  # Thay đổi class ID muốn track

class_0_table = {}  # {new_id: {'Class': TARGET_CLASS_1, 'ID': new_id, 'position': (x, y)}}
class_1_table = {}  # {new_id: {'Class': TARGET_CLASS_2, 'ID': new_id, 'position': (x, y)}}

class_0_delayed_table = {}
class_1_delayed_table = {}

# Thêm vào phần BIẾN TOÀN CỤC
processed_ids = {
    'class_0': set(),  # Set chứa các ID class_0 đã được xử lý
    'class_1': set()   # Set chứa các ID class_1 đã được xử lý
}

# Mapping từ original ID sang new ID
original_to_new_id = {}  # {original_id: new_id}
new_id_counter = {'class_0': 1, 'class_1': 1}  # Counter cho ID mới

threashold_frame = 3  # Số frame cần để kích hoạt tracking

# Biến cho giao diện chạy
run_window = None
video_label = None
stats_label = None
plc_count = 0  # Đếm số lượng vật thể PLC đã thao tác
plc_started = False
gui_active = False

# Biến theo dõi trạng thái các nút nhấn
button_states = {
    'start': False,
    'stop': False, 
    'home': False,
    'conveyor': False
}
# ========== LUỒNG ĐỌC FRAME ==========

def reader_thread():
    global shared_frame, running
    while running:
        success, frame = cap.read()
        if not success:
            running = False
            break
        
        frame = undistort_with_map(frame, map1, map2)
        
        with frame_lock:
            shared_frame = frame.copy()
        
        # Đẩy frame vào queue
        frame_queue.put(frame.copy())
        time.sleep(0.001)

# ========== LUỒNG TRACKING ==========

def tracker_thread():
    global shared_frame, running, frame_count
    
    while running:
        frame_to_process = None
        with frame_lock:
            if shared_frame is not None:
                frame_to_process = shared_frame.copy()
                shared_frame = None

        if frame_to_process is None:
            time.sleep(0.001)
            continue

        frame_count += 1

        # Detect objects
        results = model.predict(frame_to_process,
                                conf=0.75,
                                iou=0.75,
                                imgsz=640,
                                device=device)[0]

        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                w, h = x2 - x1, y2 - y1
                dets.append([x1, y1, w, h, conf, cls])

        # Update tracker
        fake_results = FakeResults(dets)
        outputs = tracker.update(fake_results, frame_to_process)

        # Tạo dict cho frame này - sử dụng outputs từ tracker
        frame_tracking_data = {}
        
        # Xử lý outputs từ tracker
        for output in outputs:
            x1, y1, x2, y2, track_id, score, cls_id, _ = output
            track_id = int(track_id)
            
            frame_tracking_data[track_id] = {
                'ID': track_id,
                'class': int(cls_id),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'is_activated': True
            }

        # Thêm thông tin về tracks đã biến mất (có thể lấy từ tracker.lost_stracks)
        try:
            for lost_track in tracker.lost_stracks:
                track_id = int(lost_track.track_id)
                if track_id not in frame_tracking_data:
                    frame_tracking_data[track_id] = {
                        'ID': track_id,
                        'class': None,
                        'bbox': None,
                        'is_activated': False
                    }
        except:
            pass
        
        # Đẩy tracking data vào queue
        tracking_queue.put(frame_tracking_data)

# ========== LUỒNG XỬ LÝ KẾT QUẢ ==========

def output_thread():
    """Luồng xử lý kết quả - SỬA LỖI"""
    global running, track_count, track_class, class_0_table, class_1_table
    global original_to_new_id, new_id_counter
    global class_0_delayed_table, class_1_delayed_table, gui_active

    # Cấu trúc lưu lịch sử để tạo bảng trễ
    history_class_0 = deque()
    history_class_1 = deque()

    # Delay thời gian (tính bằng giây)
    t1 = 5.86  # Delay cho class 0 # 5.52
    t2 = 6.45  # Delay cho class 1 # 6.79

    while running:
        current_time = time.time()
        current_frame = None

        # Lấy frame mới nhất từ queue
        while not frame_queue.empty():
            current_frame = frame_queue.get()

        # Lấy tracking data từ queue
        while not tracking_queue.empty():
            tracking_data = tracking_queue.get()

            # Xử lý từng track trong frame
            for original_id, info in tracking_data.items():
                if info['is_activated']:
                    # Track đang hoạt động
                    track_count[original_id] += 1
                    track_class[original_id] = info['class']

                    # Kiểm tra nếu xuất hiện đủ ngưỡng
                    if track_count[original_id] >= threashold_frame:
                        class_id = info['class']

                        # Tạo new_id nếu chưa có
                        if original_id not in original_to_new_id:
                            if class_id == TARGET_CLASS_1:
                                new_id = new_id_counter['class_0']
                                new_id_counter['class_0'] += 1
                            elif class_id == TARGET_CLASS_2:
                                new_id = new_id_counter['class_1']
                                new_id_counter['class_1'] += 1
                            else:
                                continue  # Bỏ qua class khác

                            original_to_new_id[original_id] = new_id

                        # Tính toán vị trí trung tâm
                        if info['bbox']:
                            x1, y1, x2, y2 = info['bbox']
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            position = (center_x, center_y)
                        else:
                            position = None

                        # Cập nhật bảng chính
                        new_id = original_to_new_id[original_id]
                        if class_id == TARGET_CLASS_1:
                            class_0_table[new_id] = {
                                'Class': 0,
                                'ID': new_id,
                                'position': position,
                                'bbox': info['bbox']
                            }
                        elif class_id == TARGET_CLASS_2:
                            class_1_table[new_id] = {
                                'Class': 1,
                                'ID': new_id,
                                'position': position,
                                'bbox': info['bbox']
                            }

                else:
                    # Track đã biến mất – xóa khỏi bảng chính
                    if original_id in original_to_new_id:
                        new_id = original_to_new_id[original_id]
                        class_id = track_class.get(original_id)

                        if class_id == TARGET_CLASS_1 and new_id in class_0_table:
                            del class_0_table[new_id]
                        elif class_id == TARGET_CLASS_2 and new_id in class_1_table:
                            del class_1_table[new_id]

                        # Xóa mapping
                        del original_to_new_id[original_id]
                        if original_id in track_count:
                            del track_count[original_id]
                        if original_id in track_class:
                            del track_class[original_id]

        # -------------------------------
        # Cập nhật lịch sử để tạo bảng trễ
        # -------------------------------

        history_class_0.append((current_time, class_0_table.copy()))
        history_class_1.append((current_time, class_1_table.copy()))

        # Loại bỏ và cập nhật bảng trễ sau thời gian t1
        while history_class_0 and current_time - history_class_0[0][0] > t1:
            _, snapshot_0 = history_class_0.popleft()
            # Đổi tất cả position trong snapshot_0 sang tọa độ thực
            converted_0 = {}
            for k, v in snapshot_0.items():
                pos = v.get('position')
                if pos is not None:
                    # Chỉ convert khi có H_loaded
                    try:
                        xw, yw = pixel_to_world(pos[0], pos[1], H_loaded)
                        v_new = v.copy()
                        v_new['position'] = (float(xw), float(yw))
                        converted_0[k] = v_new
                    except:
                        converted_0[k] = v.copy()
                else:
                    converted_0[k] = v.copy()
            class_0_delayed_table = converted_0

        while history_class_1 and current_time - history_class_1[0][0] > t2:
            _, snapshot_1 = history_class_1.popleft()
            converted_1 = {}
            for k, v in snapshot_1.items():
                pos = v.get('position')
                if pos is not None:
                    try:
                        xw, yw = pixel_to_world(pos[0], pos[1], H_loaded)
                        v_new = v.copy()
                        v_new['position'] = (float(xw), float(yw))
                        converted_1[k] = v_new
                    except:
                        converted_1[k] = v.copy()
                else:
                    converted_1[k] = v.copy()
            class_1_delayed_table = converted_1

        # Hiển thị video với annotations
        if current_frame is not None:
            display_frame = current_frame.copy()
            
            # Vẽ Class 0 objects (màu xanh lá)
            for new_id, data in class_0_table.items():
                if data['bbox'] and data['position']:
                    x1, y1, x2, y2 = data['bbox']
                    center_x, center_y = data['position']
                    
                    # Vẽ text thông tin
                    label = f"Cake:{data['ID']}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Vẽ điểm trung tâm
                    cv2.circle(display_frame, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # Vẽ Class 1 objects (màu đỏ)
            for new_id, data in class_1_table.items():
                if data['bbox'] and data['position']:
                    x1, y1, x2, y2 = data['bbox']
                    center_x, center_y = data['position']
                    
                    # Vẽ text thông tin
                    label = f"Box:{data['ID']}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Vẽ điểm trung tâm
                    cv2.circle(display_frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Hiển thị thông tin tổng quan
            info_text = f"FPS: 30"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (142, 63, 239), 2)

            # Cập nhật giao diện chỉ khi GUI còn hoạt động
            if gui_active:
                update_stats_display()
                update_video_display(display_frame)

        # Chờ một chút để giảm tải CPU
        time.sleep(0.001)

    cv2.destroyAllWindows()

# ========== LUỒNG GỬI TÍN HIỆU ĐẾN PLC ==========

def send_to_plc():
    global class_0_delayed_table, class_1_delayed_table, plc_started, plc_count
    global started
    started = False
    cycle_running = False
    ver = 0  # 0: chờ class_0, 3: chờ class_1

    mc.batchwrite_wordunits(headdevice="D500", values=write_float_to_words(0.25))

    while running:
        # Kiểm tra nếu đã được start từ giao diện
        if plc_started and not started:
            started = True
            ver = 0
            cycle_running = False
            co = 0
            current_target = None
            start_point(0.0,-150.0,-450.0)
            print("Bắt đầu chạy state machine!")

        # Chỉ chạy khi đã started
        if started:
            flag_pos = mc.batchread_bitunits(headdevice="M96", readsize=1)[0]
            flag_pos_ver1 = mc.batchread_bitunits(headdevice="M2999", readsize=1)[0] # M3001 off
            flag_pos_ver2 = mc.batchread_bitunits(headdevice="M93", readsize=1)[0]   # M3002 off
            
            # Cho class_0
            if ver == 0 and (flag_pos == 1 or flag_pos_ver2 == 1) and not cycle_running:
                nearest_pos = get_and_remove_nearest_object(class_0_delayed_table, 0, 'class_0')
                if nearest_pos is not None:
                    current_target = nearest_pos
                    cycle_running = True
                    co = 1

            # Cho class_1  
            elif ver == 3 and flag_pos_ver1 == 1 and not cycle_running:
                nearest_pos = get_and_remove_nearest_object(class_1_delayed_table, 0, 'class_1')
                if nearest_pos is not None:
                    current_target = nearest_pos
                    cycle_running = True
                    co = 2
            
            # Thực hiện state machine khi có chu trình đang chạy
            if ver == 0 and co == 1 and cycle_running and current_target is not None:
                x_target, y_target = current_target
                calculate_inverse_kinematics_pick( x_target*1.02 + 15, y_target*1.02 + 5, -450.0, 57.0) #57
                mc.batchwrite_bitunits(headdevice="M310", values=[1])
                mc.batchwrite_bitunits(headdevice="M3002", values=[1])
                ver = 3
                current_target = None
                cycle_running = False

            elif ver == 3 and co == 2 and cycle_running and current_target is not None: 
                x_target, y_target = current_target
                calculate_inverse_kinematics_place(x_target*1.02 + 15, y_target*1.02 + 3, -450.0, 35.0)
                mc.batchwrite_bitunits(headdevice="M3001", values=[1])
                mc.batchwrite_bitunits(headdevice="M325", values=[1])
                time.sleep(0.001)
                mc.batchwrite_bitunits(headdevice="M325", values=[0])
                ver = 0
                current_target = None
                cycle_running = False
                plc_count += 1

        time.sleep(0.001)  # Sleep 50ms

# ========== CHẠY ĐA LUỒNG ==========
def start_system():
    """Hàm được gọi khi nhấn nút Start"""
    global running, gui_active
    running = True
    gui_active = True
    
    # Tạo cửa sổ giao diện chạy
    create_run_window()
    
    # Khởi tạo và chạy các luồng
    t1 = threading.Thread(target=reader_thread)
    t2 = threading.Thread(target=tracker_thread)
    t3 = threading.Thread(target=output_thread)
    t4 = threading.Thread(target=send_to_plc)

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    
    # Ẩn cửa sổ tkinter chính
    root.withdraw()
    
    print("🚀 Hệ thống đã khởi động.")

def check_threads():
    """Kiểm tra trạng thái threads định kỳ"""
    global running, gui_active
    
    if running and gui_active:
        # Kiểm tra xem run_window có còn tồn tại không
        try:
            if run_window and run_window.winfo_exists():
                # Lặp lại kiểm tra sau 1 giây
                run_window.after(1000, check_threads)
            else:
                # Cửa sổ đã bị đóng
                cleanup_system()
        except tk.TclError:
            # Cửa sổ đã bị hủy
            cleanup_system()
    
def cleanup_system():
    """Cleanup khi system dừng"""
    global running, gui_active
    running = False
    gui_active = False
    
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    
    print("🛑 Đã dừng hệ thống.")
    
    # Hiện lại cửa sổ chính
    try:
        root.deiconify()
    except:
        pass

def create_run_window():
    """Tạo cửa sổ giao diện chạy"""
    global run_window, video_label, stats_label
    
    run_window = tk.Toplevel(root)
    run_window.title("Hệ thống theo dõi và đóng gói - Đang chạy")
    run_window.geometry("1000x700")
    run_window.resizable(False, False)
    run_window.configure(bg="#CBE1FF")
    
    # Frame chính chia làm 2 cột
    main_frame = tk.Frame(run_window, bg="#CBE1FF")
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Cột trái - Thông tin và điều khiển
    left_frame = tk.Frame(main_frame, bg="#CBE1FF", width=300)
    left_frame.pack(side='left', fill='y', padx=(0, 10))
    left_frame.pack_propagate(False)
    
    # Tiêu đề trái
    title_left = tk.Label(left_frame, text="THÔNG TIN HỆ THỐNG", 
                         font=("Cambria", 17, "bold"), bg="#CBE1FF", fg="#000")
    title_left.pack(pady=(10, 20))
    
    # Hiển thị thống kê
    stats_label = tk.Label(left_frame, text="", font=("Cambria", 12), 
                          bg="#CBE1FF", fg="#000", justify="left")
    stats_label.pack(pady=(0, 20))
    
    # Nút Start PLC
    start_plc_button = tk.Button(left_frame, text="Start Tracking", 
                               command=start_plc_system,
                               font=("Cambria", 15), width=15, height=2,
                               bg="#4CAF50", fg='white', relief='raised', bd=2)
    start_plc_button.pack(pady=10)
    
    # Nút Quit
    quit_button = tk.Button(left_frame, text="Quit", 
                           command=quit_system,
                           font=("Cambria", 15), width=15, height=2,
                           bg="#f44336", fg='white', relief='raised', bd=2)
    quit_button.pack(pady=10)

    # Frame chứa 4 nút nhấn hình tròn
    control_frame = tk.Frame(left_frame, bg="#CBE1FF")
    control_frame.pack(pady=20)
    
    # Tạo 4 nút nhấn theo kiểu hình chữ nhật
    button_configs = [
        ("Start", "X26", 0, 0, "#4CAF50"),    # Xanh lá
        ("Stop", "X27", 0, 1, "#f44336"),     # Đỏ
        ("Home", "X28", 1, 0, "#2196F3"),     # Xanh dương
        ("Conveyor", "X29", 1, 1, "#FF9800")  # Cam
    ]
    
    for name, x_addr, row, col, color in button_configs:
        # Tạo frame con cho mỗi nút (chứa label + button)
        btn_frame = tk.Frame(control_frame, bg="#CBE1FF")
        btn_frame.grid(row=row, column=col, padx=20, pady=15)  # Tăng khoảng cách
        
        # Label hiển thị tên nút phía trên
        label = tk.Label(btn_frame, text=name,
                        font=("Cambria", 16, "bold"),  # Tăng font size
                        bg="#CBE1FF", fg="#000")
        label.pack(pady=(0, 5))
        
        # Nút hình tròn
        canvas = tk.Canvas(btn_frame, width=80, height=80, bg="#CBE1FF", highlightthickness=0)
        canvas.pack()
        
        # Vẽ hình tròn với viền đen
        circle = canvas.create_oval(5, 5, 75, 75, fill=color, outline="#000000", width=3)
        
        # Bind sự kiện nhấn và nhả cho canvas
        canvas.bind("<Button-1>", lambda e, n=name.lower(), x=x_addr, c=canvas, cir=circle, col=color: on_button_press(n, x, c, cir, col))
        canvas.bind("<ButtonRelease-1>", lambda e, n=name.lower(), x=x_addr, c=canvas, cir=circle, col=color: on_button_release(n, x, c, cir, col))
        
        # Thêm hiệu ứng hover
        def on_enter(e, c=canvas, cir=circle, col=color):
            # Làm sáng màu khi hover
            if col == "#4CAF50":  # Xanh lá
                lighter_color = "#8ECD91"
            elif col == "#f44336":  # Đỏ
                lighter_color = "#ED7B79"
            elif col == "#2196F3":  # Xanh dương
                lighter_color = "#6FB9F6"
            elif col == "#FF9800":  # Cam
                lighter_color = "#F5C57C"
            else:
                lighter_color = col
            c.itemconfig(cir, fill=lighter_color)

        def on_leave(e, c=canvas, cir=circle, col=color):
            c.itemconfig(cir, fill=col)

        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
    
    # Cột phải - Video
    right_frame = tk.Frame(main_frame, bg="#CBE1FF")
    right_frame.pack(side='right', fill='both', expand=True)
    
    # Tiêu đề phải
    title_right = tk.Label(right_frame, text="CAMERA TRACKING", 
                          font=("Cambria", 17, "bold"), bg="#CBE1FF", fg="#000")
    title_right.pack(pady=(10, 10))
    
    # Label hiển thị video
    video_label = tk.Label(right_frame, bg="#000")
    video_label.pack(expand=True, fill='both')
    
    # Xử lý khi đóng cửa sổ
    run_window.protocol("WM_DELETE_WINDOW", on_run_window_closing)
    
    # Bắt đầu kiểm tra threads
    run_window.after(1000, check_threads)

def update_stats_display():
    """Cập nhật hiển thị thống kê"""
    global stats_label, run_window, gui_active
    
    if not gui_active:
        return
    
    try:
        if stats_label and run_window and run_window.winfo_exists():
            stats_text = f"Số lượng Bánh: {len(class_0_table)}\n"
            stats_text += f"Số lượng Hộp: {len(class_1_table)}\n"
            stats_text += f"PLC đã thao tác: {plc_count} vật thể"
            stats_label.config(text=stats_text)
    except (tk.TclError, AttributeError):
        # Widget đã bị hủy hoặc không tồn tại
        pass

def update_video_display(frame):
    """Cập nhật hiển thị video"""
    global video_label, run_window, gui_active
    
    if not gui_active:
        return
    
    try:
        if video_label and run_window and run_window.winfo_exists() and frame is not None:
            # Resize frame để phù hợp với giao diện
            frame_resized = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Chuyển đổi sang PhotoImage
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            
            # Cập nhật label
            video_label.configure(image=photo)
            video_label.image = photo  # Giữ reference
    except (tk.TclError, AttributeError):
        # Widget đã bị hủy hoặc không tồn tại
        pass

def on_closing():
    """Hàm xử lý khi đóng cửa sổ chính"""
    global running, gui_active
    running = False
    gui_active = False
    
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    
    root.destroy()

def on_run_window_closing():
    """Xử lý khi đóng cửa sổ chạy"""
    global running, gui_active
    running = False
    gui_active = False
    
    try:
        if run_window:
            run_window.destroy()
    except:
        pass
    
    try:
        root.deiconify()
    except:
        pass

def start_plc_system():
    """Bắt đầu hệ thống PLC"""
    global plc_started
    plc_started = True
    print("Bắt đầu chạy state machine!")

def quit_system():
    """Thoát hệ thống"""
    global running, gui_active, plc_started, started
    running = False
    gui_active = False
    plc_started = False
    started = False
    
    try:
        if run_window:
            run_window.destroy()
    except:
        pass
    
    try:
        root.deiconify()
    except:
        pass

def on_button_press(button_name, x_address, canvas, circle, original_color):
    """Xử lý khi nhấn nút"""
    global button_states
    # Đổi màu thành trắng khi nhấn
    canvas.itemconfig(circle, fill="#FFFFFF")
    try:
        mc.batchwrite_bitunits(headdevice=x_address, values=[1])
        button_states[button_name] = True
        print(f"Nhấn {button_name} - {x_address} = 1")
    except Exception as e:
        print(f"Lỗi khi nhấn {button_name}: {e}")

def on_button_release(button_name, x_address, canvas, circle, original_color):
    """Xử lý khi nhả nút"""
    global button_states
    # Trở về màu bình thường khi nhả
    canvas.itemconfig(circle, fill=original_color)
    try:
        mc.batchwrite_bitunits(headdevice=x_address, values=[0])
        button_states[button_name] = False
        print(f"Nhả {button_name} - {x_address} = 0")
    except Exception as e:
        print(f"Lỗi khi nhả {button_name}: {e}")

# ========== GIAO DIỆN TKINTER ==========
# Khởi tạo cửa sổ chính
root = tk.Tk()
root.title("Do an Co Dien Tu - YOLO Tracking")
root.geometry("850x500")
root.resizable(False, False)    
root.configure(bg="#CBE1FF")

# ==== Header: logo trái, tiêu đề giữa, logo phải ====
header_frame = tk.Frame(root, bg="#CBE1FF")
header_frame.pack(fill='x', pady=(10, 0))

# --- Logo trái (góc trên bên trái) ---
logo_img = None
left_logo_frame = tk.Frame(header_frame, bg="#CBE1FF")
left_logo_frame.pack(side="left", padx=(20, 0), pady=(15, 0), anchor="nw")

logo_path = "D:\\UTE\\UTE_Nam_4_ki_2_DATN\\Thu_Thap_Data\\Test_Final_Gen_Data\\Interface\\download.png"
try:
    if os.path.exists(logo_path):
        pil_logo = Image.open(logo_path)
        # Nếu có function replace_white_background, sử dụng nó
        if 'replace_white_background' in globals():
            pil_logo = replace_white_background(logo_path, bg_color=(203, 225, 255))
        width, height = pil_logo.size
        pil_logo = pil_logo.resize((int(width * 0.5), int(height * 0.5)), Image.Resampling.LANCZOS)
        logo_img = ImageTk.PhotoImage(pil_logo)
        logo_label = tk.Label(left_logo_frame, image=logo_img, bg="#CBE1FF")
        logo_label.pack()
        # Giữ reference để tránh garbage collection
        logo_label.image = logo_img
    else:
        raise FileNotFoundError("Logo file not found")
except Exception as e:
    logo_label = tk.Label(left_logo_frame, text="LOGO", bg="#CBE1FF", font=("Arial", 12, "bold"))
    logo_label.pack()

# --- Tiêu đề giữa (header_text và doan_label) ---
center_frame = tk.Frame(header_frame, bg="#CBE1FF")
center_frame.pack(side="left", expand=True, pady=(0, 0), padx=(30, 0))

header_text = (
    "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP.HCM\n"
    "KHOA CƠ KHÍ CHẾ TẠO MÁY\n"
    "NGÀNH CƠ ĐIỆN TỬ\n"
)
header_label = tk.Label(
    center_frame,
    text=header_text,
    font=("Cambria", 15, "bold"),
    fg="#000000",
    bg="#CBE1FF",
    justify="center"
)
header_label.pack(pady=(10, 5))

# ===== THÊM DÒNG "ĐỒ ÁN TỐT NGHIỆP" =====
doan_label = tk.Label(
    center_frame,
    text="ĐỒ ÁN TỐT NGHIỆP",
    font=("Cambria", 35, "bold"),
    fg="#d13015",
    bg="#CBE1FF",
    justify="center"
)
doan_label.pack(pady=(10, 0))

# --- Logo phải (góc trên bên phải) ---
new_img = None
right_logo_frame = tk.Frame(header_frame, bg="#CBE1FF")
right_logo_frame.pack(side="right", padx=(0, 20), pady=(0, 0), anchor="ne")

new_img_path = "D:\\UTE\\UTE_Nam_4_ki_2_DATN\\Thu_Thap_Data\\Test_Final_Gen_Data\\Interface\\GetArticleImage.jpg"
try:
    if os.path.exists(new_img_path):
        pil_new = Image.open(new_img_path)
        if 'replace_white_background' in globals():
            pil_new = replace_white_background(new_img_path, bg_color=(203, 225, 255))
        pil_new = pil_new.resize((120, 143), Image.Resampling.LANCZOS)
        new_img = ImageTk.PhotoImage(pil_new)
        new_label = tk.Label(right_logo_frame, image=new_img, bg="#CBE1FF")
        new_label.pack()
        # Giữ reference để tránh garbage collection
        new_label.image = new_img
    else:
        raise FileNotFoundError("New image file not found")
except Exception as e:
    new_label = tk.Label(right_logo_frame, text="ẢNH MỚI", bg="#CBE1FF", font=("Arial", 12, "bold"))
    new_label.pack()

# ===== Intro ở giữa =====
intro = tk.Label(   
    root,
    text="HỆ THỐNG THEO DÕI VÀ ĐÓNG GÓI THỰC PHẨM\n"
         "ỨNG DỤNG ROBOT DELTA",
    justify="center",
    font=("Cambria", 22, "bold"),
    bg="#CBE1FF", fg="#222"
)
intro.pack(pady=(30, 10), padx=(50, 0))

# ========== NÚT START ==========
start_button = tk.Button(
    root,
    text="Start",
    command=start_system,
    font=("Cambria", 12),
    width=10, height=2,
    bg="#e2f0f3", fg='black',
    relief='raised', bd=2
)
start_button.pack(pady=10)

# ==== Info text (dưới cùng bên trái) ====
info_frame = tk.Frame(root, bg="#CBE1FF")
info_frame.pack(fill='x', side='bottom', pady=(0, 10))

info_text = (
    "GVHD : ThS. Võ Lâm Chương\n"
    "SVTH  : Nguyễn Minh Khoa    21146112\n"
    "               Lê Nhật Duy                  21146441\n"
    "               Lê Lý Tam                      21146145"
)
info_label = tk.Label(
    info_frame, text=info_text, font=("Cambria", 14),
    fg="#000000", bg="#CBE1FF", justify="left", anchor="w"
)
info_label.pack(side="left", padx=(20, 0), pady=(0, 15))

# ========== NÚT START ==========
start_button = tk.Button(
    root,
    text="Start",
    command=start_system,
    font=("Cambria", 12),
    width=10, height=2,
    bg="#e2f0f3", fg='black',
    relief='raised', bd=2
)
start_button.pack(pady=10)

# ========== SỰ KIỆN ĐÓNG CỬA SỔ ==========
root.protocol("WM_DELETE_WINDOW", on_closing)

# ========== CHẠY GIAO DIỆN ==========
root.mainloop()

# Dừng tất cả threads khi giao diện đóng
running = False
gui_active = False
mc.close()
# Đảm bảo các luồng đã dừng
try:
    cap.release()
    cv2.destroyAllWindows()
except:
    pass
print("🛑 Đã dừng hệ thống.")
