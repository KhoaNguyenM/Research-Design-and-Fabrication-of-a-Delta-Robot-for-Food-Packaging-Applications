import pymcprotocol
import struct
import time
# from delta_robot import delta_calcInverse

import math
# Trigonometric constants
sqrt3 = math.sqrt(3.0)
pi = 3.141592653
sin120 = sqrt3 / 2.0
cos120 = -0.5
tan60 = sqrt3
sin30 = 0.5
tan30 = 1 / sqrt3
# Robot geometry (mm)
Up =50  
Wb =150  # base radius
f=Wb*6/sqrt3
e=Up*3/sqrt3  
re = 463  # end to elbow distance
rf =250   # base to elbow distance



# Forward kinematics: (theta1, theta2, theta3) -> (x0, y0, z0) in mm
# Returned status: 0=OK, -1=non-existing position
def delta_calcForward(theta1, theta2, theta3):
    t = (Wb - Up) 
    dtr = pi / 180.0
    
    theta1 *= dtr
    theta2 *= dtr
    theta3 *= dtr
    
    y1 = -(t + rf * math.cos(theta1))
    z1 = -rf * math.sin(theta1)
    
    y2 = (t + rf * math.cos(theta2)) * sin30
    x2 = y2 * tan60
    z2 = -rf * math.sin(theta2)
    
    y3 = (t + rf * math.cos(theta3)) * sin30
    x3 = -y3 * tan60
    z3 = -rf * math.sin(theta3)
    
    dnm = (y2 - y1) * x3 - (y3 - y1) * x2
    
    w1 = y1 * y1 + z1 * z1
    w2 = x2 * x2 + y2 * y2 + z2 * z2
    w3 = x3 * x3 + y3 * y3 + z3 * z3
    
    # x = (a1*z + b1)/dnm
    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0
    
    # y = (a2*z + b2)/dnm
    a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0
    
    # a*z^2 + b*z + c = 0
    a = a1 * a1 + a2 * a2 + dnm * dnm
    b = 2 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm * dnm)
    c = (b2 - y1 * dnm) * (b2 - y1 * dnm) + b1 * b1 + dnm * dnm * (z1 * z1 - re * re)
    
    # Discriminant
    d = b * b - 4.0 * a * c
    if d < 0:
        return (-1, 0, 0, 0)  # non-existing point
    
    z0 = -0.5 * (b + math.sqrt(d)) / a
    x0 = (a1 * z0 + b1) / dnm
    y0 = (a2 * z0 + b2) / dnm
    return (0, x0, y0, z0)

# Inverse kinematics helper function, calculates angle theta1 (for YZ-plane) in degrees
def delta_calcAngleYZ(x0, y0, z0):
    y1 = -0.5 * 0.57735 * f  # f/2 * tan(30)
    y0 -= Up  # shift center to edge
    # z = a + b*y
    #a = (x0 * x0 + y0 * y0 + z0 * z0 + rf * rf - re * re - y1 * y1) / (2 * z0)
    a = ( x0 * x0+ y0 * y0 + z0 * z0 + rf * rf - re * re - y1 * y1) / (2 * z0)
    b = (y1 - y0) / z0
    # Discriminant
    d = -(a + b * y1) * (a + b * y1) + rf * (b * b * rf + rf)
    if d < 0:
        return (-1, 0)  # non-existing point
    yj = (y1 - a * b - math.sqrt(d)) / (b * b + 1)  # choosing outer point
    zj = a + b * yj
    if yj > y1:
        theta = 180.0 * math.atan(-zj / (y1 - yj)) / pi + 180.0
    else:
        theta = 180.0 * math.atan(-zj / (y1 - yj)) / pi
    return (0, theta)

# Inverse kinematics: (x0, y0, z0) -> (theta1, theta2, theta3) in degrees
# Returned status: 0=OK, -1=non-existing position
def delta_calcInverse(x0, y0, z0):
    theta1 = theta2 = theta3 = 0.0
    status, theta1 = delta_calcAngleYZ(x0, y0, z0)
    
    if status == 0:
        status, theta2 = delta_calcAngleYZ(x0 * cos120 + y0 * sin120, y0 * cos120 - x0 * sin120, z0)  # rotate coords to +120 deg
    
    if status == 0:
        status, theta3 = delta_calcAngleYZ(x0 * cos120 - y0 * sin120, y0 * cos120 + x0 * sin120, z0)  # rotate coords to -120 deg
    
    return (theta1 - 2.1327, theta2 - 2.1327, theta3 - 2.1327)

def write_float_to_words(value):
    data_bytes = struct.pack('>f', value)  # '>f' hoặc '<f' tùy PLC/driver
    high_word = int.from_bytes(data_bytes[:2], byteorder='big', signed=False)
    low_word  = int.from_bytes(data_bytes[2:], byteorder='big', signed=False)
    # Chuyển về signed 16 bit nếu cần
    def to_signed(w):
        return w if w < 32768 else w - 65536
    return [to_signed(low_word), to_signed(high_word)]

def read_words_to_float(data):
    low_word = data[0] & 0xFFFFFFFF
    high_word = data[1] & 0xFFFFFFFF
    value = (high_word << 16) | low_word
    data_bytes = value.to_bytes(4, byteorder='big')
    return struct.unpack('>f', data_bytes)[0]

def write_dword_to_wordlist(value):
    if not (0 <= value <= 0xFFFFFFFF):
        raise ValueError("Giá trị vượt quá phạm vi 32-bit unsigned.")

    def to_signed16(val):
        return val - 0x10000 if val > 32767 else val

    low_word = to_signed16(value & 0xFFFF)
    high_word = to_signed16((value >> 16) & 0xFFFF)
    return [low_word, high_word]


def reset_flag():
    mc.batchwrite_bitunits(headdevice="Y43", values=[0])


def write_position_2(a, b, c):
    mc.batchwrite_wordunits(headdevice="D800", values=write_float_to_words(a))
    mc.batchwrite_wordunits(headdevice="D805", values=write_float_to_words(b))
    mc.batchwrite_wordunits(headdevice="D810", values=write_float_to_words(c))

def write_position_3():
    mc.batchwrite_wordunits(headdevice="D135", values=write_float_to_words(0.0))
    mc.batchwrite_wordunits(headdevice="D140", values=write_float_to_words(0.0))
    mc.batchwrite_wordunits(headdevice="D145", values=write_float_to_words(0.0))

def execute_step(position_a, position_b, position_c, step_name):
    t = 0.003
    """Thực hiện một bước di chuyển"""
    write_position_2(position_a, position_b, position_c)
    mc.batchwrite_bitunits(headdevice="M310", values=[1])
    mc.batchwrite_bitunits(headdevice="M255", values=[1])
    time.sleep(t)
    mc.batchwrite_bitunits(headdevice="M310", values=[0])
    mc.batchwrite_bitunits(headdevice="M255", values=[0])
    print(f"Completed: {step_name}")

def update_position():
    mc.batchwrite_bitunits(headdevice="M255", values=[1])
    time.sleep(0.003)
    mc.batchwrite_bitunits(headdevice="M255", values=[0])

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

    # mc.batchwrite_bitunits(headdevice="M111", values=[1])

    # print(f"Calculated Inverse Kinematics: a={a1}, b={b1}, c={c1}")
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

def point_to_point(x,y,z):
    a, b, c = delta_calcInverse(x, y, z)
    a = int(a)*700
    b = int(b)*700
    c = int(c)*700

    mc.batchwrite_wordunits(headdevice="D1400", values=write_dword_to_wordlist(a))
    mc.batchwrite_wordunits(headdevice="D1405", values=write_dword_to_wordlist(b))
    mc.batchwrite_wordunits(headdevice="D1410", values=write_dword_to_wordlist(c))

    mc.batchwrite_bitunits(headdevice="M999", values=[1])



mc = pymcprotocol.Type3E()

# Kết nối đến IP của PLC (là IP bạn gán cho PLC trong phần Ethernet Operation)
# mc.connect('192.168.10.100', 3000)

# if mc._is_connected:
#     print("✅ Kết nối thành công!")
# else:
#     print("❌ Kết nối thất bại!")

# import msvcrt

# mc.batchwrite_wordunits(headdevice="D500", values=write_float_to_words(0.25))

# # KHỞI TẠO BIẾN started
# started = False
# co = 0
# cycle_running = False

# print("Nhấn 's' để bắt đầu, 'q' để thoát")

# while True:
#     # Xử lý phím nhấn
#     if msvcrt.kbhit():
#         key = msvcrt.getch().decode('utf-8').lower()
#         if key == 's':
#             started = True
#             ver = 0
#             cycle_running = False
#             co = 0
#             current_target = None
#             print("Bắt đầu chạy state machine!")
#             start_point(0,0,-400.0)
#         elif key == 'q':
#             print("Thoát chương trình.")
#             break

#     # Chỉ chạy khi đã started
#     if started:

#         flag_pos = mc.batchread_bitunits(headdevice="M96", readsize=1)[0]
#         flag_pos_ver1 = mc.batchread_bitunits(headdevice="M2999", readsize=1)[0] # M3001 off
#         flag_pos_ver2 = mc.batchread_bitunits(headdevice="M93", readsize=1)[0]   # M3002 off

#         # Kiểm tra flag = 0 và có object class_0
#         if ver == 0 and (flag_pos == 1 or flag_pos_ver2 == 1) and not cycle_running:
#             print("###################################33")
#             calculate_inverse_kinematics_pick(0.0, 150.0, -450.0, 55.0)
#             mc.batchwrite_bitunits(headdevice="M310", values=[1])
#             mc.batchwrite_bitunits(headdevice="M3002", values=[1])
#             ver = 3
#             print(f"Phát hiện class_0 object, bắt đầu chu trình!")
        
#         # Kiểm tra flag = 3 và có object class_1
#         elif ver == 3 and flag_pos_ver1 == 1 and not cycle_running:
#             calculate_inverse_kinematics_place(0.0, -100.0, -450.0, 50.0)  # Tính toán lại vị trí đặt
#             mc.batchwrite_bitunits(headdevice="M3001", values=[1])
#             mc.batchwrite_bitunits(headdevice="M325", values=[1])
#             time.sleep(0.005)
#             mc.batchwrite_bitunits(headdevice="M325", values=[0])
#             print("22222222222222222222222222222222")
#             ver = 0

#     time.sleep(0.001)  # Sleep 50ms


# mc.close()