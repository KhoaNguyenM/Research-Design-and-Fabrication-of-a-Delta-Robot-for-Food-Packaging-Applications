import pymcprotocol
import struct
import time
import msvcrt
from delta_robot import delta_calcAngleYZ, delta_calcInverse
from speed  import calculate_command_speed


def write_float_to_words(value):
    data_bytes = struct.pack('>f', value)  # '>f' hoặc '<f' tùy PLC/driver
    high_word = int.from_bytes(data_bytes[:2], byteorder='big', signed=False)
    low_word  = int.from_bytes(data_bytes[2:], byteorder='big', signed=False)
    # Chuyển về signed 16 bit nếu cần
    def to_signed(w):
        return w if w < 32768 else w - 65536
    return [to_signed(low_word), to_signed(high_word)]

def read_words_to_float(data):
    low_word = data[0] & 0xFFFF # Từ thấp
    high_word = data[1] & 0xFFFF # Từ cao
    value = (high_word << 16) | low_word
    data_bytes = value.to_bytes(4, byteorder='big')
    return struct.unpack('>f', data_bytes)[0]

mc = pymcprotocol.Type3E(plctype="Q")
mc = pymcprotocol.Type3E()

# Kết nối đến IP của PLC (là IP bạn gán cho PLC trong phần Ethernet Operation)
mc.connect('192.168.10.100', 3000)

if mc._is_connected:
    print("✅ Kết nối thành công!")
else:
    print("❌ Kết nối thất bại!")


def write_position_2(a, b, c):
    mc.batchwrite_wordunits(headdevice="D800", values=write_float_to_words(a))
    mc.batchwrite_wordunits(headdevice="D805", values=write_float_to_words(b))
    mc.batchwrite_wordunits(headdevice="D810", values=write_float_to_words(c))

def write_position_3():
    mc.batchwrite_wordunits(headdevice="D135", values=write_float_to_words(0.0))
    mc.batchwrite_wordunits(headdevice="D140", values=write_float_to_words(0.0))
    mc.batchwrite_wordunits(headdevice="D145", values=write_float_to_words(0.0))

def write_position_(a, b, c):
    mc.batchwrite_wordunits(headdevice="D800", values=write_float_to_words(a))
    mc.batchwrite_wordunits(headdevice="D805", values=write_float_to_words(b))
    mc.batchwrite_wordunits(headdevice="D810", values=write_float_to_words(c))

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



mc.batchwrite_wordunits(headdevice="D500", values=write_float_to_words(0.5))

print("Nhấn 's' để bắt đầu, 'q' để thoát")

write_position_3()

mc.batchwrite_bitunits(headdevice="M202", values=[0])  # Reset M202
mc.batchwrite_bitunits(headdevice="M202", values=[1])  # Bật M202
time.sleep(0.5)
mc.batchwrite_bitunits(headdevice="M202", values=[0])  # Tắt M202")

a = 0.0
b = 0.0
c = -500.0

theta1_before, theta2_before, theta3_before = 0.0, 0.0, 0.0 # Vị trí ban đầu

status, theta1, theta2, theta3 = delta_calcInverse(a, b, c)

delta_theta1 = abs(theta1 - theta1_before)
# delta_theta2 = abs(theta2 - theta2_before)
# delta_theta3 = abs(theta3 - theta3_before)

spd = calculate_command_speed(delta_theta1, 0.5, angle_mode=True)

theta1_before, theta2_before, theta3_before = theta1, theta2, theta3  # Cập nhật vị trí ban đầu



print(f"Inverse Kinematics: theta1 = {theta1:.2f}°, theta2 = {theta2:.2f}°, theta3 = {theta3:.2f}°")
print(f"Command speed: {spd:.5f} xung/giây")



# mc.close()