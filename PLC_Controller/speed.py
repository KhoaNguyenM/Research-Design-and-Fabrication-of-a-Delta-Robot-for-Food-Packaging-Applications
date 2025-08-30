import math

def calculate_command_speed(distance, total_time, angle_mode=False):
    """
    Tính command speed (xung/giây) cho mô hình hình thang vận tốc.
    Parameters:
        distance: Quãng đường (xung) hoặc góc (độ, nếu angle_mode=True).
        total_time: Thời gian thực hiện (giây).
        angle_mode: True nếu distance là góc (độ), False nếu distance là xung.
    Returns:
        Command speed (xung/giây) hoặc None nếu thông số không khả thi.
    """
    # Thông số cố định
    pulses_per_revolution = 262144  # xung/vòng
    max_speed  = 13107200# xung/giây (50 vòng/giây)
    accel_time = 0.1  # giây
    acceleration = max_speed / accel_time  # xung/giây^2

    # Chuyển góc thành xung nếu angle_mode=True
    if angle_mode:
        distance = (distance * pulses_per_revolution) / 360

    # Giải phương trình bậc hai: v^2 - a*t*v + a*s = 0
    a = 1
    b = -acceleration * total_time
    c = acceleration * distance
    delta = b**2 - 4*a*c

    if delta < 0:
        print("Không có nghiệm thực! Kiểm tra lại thông số.")
        return None

    # Chọn nghiệm nhỏ hơn
    v = (-b - math.sqrt(delta)) / (2*a)

    # Kiểm tra tính hợp lý
    t1 = v / acceleration  # Thời gian tăng tốc
    t2 = total_time - 2 * t1  # Thời gian tốc độ không đổi
    if t2 < 0:
        print("Thời gian quá ngắn, cần dùng mô hình tam giác vận tốc!")
        return None
    if v > max_speed:
        print("Tốc độ vượt quá giới hạn tối đa!")
        return None

    # Tính quãng đường để kiểm tra
    calc_distance = v * (t1 + t2)
    print(f"Command speed: {v:.0f} xung/giây")
    print(f"Thời gian tăng tốc: {t1*1000:.2f} ms")
    print(f"Thời gian tốc độ không đổi: {t2*1000:.2f} ms")
    print(f"Quãng đường tính được: {calc_distance:.0f} xung")
    return v

# Hàm chính để chạy chương trình
def main():
    # Nhập liệu từ người dùng
    mode = input("Nhập 'angle' để dùng góc (độ) hoặc 'pulse' để dùng xung: ").strip().lower()
    angle_mode = (mode == 'angle')

    if angle_mode:
        distance = float(input("Nhập góc quay (độ): "))
    else:
        distance = float(input("Nhập quãng đường (xung): "))
    
    total_time = float(input("Nhập thời gian thực hiện (giây): "))

    # Gọi hàm tính toán
    calculate_command_speed(distance, total_time, angle_mode)

# Chạy chương trình
if __name__ == "__main__":
    main()