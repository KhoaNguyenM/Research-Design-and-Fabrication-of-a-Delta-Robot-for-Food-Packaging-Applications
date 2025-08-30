def crop_frame(frame, x, y, width, height):
    """
    Cắt một phần của frame.
    """
    h, w = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x_end = min(x + width, w)
    y_end = min(y + height, h)
    return frame[y:y_end, x:x_end]

if __name__ == "__main__":

    frame = 0

    # Cắt frame: ví dụ lấy vùng 200x200 từ góc (100, 100)
    frame = crop_frame(frame, 100, 100, 200, 200)
