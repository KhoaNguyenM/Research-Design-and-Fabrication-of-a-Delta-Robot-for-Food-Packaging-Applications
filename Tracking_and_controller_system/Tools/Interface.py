from PIL import Image


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