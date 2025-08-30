def update_tracked_objects(tracks, id_mapping, object_dict):
    """
    Cập nhật dict lưu thông tin các vật thể được theo dõi theo thời gian.
    
    Params:
        - tracks: danh sách track do DeepSORT trả về
        - id_mapping: ánh xạ từ track_id sang custom_id
        - object_dict: dict lưu thông tin theo dõi

    Returns:
        - object_dict đã được cập nhật
    """
    # Đánh dấu tất cả object là mất tích tạm thời
    for obj in object_dict.values():
        obj['status'] = False

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        cls = track.get_det_class()

        # Chuyển bounding box sang int
        x1, y1, x2, y2 = map(int, ltrb)

        # Gán custom ID
        if track_id not in id_mapping:
            id_mapping[track_id] = len(id_mapping) + 1

        custom_id = id_mapping[track_id]

        # Cập nhật thông tin vào object_dict
        object_dict[custom_id] = {
            'id': custom_id,
            'position': [x1, y1, x2, y2],
            'class': cls,
            'status': True  # Đối tượng đang xuất hiện
        }

    return object_dict

if __name__ == "__main__":
    tracks = []  # Danh sách các track từ DeepSORT
    id_mapping = {}  # Ánh xạ từ track_id sang custom_id
    object_dict = {}  # Dict lưu thông tin theo dõi
    
    updated_object_dict = update_tracked_objects(tracks, id_mapping, object_dict)

    print(updated_object_dict)