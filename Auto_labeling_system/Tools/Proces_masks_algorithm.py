import numpy as np
import cv2

def remove_background_mask(masks, remove_larger_than_threshold=False, remove_smallest=False, max_area_threshold=0, min_area_threshold=0):
    """
    Lọc danh sách các mask với tùy chọn loại bỏ mask lớn hơn ngưỡng và/hoặc các mask nhỏ hơn ngưỡng
    
    Args:
        masks: Danh sách các mask, mỗi mask là một dictionary
        remove_larger_than_threshold: Boolean chỉ định việc có loại bỏ mask lớn hơn ngưỡng hay không (mặc định là False)
        remove_smallest: Boolean chỉ định việc có loại bỏ các mask nhỏ hơn ngưỡng hay không (mặc định là False)
        area_threshold: Ngưỡng diện tích để xóa mask khi remove_larger_than_threshold=True (mặc định là 0)
        min_area_threshold: Ngưỡng diện tích tối thiểu khi remove_smallest=True (mặc định là 0)
        
    Returns:
        Danh sách các mask sau khi đã lọc theo điều kiện
    """
    if not masks:
        return []
    
    # Áp dụng các điều kiện lọc
    filtered_masks = []
    
    for mask in masks:
        # Kiểm tra điều kiện loại bỏ mask lớn hơn ngưỡng
        if remove_larger_than_threshold and mask['area'] > max_area_threshold:
            continue
            
        # Kiểm tra điều kiện loại bỏ mask nhỏ hơn ngưỡng
        if remove_smallest and mask['area'] < min_area_threshold:
            continue
            
        # Mask thỏa mãn tất cả điều kiện
        filtered_masks.append(mask)
    
    return filtered_masks

def filter_masks_by_squareness(masks, min_squareness_ratio=0.5):
    """
    Lọc bỏ các masks có tỉ lệ hình vuông thấp hơn ngưỡng đặt ra.
    
    Args:
        masks: Danh sách các mask, mỗi mask là dictionary có chứa bbox
        min_squareness_ratio: Ngưỡng tỉ lệ vuông tối thiểu (0.0 đến 1.0)
    
    Returns:
        Danh sách các mask đã được lọc, chỉ giữ lại mask có tỉ lệ vuông >= ngưỡng
    """
    result = []
    
    for mask in masks:
        # Lấy width và height từ bbox [x, y, width, height]
        width = mask['bbox'][2]
        height = mask['bbox'][3]
        
        # Tính tỉ lệ vuông (cạnh ngắn/cạnh dài)
        if width <= 0 or height <= 0:
            continue
            
        longer_side = max(width, height)
        shorter_side = min(width, height)
        squareness = shorter_side / longer_side
        
        # Chỉ giữ lại mask có tỉ lệ vuông >= ngưỡng
        if squareness >= min_squareness_ratio:
            result.append(mask)
    
    return result

def filter_masks_by_area_ratio(masks, min_area_ratio=0.3):
    """
    Lọc bỏ các masks có tỉ lệ diện tích (mask/bbox) thấp hơn ngưỡng đặt ra.
    
    Args:
        masks: Danh sách các mask, mỗi mask là dictionary có chứa 'area' và 'bbox'
        min_area_ratio: Ngưỡng tỉ lệ diện tích tối thiểu (0.0 đến 1.0)
    
    Returns:
        Danh sách các mask đã được lọc, chỉ giữ lại mask có tỉ lệ diện tích >= ngưỡng
    """
    result = []
    
    for mask in masks:
        # Lấy diện tích mask
        mask_area = mask['area']
        
        # Tính diện tích bbox
        bbox_width = mask['bbox'][2]
        bbox_height = mask['bbox'][3]
        bbox_area = bbox_width * bbox_height
        
        # Tính tỉ lệ diện tích mask/bbox
        if bbox_area <= 0:
            continue
            
        area_ratio = mask_area / bbox_area
        
        # Chỉ giữ lại mask có tỉ lệ diện tích >= ngưỡng
        if area_ratio >= min_area_ratio:
            result.append(mask)
    
    return result

def calculate_overlap_ratio(bbox1, bbox2):
    """
    Tính tỷ lệ phần giao của bbox nhỏ hơn so với diện tích của chính nó.
    
    Args:
        bbox1: Bounding box thứ nhất dạng [x, y, width, height]
        bbox2: Bounding box thứ hai dạng [x, y, width, height]
        
    Returns:
        Tỷ lệ diện tích phần giao / diện tích bbox nhỏ hơn
    """
    # Chuyển đổi từ [x, y, width, height] sang [x1, y1, x2, y2]
    x1_1, y1_1, w1, h1 = bbox1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = bbox2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Tính toán diện tích của từng bbox
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Tính toán tọa độ của vùng giao nhau
    x1_intersection = max(x1_1, x1_2)
    y1_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)
    
    # Kiểm tra xem hai bbox có giao nhau không
    if x2_intersection < x1_intersection or y2_intersection < y1_intersection:
        return 0.0
    
    # Tính diện tích phần giao nhau
    intersection_area = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)
    
    # Lấy diện tích của bbox nhỏ hơn
    smaller_area = min(area1, area2)
    
    # Tính tỷ lệ phần giao so với bbox nhỏ hơn
    overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    return overlap_ratio


# Hàm lọc mask dựa trên bounding box với tỷ lệ chồng lấp
def filter_masks_by_bbox_overlap(masks_data, overlap_threshold=0.9):
    """
    Lọc các mask dựa trên việc so sánh chồng lấp giữa các bounding box,
    loại bỏ bounding box nhỏ hơn khi có tỷ lệ chồng lấp vượt ngưỡng.
    
    Args:
        masks_data: List các dictionary chứa thông tin về mask
        overlap_threshold: Ngưỡng tỷ lệ chồng lấp (mặc định là 0.9 tương đương 90%)
        
    Returns:
        List các dictionary mask sau khi đã lọc
    """
    if len(masks_data) <= 1:
        return masks_data
    
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    masks_to_process = masks_data.copy()
    
    # Tính diện tích cho tất cả bounding boxes
    for mask in masks_to_process:
        x, y, w, h = mask['bbox']
        mask['bbox_area'] = w * h
    
    # Sắp xếp masks theo diện tích bounding box từ lớn đến nhỏ
    sorted_masks = sorted(masks_to_process, key=lambda x: x['bbox_area'], reverse=True)
    
    masks_to_keep = []
    masks_to_remove = set()
    
    # So sánh từng cặp bounding box
    for i in range(len(sorted_masks)):
        if i in masks_to_remove:
            continue
            
        current_mask = sorted_masks[i]
        current_bbox = current_mask['bbox']
        
        for j in range(i+1, len(sorted_masks)):
            if j in masks_to_remove:
                continue
                
            compare_mask = sorted_masks[j]
            compare_bbox = compare_mask['bbox']
            
            # Tính tỷ lệ chồng lấp của bbox nhỏ hơn
            overlap_ratio = calculate_overlap_ratio(current_bbox, compare_bbox)
            
            # Nếu tỷ lệ chồng lấp vượt ngưỡng, loại bỏ bbox nhỏ hơn
            if overlap_ratio > overlap_threshold:
                # Vì sorted_masks đã sắp xếp theo diện tích giảm dần,
                # nên bbox tại j luôn nhỏ hơn hoặc bằng bbox tại i
                masks_to_remove.add(j)
    
    # Lấy ra các masks cần giữ lại
    for i in range(len(sorted_masks)):
        if i not in masks_to_remove:
            masks_to_keep.append(sorted_masks[i])
    
    return masks_to_keep


def fill_ring_masks(masks_data, image_shape, add_area=5000):
    """
    Điền đặc phần bên trong của các mask hình vành khuyên để tạo thành hình tròn đặc.
    Chỉ cập nhật area cho những mask thực sự được fill ring.
    
    Args:
        masks_data: List các dictionary chứa thông tin về mask
        image_shape: Tuple chứa kích thước của ảnh (height, width)
        
    Returns:
        List các dictionary mask sau khi đã điền đặc phần trong
    """
    height, width = image_shape
    filled_masks = []
    
    for mask_info in masks_data:
        mask = mask_info['segmentation']
        original_area = mask_info.get('area', np.sum(mask))
        
        # Tìm contours của mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tạo mask mới để vẽ contour đã điền đặc
        filled_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Vẽ tất cả các contours với điền đặc (-1)
        cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)
        
        # Tính diện tích mới sau khi fill
        new_area = np.sum(filled_mask)
        
        # Cập nhật mask trong mask_info
        new_mask_info = mask_info.copy()
        new_mask_info['segmentation'] = filled_mask
        
        # Chỉ cập nhật area nếu có sự thay đổi (tức là mask được fill ring)
        if new_area != original_area:
            new_mask_info['area'] = new_area + add_area  # Cộng thêm diện tích nếu cần
            new_mask_info['is_filled'] = True
        else:
            # Giữ nguyên area ban đầu nếu không có thay đổi
            new_mask_info['is_filled'] = False
        
        filled_masks.append(new_mask_info)
    
    return filled_masks


# Hàm xử lý tổng hợp
def process_masks(masks_data, image_shape, overlap_threshold=0.9, fill_ring=True, add_area=5000):
    """
    Xử lý tổng hợp: lọc bỏ masks nhỏ dựa trên overlap và điền đặc masks vành khuyên
    
    Args:
        masks_data: List các dictionary chứa thông tin về mask
        image_shape: Tuple chứa kích thước của ảnh (height, width)
        overlap_threshold: Ngưỡng tỷ lệ chồng lấp (mặc định là 0.9 tương đương 90%)
        
    Returns:
        List các dictionary mask sau khi đã lọc và điền đặc
    """
    # Bước 1: Lọc bỏ masks dựa trên overlap
    filtered_masks = filter_masks_by_bbox_overlap(masks_data, overlap_threshold)

    # Bước 2: Điền đặc các mask hình vành khuyên
    if fill_ring:
        filled_masks = fill_ring_masks(filtered_masks, image_shape)
    else:
        filled_masks = filtered_masks
    
    return filled_masks




