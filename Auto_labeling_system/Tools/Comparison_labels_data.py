import numpy as np
import cv2

def rgb_to_lab(rgb):
    """
    Chuyển đổi màu từ RGB sang LAB color space
    LAB color space phản ánh tốt hơn sự nhận biết màu sắc của mắt người
    
    Parameters:
    -----------
    rgb : numpy.ndarray
        Giá trị RGB (0-255)
    
    Returns:
    --------
    numpy.ndarray
        Giá trị LAB
    """
    # Chuẩn hóa RGB về [0,1]
    rgb_normalized = rgb / 255.0
    
    # Chuyển đổi sang LAB
    # Tạo ảnh 1x1 pixel để sử dụng cv2.cvtColor
    rgb_pixel = np.uint8([[rgb]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    
    return lab_pixel[0][0].astype(np.float32)

def rgb_to_hsv(rgb):
    """
    Chuyển đổi màu từ RGB sang HSV color space
    HSV tách biệt hue (màu sắc), saturation (độ bão hòa), value (độ sáng)
    
    Parameters:
    -----------
    rgb : numpy.ndarray
        Giá trị RGB (0-255)
    
    Returns:
    --------
    numpy.ndarray
        Giá trị HSV
    """
    rgb_pixel = np.uint8([[rgb]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)
    return hsv_pixel[0][0].astype(np.float32)

def delta_e_cie76(lab1, lab2):
    """
    Tính Delta E CIE 1976 - thước đo chuẩn cho sự khác biệt màu sắc
    
    Parameters:
    -----------
    lab1, lab2 : numpy.ndarray
        Giá trị màu trong LAB color space
    
    Returns:
    --------
    float
        Giá trị Delta E (càng nhỏ thì màu càng giống)
    """
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

def hsv_distance_weighted(hsv1, hsv2):
    """
    Tính khoảng cách màu trong HSV với trọng số khác nhau cho H, S, V
    
    Parameters:
    -----------
    hsv1, hsv2 : numpy.ndarray
        Giá trị màu trong HSV color space
    
    Returns:
    --------
    float
        Khoảng cách có trọng số
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    
    # Tính khoảng cách cho Hue (circular distance)
    hue_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
    hue_distance = hue_diff / 180.0
    
    # Tính khoảng cách cho Saturation và Value
    sat_distance = abs(s1 - s2) / 255.0
    val_distance = abs(v1 - v2) / 255.0
    
    # Trọng số: Hue quan trọng nhất, sau đó đến Saturation, cuối cùng là Value
    weighted_distance = (0.5 * hue_distance + 
                        0.3 * sat_distance + 
                        0.2 * val_distance)
    
    return weighted_distance

def advanced_color_distance(color1, color2, method='lab'):
    """
    Tính khoảng cách màu sắc tiên tiến
    
    Parameters:
    -----------
    color1, color2 : numpy.ndarray
        Giá trị màu RGB (0-255)
    method : str
        Phương pháp so sánh ('lab', 'hsv', 'combined')
    
    Returns:
    --------
    float
        Khoảng cách màu sắc
    """
    if method == 'lab':
        lab1 = rgb_to_lab(color1)
        lab2 = rgb_to_lab(color2)
        return delta_e_cie76(lab1, lab2) / 100.0  # Chuẩn hóa về [0,1]
    
    elif method == 'hsv':
        hsv1 = rgb_to_hsv(color1)
        hsv2 = rgb_to_hsv(color2)
        return hsv_distance_weighted(hsv1, hsv2)
    
    elif method == 'combined':
        # Kết hợp cả LAB và HSV
        lab_distance = advanced_color_distance(color1, color2, 'lab')
        hsv_distance = advanced_color_distance(color1, color2, 'hsv')
        return 0.7 * lab_distance + 0.3 * hsv_distance
    
    else:
        raise ValueError("Method phải là 'lab', 'hsv', hoặc 'combined'")

def calculate_color_features_enhanced(image, mask):
    """
    Phiên bản cải tiến của calculate_color_features với nhiều thông tin màu sắc hơn
    
    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh gốc (RGB format)
    mask : numpy.ndarray
        Mask nhị phân
    
    Returns:
    --------
    dict
        Dictionary chứa các đặc trưng màu sắc
    """
    # Lấy các pixel thuộc vật thể
    object_pixels = image[mask > 0]
    
    if len(object_pixels) == 0:
        return {
            'avg_color': np.zeros(3),
            'dominant_color': np.zeros(3),
            'color_std': np.zeros(3)
        }
    
    # Màu trung bình
    avg_color = np.mean(object_pixels, axis=0)
    
    # Màu chi phối (sử dụng K-means hoặc histogram)
    # Đơn giản hóa: lấy median
    dominant_color = np.median(object_pixels, axis=0)
    
    # Độ lệch chuẩn màu sắc (đo độ đồng nhất màu)
    color_std = np.std(object_pixels, axis=0)
    
    return {
        'avg_color': avg_color,
        'dominant_color': dominant_color,
        'color_std': color_std
    }

# Hàm cập nhật cho phần so sánh trong label_masks_from_examples
def compare_colors_enhanced(features1, features2, method='combined'):
    """
    So sánh đặc trưng màu sắc giữa hai vật thể
    
    Parameters:
    -----------
    features1, features2 : dict
        Dictionary chứa đặc trưng màu sắc
    method : str
        Phương pháp so sánh màu
    
    Returns:
    --------
    float
        Khoảng cách màu sắc (càng nhỏ càng giống)
    """
    # So sánh màu trung bình
    avg_distance = advanced_color_distance(
        features1['avg_color'], 
        features2['avg_color'], 
        method
    )
    
    # So sánh màu chi phối
    dom_distance = advanced_color_distance(
        features1['dominant_color'], 
        features2['dominant_color'], 
        method
    )
    
    # Kết hợp cả hai với trọng số
    combined_distance = 0.6 * avg_distance + 0.4 * dom_distance
    
    return combined_distance

def extract_mask_features(image, mask_dict):
    """
    Trích xuất các đặc trưng từ một mask (PHIÊN BẢN ĐÚNG)
    
    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh gốc (RGB format)
    mask_dict : dict
        Dictionary chứa thông tin về mask
    
    Returns:
    --------
    dict
        Dictionary chứa các đặc trưng của mask
    """
    # Lấy mask và thông tin
    mask = mask_dict['segmentation']
    area = mask_dict['area']
    
    # Tính đặc trưng màu sắc CẢI TIẾN
    color_features = calculate_color_features_enhanced(image, mask)
    
    return {
        'area': area,
        **color_features  # Unpack tất cả đặc trưng màu sắc
    }

def label_masks_from_examples(image, masks, labeled_indices, labels,
                              area_weight=0.4, color_weight=0.6):
    """
    Gán nhãn cho masks dựa trên các mask mẫu đã được gán nhãn (PHIÊN BẢN CẢI TIẾN)
    
    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh gốc (RGB format)
    masks : list
        Danh sách các mask (dictionaries)
    labeled_indices : list
        Chỉ số của các mask đã được gán nhãn làm mẫu
    labels : dict
        Dictionary ánh xạ từ chỉ số mask đến nhãn tương ứng
    area_weight : float
        Trọng số cho đặc trưng diện tích
    color_weight : float
        Trọng số cho đặc trưng màu sắc
    
    Returns:
    --------
    dict
        Dictionary ánh xạ từ chỉ số mask đến nhãn được gán
    """
    # Tạo dictionary ánh xạ từ chỉ số mask đến nhãn
    mask_labels = {}
    for i, idx in enumerate(labeled_indices):
        if idx >= len(masks):
            raise ValueError(f"Chỉ số mask {idx} vượt quá số lượng mask có sẵn ({len(masks)})")
        if i not in labels:
            raise ValueError(f"Không tìm thấy nhãn cho vị trí {i} trong dictionary labels")
        mask_labels[idx] = labels[i]
    
    # Trích xuất đặc trưng cho các mask mẫu
    reference_features = {}
    for idx in labeled_indices:
        label = mask_labels[idx]
        features = extract_mask_features(image, masks[idx])
        
        if label not in reference_features:
            reference_features[label] = []
        reference_features[label].append(features)
    
    # Tính toán giá trị đặc trưng trung bình cho mỗi nhãn
    label_references = {}
    for label, features_list in reference_features.items():
        avg_area = np.mean([f['area'] for f in features_list])
        avg_color = np.mean([f['avg_color'] for f in features_list], axis=0)
        dominant_color = np.mean([f['dominant_color'] for f in features_list], axis=0)
        
        label_references[label] = {
            'area': avg_area,
            'avg_color': avg_color,
            'dominant_color': dominant_color
        }
    
    # Chuẩn hóa giá trị diện tích để có phạm vi từ 0 đến 1
    all_areas = [f['area'] for features in reference_features.values() for f in features]
    min_area, max_area = min(all_areas), max(all_areas)
    area_range = max_area - min_area
    
    if area_range > 0:
        for label, ref in label_references.items():
            ref['normalized_area'] = (ref['area'] - min_area) / area_range
    else:
        for label, ref in label_references.items():
            ref['normalized_area'] = 0.5
    
    # Gán nhãn cho tất cả masks
    result_labels = {}
    
    # Sao chép nhãn đã biết trước
    for idx in labeled_indices:
        result_labels[idx] = mask_labels[idx]
    
    # Gán nhãn cho các mask chưa biết
    for i, mask_dict in enumerate(masks):
        if i in labeled_indices:
            continue  # Bỏ qua mask đã được gán nhãn
        
        # Trích xuất đặc trưng của mask hiện tại
        features = extract_mask_features(image, mask_dict)
        
        # Chuẩn hóa diện tích
        if area_range > 0:
            normalized_area = (features['area'] - min_area) / area_range
        else:
            normalized_area = 0.5
        
        # So sánh với các nhãn đã biết
        min_distance = float('inf')
        best_label = None
        
        for label, ref in label_references.items():
            # So sánh diện tích
            area_distance = abs(normalized_area - ref['normalized_area'])
            
            # So sánh màu sắc CẢI TIẾN
            color_distance = compare_colors_enhanced(features, ref, method='combined')
            
            # Tính tổng khoảng cách có trọng số
            total_distance = area_weight * area_distance + color_weight * color_distance
            
            if total_distance < min_distance:
                min_distance = total_distance
                best_label = label
        
        result_labels[i] = best_label
    
    return result_labels

def compare_and_label_mask(reference_image, reference_masks, labeled_indices, base_labels,
                           new_image, new_masks, area_weight=0.4, color_weight=0.6):
    """
    So sánh và gán nhãn cho các mask trong ảnh mới dựa trên các mẫu đã được gán nhãn từ ảnh tham chiếu
    
    Parameters:
    -----------
    reference_image : numpy.ndarray
        Ảnh tham chiếu gốc (RGB format)
    reference_masks : list
        Danh sách các mask của ảnh tham chiếu
    labeled_indices : list
        Chỉ số của các mask đã được gán nhãn làm mẫu
    base_labels : dict
        Dictionary ánh xạ từ vị trí trong labeled_indices đến nhãn tương ứng
    new_image : numpy.ndarray
        Ảnh mới cần gán nhãn (RGB format)
    new_masks : list
        Danh sách các mask của ảnh mới
    area_weight : float, optional
        Trọng số cho đặc trưng diện tích, mặc định là 0.4
    color_weight : float, optional
        Trọng số cho đặc trưng màu sắc, mặc định là 0.6
    
    Returns:
    --------
    dict
        Dictionary ánh xạ từ chỉ số mask đến nhãn được gán trong ảnh mới
    """
    # Tạo reference_labels từ labeled_indices và base_labels
    reference_labels = {}
    for i, idx in enumerate(labeled_indices):
        if i in base_labels:
            reference_labels[idx] = base_labels[i]
    
    # Trích xuất đặc trưng cho các mask mẫu từ ảnh tham chiếu
    reference_features = {}
    for idx, label in reference_labels.items():
        features = extract_mask_features(reference_image, reference_masks[idx])
        
        if label not in reference_features:
            reference_features[label] = []
        reference_features[label].append(features)
    
    # Tính toán giá trị đặc trưng trung bình cho mỗi nhãn
    label_references = {}
    for label, features_list in reference_features.items():
        avg_area = np.mean([f['area'] for f in features_list])
        avg_color = np.mean([f['avg_color'] for f in features_list], axis=0)
        dominant_color = np.mean([f['dominant_color'] for f in features_list], axis=0)
        
        label_references[label] = {
            'area': avg_area,
            'avg_color': avg_color,
            'dominant_color': dominant_color
        }
    
    # Chuẩn hóa giá trị diện tích để có phạm vi từ 0 đến 1
    all_areas = [f['area'] for features in reference_features.values() for f in features]
    
    if len(all_areas) == 0:
        raise ValueError("Không có mask mẫu nào để tham chiếu")
    
    min_area, max_area = min(all_areas), max(all_areas)
    area_range = max_area - min_area
    
    if area_range > 0:
        for label, ref in label_references.items():
            ref['normalized_area'] = (ref['area'] - min_area) / area_range
    else:
        for label, ref in label_references.items():
            ref['normalized_area'] = 0.5
    
    # Gán nhãn cho tất cả masks trong ảnh mới
    result_labels = {}
    
    # Trích xuất và gán nhãn cho từng mask trong ảnh mới
    for i, mask_dict in enumerate(new_masks):
        # Trích xuất đặc trưng của mask hiện tại
        features = extract_mask_features(new_image, mask_dict)
        
        # Chuẩn hóa diện tích
        if area_range > 0:
            normalized_area = (features['area'] - min_area) / area_range
        else:
            normalized_area = 0.5
        
        # So sánh với các nhãn đã biết
        min_distance = float('inf')
        best_label = None
        
        for label, ref in label_references.items():
            # So sánh diện tích
            area_distance = abs(normalized_area - ref['normalized_area'])
            
            # So sánh màu sắc CẢI TIẾN
            color_distance = compare_colors_enhanced(features, ref, method='combined')
            
            # Tính tổng khoảng cách có trọng số
            total_distance = area_weight * area_distance + color_weight * color_distance
            
            if total_distance < min_distance:
                min_distance = total_distance
                best_label = label
        
        result_labels[i] = best_label
    
    return result_labels

