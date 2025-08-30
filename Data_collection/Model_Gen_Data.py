import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
import argparse
from collections import Counter

from Tools import remove_background_mask, filter_masks_by_squareness, process_masks
from Tools import label_masks_from_examples, compare_and_label_mask, extract_mask_features

# Define global constants
MODEL_CFG = "D:/UTE/UTE_Nam_4_ki_2_DATN/Model&code_SAM/Model_SAM2_JinsuaFeito/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = "D:/UTE/UTE_Nam_4_ki_2_DATN/Model&code_SAM/Model_SAM2_JinsuaFeito/sam2.1_hiera_base_plus.pt"

FOLDER_LABELS = "D:/UTE/UTE_Nam_4_ki_2_DATN/Thu_Thap_Data"

crop1 =  np.array([[46, 3], [271, 3], [271, 355], [46, 355]])
crop2 = np.array([[425, 6], [562, 6], [562, 358], [425, 358]])

BOX_NMS_THRESH_SET = 0.7
PRED_IOU_THRESH_SET = 0.7
STABILITY_SCORE_THRESH_SET = 0.7

MAX_AREA_THRESHOLD_SET = 12000
MIN_AREA_THRESHOLD_SET = 4000

MIN_SQUARNESS_RATIO = 0.85  # Minimum squareness ratio for masks to be considered valid

ADD_AREA = 5000

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def display_masks_for_labeling(image, masks):
    """
    Display masks with numbers for manual labeling (non-blocking)
    """
    import colorsys
    
    # Create a copy of the image to draw on
    result_image = image.copy()
    
    # Draw contours and numbers for each mask
    for i, mask_dict in enumerate(masks):
        # Create a unique color for this mask
        hue = i / max(1, len(masks))
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        
        mask = mask_dict['segmentation']
        
        # Find mask contours
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours
        cv2.drawContours(result_image, contours, -1, color, 2)
        
        # Find position to display number
        x, y, w, h = mask_dict['bbox']
        cv2.putText(
            result_image, 
            str(i), 
            (int(x), int(y) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            color, 
            2
        )
    
    # Display the image in a way that doesn't block
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(result_image)
    plt.title('Objects (Enter labels for each numbered object)')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()  # Draw but don't block
    plt.pause(0.001)  # Small pause to ensure drawing happens
    
    return fig, result_image

def get_manual_labels(root, masks, fig):
    """Get manual labels for each mask from user input and close figure when done."""
    labels = {}
    class_labels = set()
    
    for i in range(len(masks)):
        label = simpledialog.askstring("Input", f"Enter label for object #{i}:", parent=root)
        if label is None or label.strip() == "":
            label = f"object{i}"  # Default label if none provided
        
        labels[i] = label
        class_labels.add(label)
    
    # Close the figure after all labels are entered
    plt.close(fig)
    
    return labels, class_labels

def save_yolo_labels(masks, labels_map, img_width, img_height, output_path, class_mapping):
    """
    Save labels in YOLO format
    """
    with open(output_path, 'w') as f:
        for i, mask_dict in enumerate(masks):
            if i not in labels_map:
                continue
                
            # Get bounding box and label
            x, y, w, h = mask_dict['bbox']
            label = labels_map[i]
            class_idx = class_mapping[label]
            
            # Convert to YOLO format
            # YOLO format: <class_idx> <center_x> <center_y> <width> <height>
            # Where all values except class_idx are normalized to 0-1
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # Write to file
            f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def process_images(image_files, labels_folder):
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

    print(f"Selected reference image: {os.path.basename(first_img_path)}")
    
    # Load first image using OpenCV instead of PIL
    try:
        first_image = cv2.imread(first_img_path)
        if first_image is None:
            raise Exception("Failed to load image")
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
    
    # Display masks with numbers for labeling
    fig, display_img = display_masks_for_labeling(first_image, first_masks)
    
    # Get manual labels from user
    root = tk.Tk()
    root.withdraw()
    
    print("\nEnter labels for each numbered object in the displayed image:")
    manual_labels, class_set = get_manual_labels(root, first_masks, fig)
    
    # Create class mapping
    class_list = sorted(list(class_set))
    class_mapping = {label: i for i, label in enumerate(class_list)}
    
    print(f"Class mapping: {class_mapping}")
    
    # Save YOLO labels for first image
    base_filename = os.path.splitext(os.path.basename(first_img_path))[0]
    label_file = os.path.join(labels_folder, f"{base_filename}.txt")
    save_yolo_labels(first_masks, manual_labels, first_img_width, first_img_height, label_file, class_mapping)
    
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
                # Convert BGR to RGB (OpenCV loads as BGR, but we need RGB for processing)
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
                print(f"No valid masks found for {img_path}")
                try:
                    os.remove(img_path)
                    print(f"Deleted image without valid masks: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"Error deleting {os.path.basename(img_path)}: {e}")
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
            
            # Create YOLO format labels
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            label_file = os.path.join(labels_folder, f"{base_filename}.txt")
            save_yolo_labels(masks, auto_labels, img_width, img_height, label_file, class_mapping)
    
    # Save class mapping to a file
    with open(os.path.join(labels_folder, "classes.txt"), 'w') as f:
        for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    
    print(f"\nProcessing complete! Labels saved to {labels_folder}")
    print(f"Class mapping saved to {os.path.join(labels_folder, 'classes.txt')}")
    print("Advanced color comparison with LAB and HSV color spaces was used for automatic labeling.")

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
    
    # Create labels folder
    labels_folder = os.path.join(FOLDER_LABELS, "labels")
    
    # Process all images
    process_images(image_files, labels_folder)

if __name__ == "__main__":
    main()
