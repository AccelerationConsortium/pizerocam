import os
import cv2
import numpy as np
from pathlib import Path
from google.cloud import vision
import itertools
import argparse
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Google Cloud credentials if provided in .env
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# --- CONFIG ---
# Path to the image to analyze
IMAGE_PATH = "photos-2025-03-26-pH/capture_20250714-182920_100100100.jpg"  # Change to your image path

# --- STEP 1: OCR Detection and Text Box Visualization ---

def detect_text_boxes(image_path):
    """Detect text and bounding boxes using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    image = cv2.imread(image_path)
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise RuntimeError("Failed to encode image for Vision API")
    vision_image = vision.Image(content=buffer.tobytes())
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations
    detections = []
    for text in texts[1:]:  # Skip the first, which is all text
        vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
        detections.append({
            'text': text.description,
            'bbox': vertices
        })
    return image, detections

def filter_valid_ph_detections(detections, image_shape):
    """
    Filter detections to keep only valid pH numbers (1-14) and remove duplicates using IoU clustering.
    
    Args:
        detections: List of detection dicts from Vision API
        image_shape: (height, width, channels) of image
    
    Returns:
        Filtered list of detections
    """
    filtered = []
    
    print(f"Processing {len(detections)} OCR detections...")
    
    for det in detections:
        text = det['text']
        
        # Filter 1: Must be numeric
        if not text.isdigit():
            continue
        
        # Filter 2: Must be valid pH range (1-14)
        try:
            ph_val = int(text)
            if ph_val < 1 or ph_val > 14:
                print(f"  Rejected '{text}' - out of pH range (1-14)")
                continue
        except ValueError:
            continue
        
        filtered.append(det)
    
    print(f"After basic filtering: {len(filtered)} valid pH detections (from {len(detections)} total)")
    
    # Remove spatial duplicates using IoU clustering
    filtered = remove_duplicate_detections(filtered)
    
    return filtered

def remove_duplicate_detections(detections, overlap_threshold=0.3, center_threshold=50):
    """
    Remove duplicate detections of the same text at overlapping positions using IoU clustering
    and center distance checking.
    
    Args:
        detections: List of detection dicts
        overlap_threshold: IoU threshold to consider boxes as duplicates (0.0 - 1.0)
        center_threshold: Max distance between centers to consider duplicates (pixels)
    
    Returns:
        Deduplicated list of detections
    """
    if len(detections) <= 1:
        return detections
    
    def box_area(bbox):
        """Calculate area of bounding box."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width * height
    
    def box_iou(bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes."""
        x1_coords = [p[0] for p in bbox1]
        y1_coords = [p[1] for p in bbox1]
        x2_coords = [p[0] for p in bbox2]
        y2_coords = [p[1] for p in bbox2]
        
        x1_min, x1_max = min(x1_coords), max(x1_coords)
        y1_min, y1_max = min(y1_coords), max(y1_coords)
        x2_min, x2_max = min(x2_coords), max(x2_coords)
        y2_min, y2_max = min(y2_coords), max(y2_coords)
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0  # No overlap
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = box_area(bbox1)
        area2 = box_area(bbox2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def center_distance(bbox1, bbox2):
        """Calculate Euclidean distance between box centers."""
        x1 = np.mean([p[0] for p in bbox1])
        y1 = np.mean([p[1] for p in bbox1])
        x2 = np.mean([p[0] for p in bbox2])
        y2 = np.mean([p[1] for p in bbox2])
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Group detections by text
    text_groups = {}
    for det in detections:
        text = det['text']
        if text not in text_groups:
            text_groups[text] = []
        text_groups[text].append(det)
    
    deduplicated = []
    duplicates_removed = 0
    
    # For each text group, remove overlapping duplicates
    for text, group in text_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue
        
        # Keep first, check others for overlap
        kept = [group[0]]
        
        for det in group[1:]:
            # Check if this detection overlaps or is close to any kept detection
            is_duplicate = False
            for kept_det in kept:
                iou = box_iou(det['bbox'], kept_det['bbox'])
                dist = center_distance(det['bbox'], kept_det['bbox'])
                
                # Consider duplicate if IoU is high OR centers are very close
                if iou > overlap_threshold or dist < center_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    x1 = np.mean([p[0] for p in det['bbox']])
                    x2 = np.mean([p[0] for p in kept_det['bbox']])
                    print(f"  Removed duplicate '{text}' - IoU={iou:.2f}, Dist={dist:.1f}px (x1={x1:.0f}, x2={x2:.0f})")
                    break
            
            if not is_duplicate:
                kept.append(det)
        
        deduplicated.extend(kept)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate detection(s)")
    print(f"Final count: {len(deduplicated)} unique pH detections")
    
    return deduplicated

def draw_text_boxes(image, detections, out_path):
    """Draw bounding boxes and text labels on the image and save it."""
    img = image.copy()
    for det in detections:
        bbox = np.array(det['bbox'], dtype=np.int32)
        cv2.polylines(img, [bbox], isClosed=True, color=(0,255,0), thickness=10)
        x, y = bbox[0]
        cv2.putText(img, det['text'], (x+150, y+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 10)
    cv2.imwrite(out_path, img)
    print(f"Saved labeled text box image to {out_path}")

def cluster_rows(detections, row_gap_thresh=None):
    """Cluster text boxes into rows based on y-coordinate proximity using text height as tolerance."""
    if not detections:
        return []
    
    # Calculate average text height to use as clustering tolerance
    text_heights = []
    for det in detections:
        bbox = det['bbox']
        text_height = max([p[1] for p in bbox]) - min([p[1] for p in bbox])
        text_heights.append(text_height)
        # Also calculate centers for later use
        det['y_center'] = int(np.mean([p[1] for p in bbox]))
        det['x_center'] = int(np.mean([p[0] for p in bbox]))
    
    # Use average text height as tolerance, with fallback
    if row_gap_thresh is None:
        avg_text_height = np.mean(text_heights) if text_heights else 40
        row_gap_thresh = int(avg_text_height)
        print(f"Using text-based row clustering tolerance: {row_gap_thresh} pixels (avg text height: {avg_text_height:.1f})")
    
    # Sort by y_center
    detections_sorted = sorted(detections, key=lambda d: d['y_center'])
    
    # Group into rows using the text height-based tolerance
    rows = []
    for k, group in itertools.groupby(detections_sorted, key=lambda d: d['y_center']//row_gap_thresh):
        rows.append(list(group))
    
    print(f"Clustered {len(detections)} detections into {len(rows)} rows")
    return rows

def define_color_boxes(rows, delta_y_frac=0.4, width_frac=0.5):
    """For each number, define a color box below it."""
    # Calculate text box heights to ensure minimum color box height
    all_detections = [det for row in rows for det in row]
    text_heights = []
    for det in all_detections:
        bbox = det['bbox']
        text_height = max([p[1] for p in bbox]) - min([p[1] for p in bbox])
        text_heights.append(text_height)
    
    if text_heights:
        avg_text_height = np.mean(text_heights)
        min_color_box_height = int(avg_text_height)  # Minimum height = text height
        print(f"Average text height: {avg_text_height:.1f} pixels, minimum color box height: {min_color_box_height}")
    else:
        min_color_box_height = 20  # fallback minimum
        avg_text_height = 20
    
    if len(rows) < 2:
        print("Warning: Less than 2 rows detected. Using text box height to estimate color box size.")
        # Use text height as a reasonable scale reference
        row_delta = avg_text_height * 3  # Color box should be ~3x text height below
        print(f"Estimated row_delta from text height: {row_delta:.1f} pixels")
    else:
        # Use mean y distance between row centers
        row_delta = abs(np.mean([det['y_center'] for det in rows[1]]) - np.mean([det['y_center'] for det in rows[0]]))
        print(f"Calculated row_delta from row spacing: {row_delta:.1f} pixels")
    
    # Add minimum spacing to ensure color boxes don't overlap with text
    min_spacing = avg_text_height * 0.5  # At least half text height spacing
    print(f"Using minimum spacing: {min_spacing:.1f} pixels")
    
    color_boxes = []
    for row in rows:
        # Sort by x
        row_sorted = sorted(row, key=lambda d: d['x_center'])
        # Compute average x distance between numbers
        if len(row_sorted) > 1:
            x_dists = [row_sorted[i+1]['x_center'] - row_sorted[i]['x_center'] for i in range(len(row_sorted)-1)]
            avg_x_dist = np.mean(x_dists)
        else:
            avg_x_dist = 40  # fallback
        for det in row_sorted:
            bbox = det['bbox']
            x_center = det['x_center']
            y_max = max([p[1] for p in bbox])
            
            # Calculate this specific text box height and width
            text_height = max([p[1] for p in bbox]) - min([p[1] for p in bbox])
            text_width = max([p[0] for p in bbox]) - min([p[0] for p in bbox])
            
            # Calculate width ensuring it's at least as wide as the text
            calculated_width = int(avg_x_dist * width_frac)
            width = max(calculated_width, text_width, 100)  # Minimum 100 pixels wide
            
            calculated_height = int(row_delta * delta_y_frac)
            
            # Ensure height is at least the height of this number's bounding box
            height = max(calculated_height, text_height, min_color_box_height)
            
            x1 = int(x_center - width//2)
            x2 = int(x_center + width//2)
            
            # FIXED: More robust positioning - ensure adequate spacing below text
            spacing = max(min_spacing, row_delta * 0.2)  # Use larger of minimum spacing or 30% of row_delta
            y1 = int(y_max + spacing)
            y2 = int(y1 + height)
            
            print(f"pH {det['text']}: text_height={text_height}, spacing={spacing:.1f}, box=({x1},{y1},{x2},{y2})")
            
            color_boxes.append({
                'ph_text': det['text'],
                'rect': (x1, y1, x2, y2),
                'x_center': x_center,
                'y_max': y_max
            })
    return color_boxes

def draw_color_boxes(image, detections, color_boxes, out_path=None):
    img = image.copy()
    for det in detections:
        bbox = np.array(det['bbox'], dtype=np.int32)
        cv2.polylines(img, [bbox], isClosed=True, color=(0,255,0), thickness=10)
        x, y = bbox[0]
        cv2.putText(img, det['text'], (x+150, y+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 10)
    for box in color_boxes:
        x1, y1, x2, y2 = box['rect']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 10)  # White outline
        cv2.putText(img, f"pH {box['ph_text']}", (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 10)  # White label, size 3
    if out_path:
        cv2.imwrite(out_path, img)
        print(f"Saved labeled color box image to {out_path}")
    return img

def split_multi_digit_detection(det, avg_digit_width, width_thresh=1.3):
    """
    Intelligently split multi-digit numeric detections into individual digits.
    
    Uses spatial analysis to distinguish between:
    - Valid multi-digit pH values (10-14) that should stay together
    - Multiple single-digit pH values (456) that were incorrectly grouped
    
    Args:
        det: Detection dict with 'text' and 'bbox'
        avg_digit_width: Average width of single-digit detections
        width_thresh: Threshold multiplier - split if per-digit width exceeds this
    
    Returns:
        List of detection dicts (split or original)
    """
    text = det['text']
    bbox = det['bbox']
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_width = x_max - x_min
    N = len(text)
    
    if not (N > 1 and text.isdigit()):
        return [det]
    
    # Strategy 1: Check if it's a valid multi-digit pH value (10-14)
    try:
        ph_val = int(text)
        if 10 <= ph_val <= 14:
            # Valid two-digit pH - DON'T split
            print(f"  Keeping '{text}' intact (valid pH {ph_val})")
            return [det]
    except ValueError:
        pass
    
    # Strategy 2: Use spatial analysis - check per-digit width
    per_digit_width = box_width / N
    
    # If per-digit width is significantly larger than avg single digit,
    # it means the digits are spaced far apart (likely separate pH values)
    if per_digit_width > avg_digit_width * width_thresh:
        print(f"  Splitting '{text}' (per-digit width {per_digit_width:.1f}px > {avg_digit_width:.1f}px × {width_thresh})")
        new_dets = []
        for i, char in enumerate(text):
            # Distribute characters evenly across the bounding box
            if N == 1:
                center_x = (x_min + x_max) // 2
            else:
                # Linear interpolation between left and right edges
                center_x = int(x_min + i * (box_width) / (N - 1))
            
            # Use either average digit width or proportional width, whichever is smaller
            estimated_char_width = min(avg_digit_width, box_width // N)
            char_x1 = int(center_x - estimated_char_width // 2)
            char_x2 = int(center_x + estimated_char_width // 2)
            
            # Clamp to original box boundaries
            char_x1 = max(x_min, char_x1)
            char_x2 = min(x_max, char_x2)
            
            char_bbox = [
                (char_x1, y_min),
                (char_x2, y_min),
                (char_x2, y_max),
                (char_x1, y_max)
            ]
            new_dets.append({'text': char, 'bbox': char_bbox})
        return new_dets
    else:
        # Digits are close together - likely a single multi-digit number
        # But it's not a valid pH (not 10-14), so it will be filtered out later
        print(f"  Keeping '{text}' intact (compact spacing: {per_digit_width:.1f}px per digit)")
        return [det]

def get_average_colors(image, color_boxes):
    """
    Returns a list of dicts: [{'ph_text': ..., 'avg_color': [B, G, R], 'rect': (x1, y1, x2, y2)}, ...]
    """
    results = []
    for box in color_boxes:
        x1, y1, x2, y2 = box['rect']
        # Clamp to image bounds
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
        roi = image[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            avg_color = [0, 0, 0]
        else:
            avg_color = roi.mean(axis=(0, 1)).tolist()  # BGR order
        results.append({
            'ph_text': box['ph_text'],
            'avg_color': avg_color,
            'rect': (x1, y1, x2, y2)
        })
    return results

def find_closest_color(target_bgr, reference_dict):
    """
    Returns the label and distance of the closest reference color to target_bgr.
    reference_dict: {label: [B, G, R], ...}
    """
    min_dist = float('inf')
    best_label = None
    for label, ref_bgr in reference_dict.items():
        dist = sum((a - b) ** 2 for a, b in zip(target_bgr, ref_bgr)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_label = label
    return best_label, min_dist

def get_average_color_of_box(image, x, y, width, height):
    """
    Returns the average BGR color of the box and the region of interest.
    """
    x1, y1, x2, y2 = int(x), int(y), int(x+width), int(y+height)
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    roi = image[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        avg_color_bgr = [0, 0, 0]
    else:
        avg_color_bgr = [int(round(c)) for c in roi.mean(axis=(0, 1))]
    return avg_color_bgr, (x1c, y1c, x2c, y2c), roi

def highlight_and_label_box(image, box, label, out_path, color_boxes=None, detections=None):
    img = image.copy()
    # Optionally draw all color boxes and/or text boxes
    if color_boxes is not None and detections is not None:
        img = draw_color_boxes(img, detections, color_boxes, out_path=None)
    elif color_boxes is not None:
        img = draw_color_boxes(img, [], color_boxes, out_path=None)
    # Draw the highlighted box and label
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 10
    cv2.putText(img, f"pH {label}", (int(x1), int(y1-20)), font, font_scale, (0, 255, 255), thickness)  # White label, size 3
    if out_path:
        cv2.imwrite(out_path, img)
        print(f"Saved highlighted box with label to {out_path}")
    return img

def convert_bgr_to_color_space(bgr_color, color_space='rgb'):
    """
    Convert BGR color to specified color space.
    Args:
        bgr_color: BGR color as [B, G, R] list/array
        color_space: 'rgb', 'lab', or 'hsv'
    Returns:
        Converted color as numpy array
    """
    bgr_array = np.array(bgr_color, dtype=np.uint8).reshape(1, 1, 3)
    
    if color_space.lower() == 'rgb':
        # Convert BGR to RGB for consistency
        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        return rgb.flatten().astype(np.float32)
    elif color_space.lower() == 'lab':
        # Convert BGR to LAB
        lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)
        return lab.flatten().astype(np.float32)
    elif color_space.lower() == 'hsv':
        # Convert BGR to HSV
        hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        return hsv.flatten().astype(np.float32)
    else:
        raise ValueError(f"Unsupported color space: {color_space}. Use 'rgb', 'lab', or 'hsv'.")

def find_closest_detected_ph(target_bgr, avg_colors, color_space='rgb'):
    """
    Returns the ph_text and distance of the closest detected color box in avg_colors.
    Args:
        target_bgr: Target color in BGR format [B, G, R]
        avg_colors: List of color data from get_average_colors
        color_space: 'rgb', 'lab', or 'hsv' - color space for distance calculation
    Returns:
        (best_ph, min_dist) tuple
    """
    min_dist = float('inf')
    best_ph = None
    
    # Convert target color to specified color space
    target_converted = convert_bgr_to_color_space(target_bgr, color_space)
    
    for entry in avg_colors:
        bgr = [int(round(c)) for c in entry['avg_color']]
        # Convert reference color to specified color space
        ref_converted = convert_bgr_to_color_space(bgr, color_space)
        # Calculate Euclidean distance in the color space
        dist = np.linalg.norm(ref_converted - target_converted)
        if dist < min_dist:
            min_dist = dist
            best_ph = entry['ph_text']
    return best_ph, min_dist

def interpolate_ph_from_distances(distances):
    """
    Interpolate pH value to one decimal place based on two closest references.
    Args:
        distances: Dict of {pH_string: distance}
    Returns:
        Interpolated pH value rounded to 1 decimal place as float, or None if invalid
    """
    if not distances:
        return None

    # Convert pH strings to floats and sort by distance
    try:
        ph_distances = [(float(ph), dist) for ph, dist in distances.items()]
    except ValueError:
        return None

    sorted_phs = sorted(ph_distances, key=lambda x: x[1])
    
    if len(sorted_phs) < 2:
        # Not enough references for interpolation
        return round(sorted_phs[0][0], 1) if sorted_phs else None
    
    # Get two closest pH values
    ph1, dist1 = sorted_phs[0]
    ph2, dist2 = sorted_phs[1]
    
    # If distances are very similar or first distance is zero, return closest
    if dist1 == 0 or dist2 == 0:
        return round(ph1, 1)
    
    # Inverse distance weighting for interpolation
    weight1 = 1.0 / dist1
    weight2 = 1.0 / dist2
    total_weight = weight1 + weight2
    
    interpolated_ph = (ph1 * weight1 + ph2 * weight2) / total_weight
    return round(interpolated_ph, 1)

def ph_from_image(image_path, return_all_color_spaces=False, output_dir=None, interpolate=True):
    """
    Detect pH from color grid image.
    
    Args:
        image_path: Path to the image file to analyze
        return_all_color_spaces: If True, returns dict with results from RGB, LAB, and HSV.
                                 If False, returns single pH value from RGB (default behavior)
        output_dir: Directory to save annotated images. If None, saves to ~/Pictures/pH_photos/
                   If "same", saves to same directory as input image
        interpolate: If True, interpolates pH to 1 decimal place; if False, returns exact match
    
    Returns:
        If return_all_color_spaces=False: pH value as string or float (e.g., "7" or 7.3)
        If return_all_color_spaces=True: dict like {'rgb': 7.3, 'lab': 7.1, 'hsv': 8.0, 
                                                     'distances': {'rgb': 15.2, 'lab': 12.1, 'hsv': 18.5}}
    """
    # Setup output directory
    if output_dir == "same":
        # Save in same directory as input image
        output_dir = Path(image_path).parent
    elif output_dir is None:
        # Default to ~/Pictures/pH_photos/
        home_dir = Path.home()
        output_dir = home_dir / "Pictures" / "pH_photos"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base filename without extension from input image
    input_filename = Path(image_path).stem  # e.g., "capture_20250715-151115_200200200"
    
    image, detections = detect_text_boxes(image_path)
    
    # STEP 1: Split multi-digit detections FIRST (before filtering)
    # Compute average width of single-digit boxes
    single_digit_widths = [
        max([p[0] for p in det['bbox']]) - min([p[0] for p in det['bbox']])
        for det in detections if len(det['text']) == 1 and det['text'].isdigit()
    ]
    avg_digit_width = np.mean(single_digit_widths) if single_digit_widths else 40
    
    # Split multi-digit detections (e.g., "456" -> "4", "5", "6")
    split_detections = []
    for det in detections:
        split_detections.extend(split_multi_digit_detection(det, avg_digit_width))
    detections = split_detections
    
    # STEP 2: Filter out invalid and duplicate detections
    detections = filter_valid_ph_detections(detections, image.shape)
    
    if not detections:
        print("[ERROR] No valid pH numbers detected.")
        return "NULL"

    out_path1 = output_dir / f"{input_filename}_step1_text_boxes.png"
    draw_text_boxes(image, detections, str(out_path1))

    # --- STEP 3: Cluster rows and define color boxes ---
    rows = cluster_rows(detections)
    
    color_boxes = define_color_boxes(rows)
    out_path2 = output_dir / f"{input_filename}_step2_color_boxes.png"
    draw_color_boxes(image, detections, color_boxes, str(out_path2))

    # --- STEP 4: Get average colors ---
    avg_colors = get_average_colors(image, color_boxes)
    
    if not avg_colors:
        print("[ERROR] No color boxes defined.")
        return "NULL"

    # --- STEP 5: Find pH using different color spaces ---
    avg_bgr, box_coords, roi = get_average_color_of_box(image, 750, 1100, 150, 300) # this is the box of pH strip
    
    if return_all_color_spaces:
        # Calculate pH for all three color spaces
        results = {}
        all_distances = {}
        color_spaces = ['rgb', 'lab', 'hsv']
        
        for color_space in color_spaces:
            # Get distances to all pH values in this color space
            distances_dict = {}
            target_converted = convert_bgr_to_color_space(avg_bgr, color_space)
            
            for entry in avg_colors:
                bgr = [int(round(c)) for c in entry['avg_color']]
                ref_converted = convert_bgr_to_color_space(bgr, color_space)
                dist = np.linalg.norm(ref_converted - target_converted)
                distances_dict[entry['ph_text']] = dist
            
            # Interpolate or get closest
            if interpolate:
                ph_value = interpolate_ph_from_distances(distances_dict)
                if ph_value is None:
                    print(f"[WARNING] Could not interpolate pH for {color_space}")
                    results[color_space] = None
                else:
                    results[color_space] = ph_value
            else:
                if distances_dict:
                    closest_ph = min(distances_dict, key=distances_dict.get)
                    results[color_space] = closest_ph
                else:
                    results[color_space] = None
            
            # Store minimum distance for reference
            if distances_dict:
                all_distances[color_space] = min(distances_dict.values())
                print(f"{color_space.upper()} color space: pH={results[color_space]}, min distance={all_distances[color_space]:.2f}")
            else:
                all_distances[color_space] = -1
        
        # Save highlighted image with RGB result (default)
        out_path3 = output_dir / f"{input_filename}_step3_highlighted_box.png"
        highlight_and_label_box(image, box_coords, results.get('rgb'), str(out_path3), 
                               color_boxes=color_boxes, detections=detections)
        
        results['distances'] = all_distances
        return results
    else:
        # Single color space mode - return pH using RGB
        distances_dict = {}
        target_converted = convert_bgr_to_color_space(avg_bgr, 'rgb')
        
        for entry in avg_colors:
            bgr = [int(round(c)) for c in entry['avg_color']]
            ref_converted = convert_bgr_to_color_space(bgr, 'rgb')
            dist = np.linalg.norm(ref_converted - target_converted)
            distances_dict[entry['ph_text']] = dist
        
        # Interpolate or get closest
        ph_result = None
        if interpolate:
            ph_result = interpolate_ph_from_distances(distances_dict)
        else:
            if distances_dict:
                ph_result = min(distances_dict, key=distances_dict.get)
        
        out_path3 = output_dir / f"{input_filename}_step3_highlighted_box.png"
        highlight_and_label_box(image, box_coords, ph_result, str(out_path3), 
                               color_boxes=color_boxes, detections=detections)
        
        return ph_result if ph_result is not None else "NULL"


def main():
    parser = argparse.ArgumentParser(description="Detect pH from color grid image.")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    args = parser.parse_args()
    result = ph_from_image(args.image_path)
    print(result if result is not None else "NULL")

if __name__ == "__main__":
    main() 