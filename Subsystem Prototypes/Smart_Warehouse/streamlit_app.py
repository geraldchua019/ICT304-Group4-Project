import streamlit as st
import cv2
import os
import tempfile
from datetime import datetime
import re
import numpy as np
import pandas as pd
from io import BytesIO
import base64
import uuid
from PIL import Image as PILImage, ImageDraw, ImageFont

# Configure page
st.set_page_config(
    page_title="AI Inventory Detection System",
    page_icon="📦",
    layout="wide"
)

# ─────────────────────────────────────────────
# CLIP BRAND DETECTION
# ─────────────────────────────────────────────

BRAND_LABELS = [
    "a Dell laptop",
    "an HP laptop",
    "an ASUS laptop",
    "an Acer laptop",
    "a Lenovo laptop",
    "an Apple MacBook",
    "a Samsung laptop or phone",
    "an unknown laptop with no visible brand"
]

BRAND_LABEL_MAP = {
    "a Dell laptop": "Dell",
    "an HP laptop": "HP",
    "an ASUS laptop": "ASUS",
    "an Acer laptop": "Acer",
    "a Lenovo laptop": "Lenovo",
    "an Apple MacBook": "Apple",
    "a Samsung laptop or phone": "Samsung",
    "an unknown laptop with no visible brand": "Unknown"
}

PHONE_LABELS = [
    "a Samsung phone",
    "an Apple iPhone",
    "a Google Pixel phone",
    "a Huawei phone",
    "an unknown phone with no visible brand"
]

PHONE_LABEL_MAP = {
    "a Samsung phone": "Samsung",
    "an Apple iPhone": "Apple",
    "a Google Pixel phone": "Google",
    "a Huawei phone": "Huawei",
    "an unknown phone with no visible brand": "Unknown"
}


def classify_brand_clip(clip_model, clip_processor, cropped_image, item_type="laptop"):
    """Use CLIP to classify brand from a cropped image"""
    try:
        import torch
        from PIL import Image as PILImage

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        # Pick labels based on item type
        if item_type == "cell phone":
            labels = PHONE_LABELS
            label_map = PHONE_LABEL_MAP
        else:
            labels = BRAND_LABELS
            label_map = BRAND_LABEL_MAP

        inputs = clip_processor(
            text=labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = clip_model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]
        results = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)

        top_label, top_conf = results[0]
        brand_name = label_map[top_label]

        # Don't return Unknown as a brand
        if brand_name == "Unknown" or top_conf < 0.4:
            return []

        return [{
            "text": brand_name,
            "confidence": top_conf,
            "is_brand": True,
            "detection_method": "clip"
        }]

    except Exception as e:
        return []


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_pdf_delivery_order(pdf_file):
    """Extract text from PDF Delivery Order"""
    try:
        import pdfplumber
    except ImportError:
        st.error("❌ Need pdfplumber: pip install pdfplumber")
        return None

    try:
        do_data = {
            "items": {},
            "metadata": {
                "supplier": "N/A",
                "do_number": "N/A",
                "delivery_date": "N/A"
            }
        }

        item_mappings = {
            'laptop': ['laptop', 'notebook', 'computer', 'latitude'],
            'bottle': ['bottle', 'water bottle', 'drink'],
            'cup': ['cup', 'mug'],
            'cell phone': ['phone', 'cell phone', 'mobile', 'smartphone', 'cellphone', 'iphone', 'samsung'],
            'book': ['book'],
            'box': ['box', 'carton', 'package'],
            'keyboard': ['keyboard'],
            'mouse': ['mouse'],
            'backpack': ['backpack', 'bag', 'rucksack'],
            'chair': ['chair', 'seat'],
        }

        all_text = []

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)

                tables = page.extract_tables()

                if tables:
                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        header_row = None
                        qty_col_idx = None
                        desc_col_idx = None

                        for row_idx, row in enumerate(table[:3]):
                            if not row:
                                continue
                            row_lower = [str(cell).lower() if cell else "" for cell in row]
                            for col_idx, cell in enumerate(row_lower):
                                if 'quantity' in cell or 'qty' in cell:
                                    qty_col_idx = col_idx
                                    header_row = row_idx
                                if 'item' in cell or 'description' in cell:
                                    desc_col_idx = col_idx

                        if qty_col_idx is None:
                            continue
                        if desc_col_idx is None:
                            desc_col_idx = 1 if len(table[0]) > 1 else 0

                        for row_idx in range(header_row + 1 if header_row else 1, len(table)):
                            row = table[row_idx]
                            if not row or len(row) <= max(qty_col_idx, desc_col_idx):
                                continue

                            qty_cell = str(row[qty_col_idx]) if row[qty_col_idx] else ""
                            qty_cell = qty_cell.strip()
                            desc_cell = str(row[desc_col_idx]) if row[desc_col_idx] else ""
                            desc_cell = desc_cell.lower().strip()

                            if not qty_cell or not desc_cell:
                                continue

                            qty_match = re.search(r'\b(\d+)\b', qty_cell)
                            if not qty_match:
                                continue

                            quantity = int(qty_match.group(1))
                            if quantity < 1 or quantity > 1000:
                                continue

                            matched_item = None
                            for standard_name, variations in item_mappings.items():
                                for variation in variations:
                                    if variation in desc_cell:
                                        matched_item = standard_name
                                        break
                                if matched_item:
                                    break

                            if matched_item:
                                if matched_item in do_data["items"]:
                                    do_data["items"][matched_item] += quantity
                                else:
                                    do_data["items"][matched_item] = quantity

        combined_text = "\n".join(all_text)

        do_match = re.search(r'do[-\s]?(\d{4}[-]?\d+)', combined_text.lower())
        if do_match:
            do_data["metadata"]["do_number"] = do_match.group(0).upper()

        lines = combined_text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'supplier' in line_lower or 'company name' in line_lower:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        supplier = parts[1].strip()
                        if supplier and len(supplier) > 3:
                            do_data["metadata"]["supplier"] = supplier
                elif i + 1 < len(lines):
                    supplier = lines[i + 1].strip()
                    if supplier and len(supplier) > 3 and not supplier[0].isdigit():
                        do_data["metadata"]["supplier"] = supplier
                break

        if do_data["items"]:
            return do_data
        else:
            return None

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x2_min, y2_min, x2_max, y2_max = box2['x1'], box2['y1'], box2['x2'], box2['y2']

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def filter_overlapping_detections(boxes_with_info, iou_threshold=0.5):
    label_groups = {}
    for box in boxes_with_info:
        label = box['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(box)

    filtered_boxes = []
    for label, boxes in label_groups.items():
        boxes.sort(key=lambda x: x['conf'], reverse=True)
        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            boxes = [box for box in boxes if calculate_iou(current, box) < iou_threshold]
        filtered_boxes.extend(keep)

    return filtered_boxes


@st.cache_resource
def load_models():
    """Load YOLO and CLIP models"""
    try:
        from ultralytics import YOLO
        from transformers import CLIPProcessor, CLIPModel

        with st.spinner("Loading AI models... (first time may take a minute)"):
            # Object detection model
            yolo_model = YOLO("yolov8n.pt")

            # CLIP brand classification model
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()

        return yolo_model, clip_model, clip_processor

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


def count_items_in_photo(yolo_model, clip_model, clip_processor, image, exclude_labels=None):
    """Detect items and classify brands using CLIP"""
    if exclude_labels is None:
        exclude_labels = []

    results = yolo_model(image, verbose=False, imgsz=1280, conf=0.3)
    result = results[0]

    clip_items = ["laptop", "cell phone"]
    text_items = ["bottle", "book", "box"]

    boxes_with_info = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = yolo_model.names[cls]

        if label in exclude_labels:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        area = width * height
        img_area = image.shape[0] * image.shape[1]

        if label == "laptop" and area < img_area * 0.01:
            continue

        boxes_with_info.append({
            'label': label,
            'conf': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'area': area
        })

    boxes_with_info = filter_overlapping_detections(boxes_with_info, iou_threshold=0.4)
    count = len(boxes_with_info)

    annotated = image.copy()
    boxes_with_info.sort(key=lambda b: (b['center_y'], b['center_x']))

    detections = []
    item_counter = {}

    for idx, box_info in enumerate(boxes_with_info, 1):
        label = box_info['label']
        conf = box_info['conf']
        x1, y1, x2, y2 = box_info['x1'], box_info['y1'], box_info['x2'], box_info['y2']

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(annotated, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if label not in item_counter:
            item_counter[label] = 0
        item_counter[label] += 1
        item_num = item_counter[label]

        detection = {
            "label": label,
            "confidence": conf,
            "brands": [],
            "position": idx,
            "bbox": (x1, y1, x2, y2)
        }

        # Crop item for brand detection
        if label in clip_items or label in text_items:
            pad = 40
            h, w = image.shape[:2]
            x1_crop = max(0, x1 - pad)
            y1_crop = max(0, y1 - pad)
            x2_crop = min(w, x2 + pad)
            y2_crop = min(h, y2 + pad)

            cropped_item = image[y1_crop:y2_crop, x1_crop:x2_crop]

            if cropped_item.shape[0] >= 20 and cropped_item.shape[1] >= 20:
                if label in clip_items:
                    # Use CLIP for laptops and phones
                    brands = classify_brand_clip(clip_model, clip_processor, cropped_item, label)
                else:
                    # Use EasyOCR for bottles, books, boxes
                    try:
                        import easyocr
                        reader = easyocr.Reader(['en'], gpu=False)
                        ocr_results = reader.readtext(cropped_item, paragraph=False)
                        brands = []
                        for (bbox, text, ocr_conf) in ocr_results:
                            if ocr_conf > 0.3 and len(text.strip()) >= 2:
                                brands.append({
                                    "text": text.strip(),
                                    "confidence": ocr_conf,
                                    "is_brand": False,
                                    "detection_method": "ocr"
                                })
                    except:
                        brands = []

                detection["brands"] = brands

                if brands:
                    brand_text = " | ".join([b["text"] for b in brands[:2]])
                    full_text = f"#{item_num} {brand_text}"
                    text_color = (255, 0, 0) if label == "laptop" else (0, 0, 255) if label == "bottle" else (0, 255, 255)
                    cv2.putText(annotated, full_text, (x1, max(y1 - 10, 0)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        detections.append(detection)

    cv2.putText(annotated, f"TOTAL: {count} items", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return count, annotated, detections


def compare_quantities(actual_counts, do_data):
    if not do_data:
        return None

    comparison = {
        "matches": [],
        "discrepancies": [],
        "has_discrepancy": False
    }

    for item_type, expected_qty in do_data["items"].items():
        actual_qty = actual_counts.get(item_type, 0)
        difference = actual_qty - expected_qty

        status = {
            "item_type": item_type,
            "expected": expected_qty,
            "actual": actual_qty,
            "difference": difference,
            "status": "✅ MATCH" if difference == 0 else "⚠️ DISCREPANCY"
        }

        if difference != 0:
            comparison["discrepancies"].append(status)
            comparison["has_discrepancy"] = True
        else:
            comparison["matches"].append(status)

    for item_type, actual_qty in actual_counts.items():
        if item_type not in do_data["items"]:
            comparison["discrepancies"].append({
                "item_type": item_type,
                "expected": 0,
                "actual": actual_qty,
                "difference": actual_qty,
                "status": "⚠️ UNEXPECTED ITEM"
            })
            comparison["has_discrepancy"] = True

    return comparison


def create_excel_report(results_log, do_data=None, comparison=None):
    all_detections = []
    clip_items = ["laptop", "cell phone"]

    for result in results_log:
        image_name = result['file']
        for i, detection in enumerate(result['detections'], 1):
            item_type = detection['label']
            confidence = detection['confidence']

            if detection['brands']:
                for brand_info in detection['brands']:
                    detection_method = brand_info.get('detection_method', 'ocr')
                    all_detections.append({
                        'Image': image_name,
                        'Item Number': i,
                        'Item Type': item_type,
                        'Detection Confidence': f"{confidence:.1%}",
                        'Brand/Text': brand_info['text'],
                        'Brand Confidence': f"{brand_info['confidence']:.1%}",
                        'Detection Method': '🤖 CLIP' if detection_method == 'clip' else '📝 OCR'
                    })
            elif item_type in clip_items:
                all_detections.append({
                    'Image': image_name,
                    'Item Number': i,
                    'Item Type': item_type,
                    'Detection Confidence': f"{confidence:.1%}",
                    'Brand/Text': 'N/A',
                    'Brand Confidence': '',
                    'Detection Method': ''
                })

    item_counts = {}
    item_brands = {}

    for result in results_log:
        for detection in result['detections']:
            item_type = detection['label']
            item_counts[item_type] = item_counts.get(item_type, 0) + 1

            if detection['brands']:
                if item_type not in item_brands:
                    item_brands[item_type] = {}
                for brand_info in detection['brands']:
                    brand = brand_info['text']
                    if brand not in item_brands[item_type]:
                        item_brands[item_type][brand] = 0
                    item_brands[item_type][brand] += 1

    summary_data = []
    for item_type, count in sorted(item_counts.items()):
        summary_data.append({
            'Item Type': item_type.capitalize(),
            'Total Count': count
        })

    brand_summary = []
    for item_type, brands in sorted(item_brands.items()):
        total_items = item_counts.get(item_type, 0)
        for brand, count in sorted(brands.items()):
            percentage = (count / total_items * 100) if total_items > 0 else 0
            brand_summary.append({
                'Item Type': item_type.capitalize(),
                'Brand': brand,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if comparison and do_data:
            comparison_data = []
            comparison_data.append({'Item Type': 'PDF DO INFORMATION', 'Expected Quantity': '', 'Actual Quantity': '', 'Difference': '', 'Status': ''})
            comparison_data.append({'Item Type': f"DO Number: {do_data['metadata'].get('do_number', 'N/A')}", 'Expected Quantity': '', 'Actual Quantity': '', 'Difference': '', 'Status': ''})
            comparison_data.append({'Item Type': f"Supplier: {do_data['metadata'].get('supplier', 'N/A')}", 'Expected Quantity': '', 'Actual Quantity': '', 'Difference': '', 'Status': ''})
            comparison_data.append({'Item Type': '', 'Expected Quantity': '', 'Actual Quantity': '', 'Difference': '', 'Status': ''})

            for match in comparison["matches"]:
                comparison_data.append({'Item Type': match["item_type"].upper(), 'Expected Quantity': match["expected"], 'Actual Quantity': match["actual"], 'Difference': match["difference"], 'Status': match["status"]})
            for disc in comparison["discrepancies"]:
                comparison_data.append({'Item Type': disc["item_type"].upper(), 'Expected Quantity': disc["expected"], 'Actual Quantity': disc["actual"], 'Difference': f"{disc['difference']:+d}", 'Status': disc["status"]})

            pd.DataFrame(comparison_data).to_excel(writer, sheet_name='PDF DO Comparison', index=False)

        if all_detections:
            pd.DataFrame(all_detections).to_excel(writer, sheet_name='All Detections', index=False)
        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Item Summary', index=False)
        if brand_summary:
            pd.DataFrame(brand_summary).to_excel(writer, sheet_name='Brand Summary', index=False)

    output.seek(0)
    return output


# ─────────────────────────────────────────────
# ASSET STICKER GENERATION
# ─────────────────────────────────────────────

def generate_asset_id():
    """Generate a unique asset ID like AST-2024-XXXX"""
    short = str(uuid.uuid4()).upper()[:8]
    year = datetime.now().strftime("%Y")
    return f"AST-{year}-{short}"


def generate_sticker_png(asset_id, image_name, detection, item_num, date_str, cropped_img=None):
    """Generate a PNG asset sticker with barcode for a single detected item"""
    try:
        import barcode
        from barcode.writer import ImageWriter
    except ImportError:
        return None

    # ── Sticker dimensions ──
    W, H = 600, 320
    bg_color = (255, 255, 255)
    primary = (30, 30, 80)
    accent = (0, 180, 120)
    light_gray = (240, 240, 245)

    sticker = PILImage.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(sticker)

    # Header bar
    draw.rectangle([(0, 0), (W, 60)], fill=primary)

    # Try to load a font, fall back to default
    try:
        font_large  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        font_small  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_bold   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font_large = font_medium = font_small = font_bold = ImageFont.load_default()

    # Header text
    draw.text((20, 10), "📦 ASSET STICKER", font=font_large, fill=(255, 255, 255))
    draw.text((W - 160, 20), date_str, font=font_small, fill=(200, 200, 220))

    # Asset ID box
    draw.rectangle([(15, 75), (W - 15, 115)], fill=light_gray, outline=primary, width=2)
    draw.text((25, 82), "ASSET ID:", font=font_bold, fill=primary)
    draw.text((120, 82), asset_id, font=font_bold, fill=accent)

    # Item info
    item_label = detection['label'].title()
    brand_name = detection['brands'][0]['text'] if detection['brands'] else "Unknown"
    conf = detection['confidence']

    draw.text((25, 125), f"Source: {image_name}  •  Item #{item_num}", font=font_small, fill=(100, 100, 100))
    draw.text((25, 148), "Item Details:", font=font_bold, fill=primary)

    y = 170
    draw.text((25, y),      f"  • Type:       {item_label}", font=font_medium, fill=(50, 50, 50))
    draw.text((25, y + 24), f"  • Brand:      {brand_name}", font=font_medium, fill=(50, 50, 50))
    draw.text((25, y + 48), f"  • Confidence: {conf:.0%}", font=font_medium, fill=(50, 50, 50))

    # Paste cropped image thumbnail on the right if available
    if cropped_img is not None:
        try:
            thumb_size = (120, 90)
            rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            thumb = PILImage.fromarray(rgb).resize(thumb_size, PILImage.LANCZOS)
            sticker.paste(thumb, (W - 145, 70))
            draw.rectangle([(W - 146, 69), (W - 24, 161)], outline=primary, width=2)
        except:
            pass

    # ── Barcode ──
    try:
        barcode_io = BytesIO()
        code128 = barcode.get('code128', asset_id, writer=ImageWriter())
        code128.write(barcode_io, options={
            "module_width": 0.8,
            "module_height": 8.0,
            "font_size": 6,
            "text_distance": 2,
            "quiet_zone": 2,
            "write_text": True,
            "background": "white",
            "foreground": "black"
        })
        barcode_io.seek(0)
        barcode_img = PILImage.open(barcode_io).convert("RGB")

        # Resize barcode to fit right side of sticker
        bc_w, bc_h = 200, 80
        barcode_img = barcode_img.resize((bc_w, bc_h), PILImage.LANCZOS)
        sticker.paste(barcode_img, (W - bc_w - 15, H - bc_h - 15))
    except Exception:
        # If barcode fails, just write the asset ID as text
        draw.text((W - 210, H - 40), asset_id, font=font_small, fill=primary)

    # Bottom accent line
    draw.rectangle([(0, H - 6), (W, H)], fill=accent)

    # Convert to PNG bytes
    output = BytesIO()
    sticker.save(output, format="PNG", dpi=(300, 300))
    output.seek(0)
    return output


# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────

def main():
    st.title("📦 AI Inventory Detection System")
    st.markdown("### Upload images and optional PDF delivery order for automated inventory verification")

    # Load models
    yolo_model, clip_model, clip_processor = load_models()

    if yolo_model is None:
        st.error("Failed to load AI models. Please check installation.")
        return

    with st.sidebar:
        st.header("📁 Upload Files")

        st.subheader("1. PDF Delivery Order (Optional)")
        pdf_file = st.file_uploader("Upload PDF DO", type=['pdf'])

        st.subheader("2. Images (Required)")
        image_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        st.markdown("---")

        st.subheader("⚙️ Detection Settings")
        all_possible_items = ['chair', 'couch', 'dining table', 'tv', 'potted plant', 'clock',
                             'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'bench']
        exclude_items = st.multiselect(
            "Ignore these items:",
            options=all_possible_items,
            default=['chair'],
            help="Select items to exclude from detection and counting"
        )

        st.markdown("---")
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)

    if process_button:
        if not image_files:
            st.error("Please upload at least one image!")
            return

        # Process PDF
        do_data = None
        if pdf_file:
            with st.spinner("📄 Reading PDF Delivery Order..."):
                do_data = extract_pdf_delivery_order(pdf_file)

            if do_data:
                st.success("✅ PDF DO extracted successfully!")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**DO Number:** {do_data['metadata']['do_number']}")
                with col2:
                    st.info(f"**Supplier:** {do_data['metadata']['supplier']}")

                st.subheader("Expected Items from PDF DO:")
                for item_type, qty in do_data['items'].items():
                    st.write(f"- **{item_type.title()}**: {qty} units")
            else:
                st.warning("⚠️ Could not extract data from PDF. Proceeding in count-only mode.")

        st.markdown("---")

        results_log = []
        actual_counts = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, image_file in enumerate(image_files):
            status_text.text(f"Processing {image_file.name}... ({idx + 1}/{len(image_files)})")

            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            count, annotated, detections = count_items_in_photo(
                yolo_model, clip_model, clip_processor, image,
                exclude_labels=exclude_items
            )

            for detection in detections:
                item_type = detection['label']
                actual_counts[item_type] = actual_counts.get(item_type, 0) + 1

            results_log.append({
                "file": image_file.name,
                "count": count,
                "detections": detections,
                "annotated": annotated
            })

            progress_bar.progress((idx + 1) / len(image_files))

        status_text.text("✅ Processing complete!")
        st.markdown("---")

        st.header("📊 Detection Results")

        for result in results_log:
            with st.expander(f"📸 {result['file']} - {result['count']} items detected", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    annotated_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)

                with col2:
                    st.subheader("Detected Items:")
                    for i, detection in enumerate(result['detections'], 1):
                        st.write(f"**{i}. {detection['label'].title()}** ({detection['confidence']:.0%})")
                        if detection['brands']:
                            for brand in detection['brands'][:3]:
                                method = brand.get('detection_method', 'ocr')
                                icon = "🤖" if method == "clip" else "📝"
                                st.write(f"   {icon} {brand['text']} ({brand['confidence']:.0%})")

        st.markdown("---")

        # ── Asset Stickers ──
        st.header("🏷️ Asset Stickers")
        st.markdown("One sticker per detected item — download and print to label each asset.")

        date_str = datetime.now().strftime("%d %b %Y")

        for result in results_log:
            if not result['detections']:
                continue

            st.subheader(f"📸 {result['file']}")

            # Re-decode original image for cropping
            cols = st.columns(min(len(result['detections']), 3))

            for i, detection in enumerate(result['detections']):
                asset_id = generate_asset_id()

                # Crop the item from the annotated image
                x1, y1, x2, y2 = detection['bbox']
                pad = 20
                h, w = result['annotated'].shape[:2]
                x1c = max(0, x1 - pad)
                y1c = max(0, y1 - pad)
                x2c = min(w, x2 + pad)
                y2c = min(h, y2 + pad)
                cropped = result['annotated'][y1c:y2c, x1c:x2c]

                sticker_png = generate_sticker_png(
                    asset_id=asset_id,
                    image_name=result['file'],
                    detection=detection,
                    item_num=i + 1,
                    date_str=date_str,
                    cropped_img=cropped if cropped.size > 0 else None
                )

                col = cols[i % len(cols)]
                with col:
                    label = detection['label'].title()
                    brand = detection['brands'][0]['text'] if detection['brands'] else "Unknown"
                    st.caption(f"Item {i+1}: {label} — {brand}")

                    if sticker_png:
                        sticker_preview = PILImage.open(BytesIO(sticker_png.getvalue()))
                        st.image(sticker_preview, use_container_width=True)
                        st.download_button(
                            label=f"⬇️ Download",
                            data=sticker_png,
                            file_name=f"sticker_{asset_id}.png",
                            mime="image/png",
                            key=f"sticker_{result['file']}_{i}",
                            use_container_width=True
                        )
                    else:
                        st.warning("Install `python-barcode` to enable stickers")

        st.markdown("---")

        comparison = None
        if do_data:
            st.header("📋 PDF DO Comparison")
            comparison = compare_quantities(actual_counts, do_data)

            if comparison:
                if comparison["matches"]:
                    st.success("✅ **Matching Items:**")
                    for match in comparison["matches"]:
                        st.write(f"- {match['item_type'].title()}: {match['actual']}/{match['expected']} ✓")

                if comparison["discrepancies"]:
                    st.error("⚠️ **Discrepancies Found:**")
                    for disc in comparison["discrepancies"]:
                        st.write(f"- {disc['item_type'].title()}: Expected {disc['expected']}, Got {disc['actual']} (Diff: {disc['difference']:+d})")
                else:
                    st.success("🎉 **ALL ITEMS MATCH!**")

        st.markdown("---")

        st.header("📥 Download Report")
        excel_data = create_excel_report(results_log, do_data, comparison)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"detection_results_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


if __name__ == "__main__":
    main()