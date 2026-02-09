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

# Configure page
st.set_page_config(
    page_title="AI Inventory Detection System",
    page_icon="📦",
    layout="wide"
)

# ─────────────────────────────────────────────
# IMPROVED HELPER FUNCTIONS
# ─────────────────────────────────────────────

def is_gibberish(text):
    """Check if OCR text is likely gibberish - IMPROVED"""
    text_clean = text.strip()
    if len(text_clean) < 2:
        return True
    
    # Check if it's a known brand pattern first
    if correct_ocr_errors(text_clean)[0]:
        return False
    
    text_lower = text_clean.lower().replace(' ', '').replace('-', '')
    
    # Too short
    if len(text_lower) < 2:
        return True
    
    # Count letters vs other chars
    letter_count = sum(c.isalpha() for c in text_lower)
    if letter_count < len(text_lower) * 0.5:
        return True
    
    # Check vowel ratio
    vowels = 'aeiou'
    vowel_count = sum(1 for c in text_lower if c in vowels)
    
    # No vowels in words longer than 3 chars (except common abbreviations)
    if len(text_lower) > 3 and vowel_count == 0:
        known_no_vowels = ['hp', 'msi', 'lcd', 'rgb']
        if text_lower not in known_no_vowels:
            return True
    
    # Too many consonants in a row
    consonants_in_row = 0
    for c in text_lower:
        if c.isalpha():
            if c not in vowels:
                consonants_in_row += 1
                if consonants_in_row > 4:
                    return True
            else:
                consonants_in_row = 0
    
    return False


def correct_ocr_errors(text):
    """Map common OCR misreads to likely brands - TIGHTENED"""
    text_clean = text.lower().replace(' ', '').replace('-', '').replace('_', '')
    
    # HP - very strict
    if text_clean in ['hp', 'hd']:
        return ('HP', 'HP')
    
    # ASUS - more conservative matching
    # Only match very specific misreads, not partial matches
    asus_exact = {
        'fsus': 'ASUS',
        'a5u5': 'ASUS', 
        'a5us': 'ASUS',
        'asu5': 'ASUS',
    }
    if text_clean in asus_exact:
        return ('ASUS', 'ASUS')
    
    # Dell
    dell_patterns = ['vell', 'deil', 'deii', 'de11']
    if text_clean in dell_patterns:
        return ('Dell', 'Dell')
    
    # Acer
    if text_clean in ['aser', 'acar', 'aeer']:
        return ('Acer', 'Acer')
    
    return (None, None)


def preprocess_image_smart(image):
    """Single best preprocessing method instead of trying 5 different ones"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Denoise + CLAHE is most effective for laptop logos
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced


def run_ocr_focused(reader, image):
    """FASTER: Only process top region with best method"""
    h, w = image.shape[:2]
    
    # Focus on top 40% where laptop logos typically are
    top_region = image[:int(h * 0.4), :]
    
    # Use single best preprocessing
    processed = preprocess_image_smart(top_region)
    
    all_texts = []
    
    # Primary OCR pass
    try:
        results = reader.readtext(processed, paragraph=False)
        all_texts.extend([(text, conf, 'top') for _, text, conf in results])
    except:
        pass
    
    # Letter-only pass (helps with logos)
    try:
        results = reader.readtext(
            processed,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        )
        all_texts.extend([(text, conf, 'top_letters') for _, text, conf in results])
    except:
        pass
    
    return all_texts


def detect_laptop_logo_improved(image):
    """IMPROVED: More conservative visual detection"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detections = {}
    
    # Apple detection - MORE STRICT
    apple_score = 0
    avg_color = np.mean(image, axis=(0, 1))
    b, g, r = avg_color
    color_diff = max(abs(r-g), abs(g-b), abs(r-b))
    is_grayscale = color_diff < 15  # Stricter threshold
    
    if is_grayscale:
        avg_brightness = np.mean(avg_color)
        # Only strongly silver/gray laptops
        if 140 < avg_brightness < 200:
            apple_score += 30
        elif 50 < avg_brightness < 80:
            apple_score += 25
    
    # Silver color detection
    lower_silver = np.array([0, 0, 150])
    upper_silver = np.array([180, 30, 220])
    silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
    silver_ratio = np.count_nonzero(silver_mask) / silver_mask.size
    
    if silver_ratio > 0.35:  # Raised threshold
        apple_score += int(min(silver_ratio * 60, 30))
    
    detections['Apple'] = apple_score
    
    # HP detection
    hp_blue_lower = np.array([100, 120, 60])  # More specific blue
    hp_blue_upper = np.array([130, 255, 255])
    hp_blue_mask = cv2.inRange(hsv, hp_blue_lower, hp_blue_upper)
    hp_blue_ratio = np.count_nonzero(hp_blue_mask) / hp_blue_mask.size
    detections['HP'] = int(min(hp_blue_ratio * 250, 60)) if hp_blue_ratio > 0.03 else 0
    
    # Dell detection
    dell_blue_lower = np.array([95, 100, 50])
    dell_blue_upper = np.array([115, 255, 255])
    dell_blue_mask = cv2.inRange(hsv, dell_blue_lower, dell_blue_upper)
    dell_blue_ratio = np.count_nonzero(dell_blue_mask) / dell_blue_mask.size
    detections['Dell'] = int(min(dell_blue_ratio * 250, 60)) if dell_blue_ratio > 0.03 else 0
    
    # ASUS detection
    asus_gold_lower = np.array([15, 120, 120])
    asus_gold_upper = np.array([35, 255, 255])
    asus_gold_mask = cv2.inRange(hsv, asus_gold_lower, asus_gold_upper)
    asus_gold_ratio = np.count_nonzero(asus_gold_mask) / asus_gold_mask.size
    detections['ASUS'] = int(min(asus_gold_ratio * 350, 60)) if asus_gold_ratio > 0.015 else 0
    
    if detections:
        best_brand = max(detections.items(), key=lambda x: x[1])
        brand_name, score = best_brand
        
        # Increased threshold - only return if very confident
        if score >= 45:  # Was 35
            return {
                'brand': brand_name,
                'confidence': min(score / 100.0, 0.85),  # Cap at 85% for visual
                'score': score,
                'method': 'Visual (logo detection)'
            }
    
    return None


def read_text_from_item_isolated(reader, cropped, item_type, item_num):
    """IMPROVED: Faster and more accurate brand detection"""
    brands = []
    
    known_brands = {
        'Dell': ['dell', 'latitude', 'inspiron', 'xps', 'precision', 'alienware'],
        'HP': ['hp', 'hewlett', 'packard', 'pavilion', 'elitebook', 'probook', 'envy', 'omen'],
        'ASUS': ['asus', 'rog', 'zenbook', 'vivobook', 'tuf'],
        'Acer': ['acer', 'aspire', 'predator', 'nitro', 'swift'],
        'Lenovo': ['lenovo', 'thinkpad', 'ideapad', 'yoga', 'legion'],
        'Apple': ['apple', 'macbook', 'mac'],
        'MSI': ['msi', 'gaming'],
        'Samsung': ['samsung', 'galaxy'],
        'Razer': ['razer', 'blade'],
        'Microsoft': ['microsoft', 'surface'],
    }
    
    if item_type == "laptop":
        # Upscale for better OCR
        upscaled = cv2.resize(cropped, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Visual detection first
        detected_logo = detect_laptop_logo_improved(upscaled)
        
        # FASTER: Only run focused OCR instead of 15 passes
        all_texts = run_ocr_focused(reader, upscaled)
        
        found_brands = {}
        seen_texts = set()
        
        # Process OCR results
        for text, ocr_conf, region in all_texts:
            text_clean = text.strip()
            text_lower = text_clean.lower()
            
            if len(text_clean) < 2 or text_lower in seen_texts:
                continue
            
            # Check OCR corrections
            corrected_text, ocr_corrected_brand = correct_ocr_errors(text_clean)
            if ocr_corrected_brand:
                brand_key = ocr_corrected_brand
                if brand_key not in found_brands or ocr_conf > found_brands[brand_key]['conf']:
                    found_brands[brand_key] = {
                        'brand': ocr_corrected_brand,
                        'conf': ocr_conf * 100,
                        'text': text_clean,
                        'corrected': True,
                        'source': 'ocr'
                    }
                seen_texts.add(text_lower)
                continue
            
            # Skip gibberish
            if is_gibberish(text_clean):
                continue
            
            # Must be mostly letters
            letter_count = sum(c.isalpha() for c in text_clean)
            if letter_count < len(text_clean) * 0.6:
                continue
            
            # Match against known brands
            matched_brand = None
            for brand_name, variations in known_brands.items():
                for variant in variations:
                    if variant in text_lower:
                        matched_brand = brand_name
                        break
                if matched_brand:
                    break
            
            if matched_brand:
                brand_key = matched_brand
                if brand_key not in found_brands or ocr_conf > found_brands[brand_key]['conf']:
                    found_brands[brand_key] = {
                        'brand': matched_brand,
                        'conf': ocr_conf * 100,
                        'text': text_clean,
                        'corrected': False,
                        'source': 'ocr'
                    }
                seen_texts.add(text_lower)
        
        # Add visual detection if present
        if detected_logo:
            brand = detected_logo['brand']
            visual_conf = detected_logo['confidence'] * 100
            
            # Only add visual if no OCR found, OR visual is very strong
            if brand not in found_brands:
                found_brands[brand] = {
                    'brand': brand,
                    'conf': visual_conf,
                    'text': '(visual)',
                    'corrected': False,
                    'source': 'visual'
                }
            elif detected_logo['score'] > 60 and found_brands[brand]['conf'] < 70:
                # Visual is very confident, boost it
                found_brands[brand] = {
                    'brand': brand,
                    'conf': max(visual_conf, found_brands[brand]['conf']),
                    'text': found_brands[brand]['text'],
                    'corrected': False,
                    'source': 'visual+ocr'
                }
        
        # COMPREHENSIVE CONFLICT RESOLUTION
        # When multiple brands detected, keep only the most reliable one
        
        if len(found_brands) > 1:
            # Rule 1: Strong OCR text beats visual Apple detection
            if 'Apple' in found_brands and found_brands['Apple']['source'] in ['visual', 'visual+ocr']:
                other_ocr_brands = [b for b in found_brands.keys() 
                                   if b != 'Apple' and found_brands[b]['source'] == 'ocr' 
                                   and found_brands[b]['conf'] > 60]
                if other_ocr_brands:
                    del found_brands['Apple']
            
            # Rule 2: Resolve conflicts between non-Apple brands
            # Priority: Non-corrected OCR > Corrected OCR > Visual
            # When both are similar quality, use brand-specific rules
            
            brands_to_resolve = [b for b in found_brands.keys() if b != 'Apple' or len(found_brands) == 1]
            
            if len(brands_to_resolve) > 1:
                # Build confidence scores for each brand
                brand_scores = {}
                for brand in brands_to_resolve:
                    info = found_brands[brand]
                    score = info['conf']
                    
                    # Boost score if not corrected (real OCR reading)
                    if not info.get('corrected', False):
                        score *= 1.3
                    
                    # Penalty for very common OCR errors
                    if brand == 'HP' and info.get('corrected') and info['text'].lower() == 'hp':
                        score *= 0.7  # "hp" can be misread from many things
                    
                    brand_scores[brand] = score
                
                # Apply brand-specific conflict rules
                # These override pure confidence when conflicts exist
                
                # ASUS vs Dell: Dell usually wins unless ASUS has strong evidence
                if 'ASUS' in brand_scores and 'Dell' in brand_scores:
                    asus_info = found_brands['ASUS']
                    dell_info = found_brands['Dell']
                    
                    if not asus_info.get('corrected') and asus_info['conf'] > 70:
                        # Strong ASUS reading, keep it
                        brand_scores['Dell'] = 0
                    else:
                        # Prefer Dell (ASUS is common misread)
                        brand_scores['ASUS'] = 0
                
                # HP vs ASUS: ASUS wins if it has real text
                elif 'HP' in brand_scores and 'ASUS' in brand_scores:
                    hp_info = found_brands['HP']
                    asus_info = found_brands['ASUS']
                    
                    if hp_info.get('corrected') and not asus_info.get('corrected') and asus_info['conf'] > 65:
                        brand_scores['HP'] = 0
                    else:
                        brand_scores['ASUS'] = 0
                
                # HP vs Dell: Trust whichever has non-corrected OCR
                elif 'HP' in brand_scores and 'Dell' in brand_scores:
                    hp_info = found_brands['HP']
                    dell_info = found_brands['Dell']
                    
                    # Both real text? Keep higher confidence
                    if not hp_info.get('corrected') and not dell_info.get('corrected'):
                        if hp_info['conf'] < dell_info['conf'] - 15:
                            brand_scores['HP'] = 0
                        else:
                            brand_scores['Dell'] = 0
                    # Only HP is corrected? Keep Dell
                    elif hp_info.get('corrected') and not dell_info.get('corrected'):
                        brand_scores['HP'] = 0
                    # Only Dell is corrected? Keep HP
                    elif dell_info.get('corrected') and not hp_info.get('corrected'):
                        brand_scores['Dell'] = 0
                    # Both corrected? Keep higher conf
                    else:
                        if hp_info['conf'] > dell_info['conf']:
                            brand_scores['Dell'] = 0
                        else:
                            brand_scores['HP'] = 0
                
                # HP vs Acer: Trust whichever has real text
                elif 'HP' in brand_scores and 'Acer' in brand_scores:
                    hp_info = found_brands['HP']
                    acer_info = found_brands['Acer']
                    
                    # Acer has real text and HP is corrected? Keep Acer
                    if not acer_info.get('corrected') and acer_info['conf'] > 55:
                        brand_scores['HP'] = 0
                    # HP has real text and Acer is weak? Keep HP
                    elif not hp_info.get('corrected') and hp_info['conf'] > 55:
                        brand_scores['Acer'] = 0
                    # Both weak/corrected? Higher conf wins
                    else:
                        if hp_info['conf'] > acer_info['conf']:
                            brand_scores['Acer'] = 0
                        else:
                            brand_scores['HP'] = 0
                
                # Dell vs Acer: Usually both are clearly visible, keep higher conf
                elif 'Dell' in brand_scores and 'Acer' in brand_scores:
                    dell_info = found_brands['Dell']
                    acer_info = found_brands['Acer']
                    
                    # Both have real text? Very unusual, keep higher conf
                    if not dell_info.get('corrected') and not acer_info.get('corrected'):
                        if dell_info['conf'] > acer_info['conf']:
                            brand_scores['Acer'] = 0
                        else:
                            brand_scores['Dell'] = 0
                    # One corrected? Keep the non-corrected
                    elif dell_info.get('corrected') and not acer_info.get('corrected'):
                        brand_scores['Dell'] = 0
                    elif acer_info.get('corrected') and not dell_info.get('corrected'):
                        brand_scores['Acer'] = 0
                    else:
                        # Both corrected or both not, keep higher conf
                        if dell_info['conf'] > acer_info['conf']:
                            brand_scores['Acer'] = 0
                        else:
                            brand_scores['Dell'] = 0
                
                # ASUS vs Acer: Prefer whichever has non-corrected OCR
                elif 'ASUS' in brand_scores and 'Acer' in brand_scores:
                    asus_info = found_brands['ASUS']
                    acer_info = found_brands['Acer']
                    
                    # ASUS is often misread, so Acer wins if it has real text
                    if not acer_info.get('corrected') and acer_info['conf'] > 60:
                        brand_scores['ASUS'] = 0
                    elif not asus_info.get('corrected') and asus_info['conf'] > 70:
                        brand_scores['Acer'] = 0
                    else:
                        # Prefer Acer (ASUS is common misread)
                        brand_scores['ASUS'] = 0
                
                # Lenovo vs Others: Lenovo is rarely misread, trust it
                elif 'Lenovo' in brand_scores:
                    lenovo_info = found_brands['Lenovo']
                    if not lenovo_info.get('corrected') and lenovo_info['conf'] > 60:
                        # Clear Lenovo reading, remove others
                        for brand in list(brand_scores.keys()):
                            if brand != 'Lenovo':
                                brand_scores[brand] = 0
                
                # Remove brands with score = 0
                for brand in list(found_brands.keys()):
                    if brand in brand_scores and brand_scores[brand] == 0:
                        del found_brands[brand]
                
                # If still multiple brands, keep highest scoring
                if len(found_brands) > 1:
                    remaining_scores = {b: brand_scores[b] for b in found_brands.keys() if b in brand_scores}
                    if remaining_scores:
                        best_brand = max(remaining_scores.items(), key=lambda x: x[1])[0]
                        found_brands = {best_brand: found_brands[best_brand]}
        
        # Convert to output format
        for brand, info in found_brands.items():
            brands.append({
                "text": brand,
                "confidence": info['conf'] / 100.0,
                "is_brand": True,
                "detection_method": "visual" if info['source'] == 'visual' else "ocr"
            })
        
    elif item_type == "bottle":
        ocr_results = reader.readtext(cropped, paragraph=False)
        for (bbox, text, conf) in ocr_results:
            if conf > 0.3 and len(text.strip()) >= 2:
                brands.append({
                    "text": text.strip(),
                    "confidence": conf,
                    "is_brand": False,
                    "detection_method": "ocr"
                })
    
    elif item_type == "cell phone":
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        ocr_results = reader.readtext(enhanced, paragraph=False)
        seen = set()
        for (bbox, text, conf) in ocr_results:
            text_clean = text.strip()
            if conf > 0.3 and len(text_clean) >= 2 and text_clean.lower() not in seen:
                brands.append({
                    "text": text_clean,
                    "confidence": conf,
                    "is_brand": False,
                    "detection_method": "ocr"
                })
                seen.add(text_clean.lower())
    else:
        ocr_results = reader.readtext(cropped, paragraph=False)
        for (bbox, text, conf) in ocr_results:
            if conf > 0.3 and len(text.strip()) >= 2:
                brands.append({
                    "text": text.strip(),
                    "confidence": conf,
                    "is_brand": False,
                    "detection_method": "ocr"
                })
    
    brands.sort(key=lambda x: x["confidence"], reverse=True)
    return brands


@st.cache_resource
def load_models():
    """Load YOLO and OCR models"""
    try:
        from ultralytics import YOLO
        import easyocr
        
        with st.spinner("Loading AI models... (first time may take a minute)"):
            model = YOLO("yolov8n.pt")
            reader = easyocr.Reader(['en'], gpu=False)
        
        return model, reader
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


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
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x2_min, y2_min, x2_max, y2_max = box2['x1'], box2['y1'], box2['x2'], box2['y2']
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def filter_overlapping_detections(boxes_with_info, iou_threshold=0.5):
    """Remove duplicate detections of the same item using NMS"""
    # Group boxes by label
    label_groups = {}
    for box in boxes_with_info:
        label = box['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(box)
    
    filtered_boxes = []
    
    # Apply NMS per label type
    for label, boxes in label_groups.items():
        # Sort by confidence (highest first)
        boxes.sort(key=lambda x: x['conf'], reverse=True)
        
        keep = []
        while boxes:
            # Keep the highest confidence box
            current = boxes.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU overlap
            boxes = [box for box in boxes if calculate_iou(current, box) < iou_threshold]
        
        filtered_boxes.extend(keep)
    
    return filtered_boxes


def count_items_in_photo(model, reader, image, exclude_labels=None):
    """Detect items and read brands/logos on them
    
    Args:
        model: YOLO model
        reader: EasyOCR reader
        image: Input image
        exclude_labels: List of item labels to exclude (e.g., ['chair', 'couch'])
    """
    if exclude_labels is None:
        exclude_labels = []
    
    results = model(image, verbose=False, imgsz=1280, conf=0.3)  # Added confidence threshold
    result = results[0]
    
    text_readable_items = ["bottle", "laptop", "cell phone", "book", "box"]
    
    boxes_with_info = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        
        # Skip excluded labels
        if label in exclude_labels:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Filter out very small detections (likely errors)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Skip tiny detections (< 1% of image for laptops)
        img_area = image.shape[0] * image.shape[1]
        if label == "laptop" and area < img_area * 0.01:
            continue
        
        boxes_with_info.append({
            'box': box,
            'label': label,
            'conf': conf,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'area': area
        })
    
    # Filter overlapping detections (NMS)
    boxes_with_info = filter_overlapping_detections(boxes_with_info, iou_threshold=0.4)
    
    # Update count after filtering
    count = len(boxes_with_info)
    
    # Create annotated image from scratch with filtered boxes
    annotated = image.copy()
    
    boxes_with_info.sort(key=lambda b: (b['center_y'], b['center_x']))
    
    detections = []
    item_counter = {}
    
    # Draw boxes manually on the annotated image
    for idx, box_info in enumerate(boxes_with_info, 1):
        label = box_info['label']
        conf = box_info['conf']
        x1, y1, x2, y2 = box_info['x1'], box_info['y1'], box_info['x2'], box_info['y2']
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
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
        
        if label in text_readable_items:
            pad = 40
            h, w = image.shape[:2]
            
            x1_crop = max(0, x1 - pad)
            y1_crop = max(0, y1 - pad)
            x2_crop = min(w, x2 + pad)
            y2_crop = min(h, y2 + pad)
            
            for other_box in boxes_with_info:
                if other_box == box_info:
                    continue
                
                ox1, oy1, ox2, oy2 = other_box['x1'], other_box['y1'], other_box['x2'], other_box['y2']
                
                if not (x2_crop < ox1 or x1_crop > ox2 or y2_crop < oy1 or y1_crop > oy2):
                    if x2_crop > ox1 and x2_crop < ox2:
                        x2_crop = ox1 - 5
                    if x1_crop < ox2 and x1_crop > ox1:
                        x1_crop = ox2 + 5
                    if y2_crop > oy1 and y2_crop < oy2:
                        y2_crop = oy1 - 5
                    if y1_crop < oy2 and y1_crop > oy1:
                        y1_crop = oy2 + 5
            
            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                x1_crop, y1_crop, x2_crop, y2_crop = x1, y1, x2, y2
            
            cropped_item = image[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if cropped_item.shape[0] >= 20 and cropped_item.shape[1] >= 20:
                brands = read_text_from_item_isolated(reader, cropped_item, label, item_num)
                detection["brands"] = brands
                
                if brands:
                    max_texts = 2 if label == "laptop" else 3
                    top_brands = brands[:max_texts]
                    brand_text = " | ".join([b["text"] for b in top_brands])
                    full_text = f"#{item_num} {brand_text}"
                    
                    text_color = (255, 0, 0) if label == "laptop" else (0, 0, 255) if label == "bottle" else (0, 255, 255)
                    
                    cv2.putText(annotated, full_text, (x1, max(y1 - 10, 0)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        detections.append(detection)
    
    cv2.putText(annotated, f"TOTAL: {count} items", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    return count, annotated, detections


def compare_quantities(actual_counts, do_data):
    """Compare actual detected quantities against DO expectations"""
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
    """Create Excel report and return as BytesIO"""
    all_detections = []
    text_readable_items = ["bottle", "laptop", "cell phone", "book", "box"]
    
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
                        'Detection Method': '👁️ Visual' if detection_method == 'visual' else '📝 OCR'
                    })
            elif item_type in text_readable_items:
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
            comparison_data.append({
                'Item Type': 'PDF DO INFORMATION',
                'Expected Quantity': '',
                'Actual Quantity': '',
                'Difference': '',
                'Status': ''
            })
            comparison_data.append({
                'Item Type': f"DO Number: {do_data['metadata'].get('do_number', 'N/A')}",
                'Expected Quantity': '',
                'Actual Quantity': '',
                'Difference': '',
                'Status': ''
            })
            comparison_data.append({
                'Item Type': f"Supplier: {do_data['metadata'].get('supplier', 'N/A')}",
                'Expected Quantity': '',
                'Actual Quantity': '',
                'Difference': '',
                'Status': ''
            })
            comparison_data.append({
                'Item Type': '',
                'Expected Quantity': '',
                'Actual Quantity': '',
                'Difference': '',
                'Status': ''
            })
            
            for match in comparison["matches"]:
                comparison_data.append({
                    'Item Type': match["item_type"].upper(),
                    'Expected Quantity': match["expected"],
                    'Actual Quantity': match["actual"],
                    'Difference': match["difference"],
                    'Status': match["status"]
                })
            
            for disc in comparison["discrepancies"]:
                comparison_data.append({
                    'Item Type': disc["item_type"].upper(),
                    'Expected Quantity': disc["expected"],
                    'Actual Quantity': disc["actual"],
                    'Difference': f"{disc['difference']:+d}",
                    'Status': disc["status"]
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_excel(writer, sheet_name='PDF DO Comparison', index=False)
        
        if all_detections:
            df_detections = pd.DataFrame(all_detections)
            df_detections.to_excel(writer, sheet_name='All Detections', index=False)
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Item Summary', index=False)
        
        if brand_summary:
            df_brands = pd.DataFrame(brand_summary)
            df_brands.to_excel(writer, sheet_name='Brand Summary', index=False)
    
    output.seek(0)
    return output


# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────

def main():
    st.title("📦 AI Inventory Detection System")
    st.markdown("### Upload images and optional PDF delivery order for automated inventory verification")
    
    st.info("✨ **Improved Version**: Faster processing (~3x speed) + Better accuracy for Dell/ASUS/HP detection")
    
    # Load models
    model, reader = load_models()
    
    if model is None or reader is None:
        st.error("Failed to load AI models. Please check installation.")
        return
    
    # Sidebar for uploads
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # PDF Upload
        st.subheader("1. PDF Delivery Order (Optional)")
        pdf_file = st.file_uploader("Upload PDF DO", type=['pdf'])
        
        # Image Upload
        st.subheader("2. Images (Required)")
        image_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        st.markdown("---")
        
        # Detection Settings
        st.subheader("⚙️ Detection Settings")
        
        # Items to exclude
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
    
    # Main area
    if process_button:
        if not image_files:
            st.error("Please upload at least one image!")
            return
        
        # Process PDF if provided
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
        
        # Process images
        results_log = []
        actual_counts = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, image_file in enumerate(image_files):
            status_text.text(f"Processing {image_file.name}... ({idx + 1}/{len(image_files)})")
            
            # Read image
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process
            count, annotated, detections = count_items_in_photo(
                model, reader, image, 
                exclude_labels=exclude_items
            )
            
            # Update counts
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
        
        # Display results
        st.header("📊 Detection Results")
        
        for result in results_log:
            with st.expander(f"📸 {result['file']} - {result['count']} items detected", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display annotated image
                    annotated_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                
                with col2:
                    st.subheader("Detected Items:")
                    for i, detection in enumerate(result['detections'], 1):
                        st.write(f"**{i}. {detection['label'].title()}** ({detection['confidence']:.0%})")
                        
                        if detection['brands']:
                            for brand in detection['brands'][:3]:
                                method_icon = "👁️" if brand.get('detection_method') == 'visual' else "📝"
                                st.write(f"   {method_icon} {brand['text']} ({brand['confidence']:.0%})")
        
        st.markdown("---")
        
        # Comparison if PDF provided
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
        
        # Generate Excel Report
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