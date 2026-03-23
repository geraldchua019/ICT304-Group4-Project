import cv2
import sys
import os
from datetime import datetime
import re
import numpy as np


# ─────────────────────────────────────────────
# EXTRACT TEXT FROM PDF DELIVERY ORDER
# ─────────────────────────────────────────────
def extract_pdf_delivery_order(pdf_path=None):
    """
    Extract text from PDF Delivery Order to get item types and quantities.
    Focuses on table extraction for better accuracy.
    """
    
    # Find PDF file
    if pdf_path is None:
        pdf_files = [f for f in os.listdir(".") if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"ℹ️  No PDF DO file found in current directory")
            print("   Running in count-only mode (no comparison)")
            return None
        pdf_path = pdf_files[0]
        print(f"📄 Found PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"ℹ️  PDF file not found: {pdf_path}")
        print("   Running in count-only mode (no comparison)")
        return None
    
    try:
        import pdfplumber
    except ImportError:
        print("❌ Need pdfplumber: pip install pdfplumber")
        return None
    
    print(f"📄 Reading PDF Delivery Order: {pdf_path}")
    print("   Extracting tables from PDF...")
    
    try:
        do_data = {
            "items": {},
            "metadata": {
                "supplier": "N/A",
                "do_number": "N/A",
                "delivery_date": "N/A"
            }
        }
        
        # Common YOLO detectable items (map variations to standard names)
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
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   ✅ PDF has {len(pdf.pages)} page(s)")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"\n   📄 Reading page {page_num}/{len(pdf.pages)}...")
                
                # Get page text for metadata
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
                
                # Extract tables - THIS IS THE KEY PART
                tables = page.extract_tables()
                
                if tables:
                    print(f"   ✅ Found {len(tables)} table(s) on page {page_num}")
                    
                    for table_idx, table in enumerate(tables, 1):
                        print(f"\n      📊 Processing Table {table_idx}:")
                        
                        if not table or len(table) < 2:
                            continue
                        
                        # Find header row (look for "quantity", "item", "description")
                        header_row = None
                        qty_col_idx = None
                        desc_col_idx = None
                        
                        for row_idx, row in enumerate(table[:3]):  # Check first 3 rows for headers
                            if not row:
                                continue
                            
                            row_lower = [str(cell).lower() if cell else "" for cell in row]
                            
                            # Look for quantity column
                            for col_idx, cell in enumerate(row_lower):
                                if 'quantity' in cell or 'qty' in cell:
                                    qty_col_idx = col_idx
                                    header_row = row_idx
                                if 'item' in cell or 'description' in cell:
                                    desc_col_idx = col_idx
                        
                        if qty_col_idx is None:
                            print(f"         ⚠️  No 'Quantity' column found, skipping table")
                            continue
                        
                        if desc_col_idx is None:
                            # Try to find description column (usually column 1 or 2)
                            desc_col_idx = 1 if len(table[0]) > 1 else 0
                        
                        print(f"         ✅ Found Quantity in column {qty_col_idx + 1}")
                        print(f"         ✅ Using Description from column {desc_col_idx + 1}")
                        
                        # Process data rows (skip header)
                        for row_idx in range(header_row + 1 if header_row else 1, len(table)):
                            row = table[row_idx]
                            
                            if not row or len(row) <= max(qty_col_idx, desc_col_idx):
                                continue
                            
                            # Get quantity (should be a number)
                            qty_cell = str(row[qty_col_idx]) if row[qty_col_idx] else ""
                            qty_cell = qty_cell.strip()
                            
                            # Get description
                            desc_cell = str(row[desc_col_idx]) if row[desc_col_idx] else ""
                            desc_cell = desc_cell.lower().strip()
                            
                            # Skip empty rows
                            if not qty_cell or not desc_cell:
                                continue
                            
                            # Extract quantity as number
                            qty_match = re.search(r'\b(\d+)\b', qty_cell)
                            if not qty_match:
                                continue
                            
                            quantity = int(qty_match.group(1))
                            
                            # Skip unreasonable quantities
                            if quantity < 1 or quantity > 1000:
                                continue
                            
                            # Match description to known items
                            matched_item = None
                            for standard_name, variations in item_mappings.items():
                                for variation in variations:
                                    if variation in desc_cell:
                                        matched_item = standard_name
                                        break
                                if matched_item:
                                    break
                            
                            if matched_item:
                                # Add or update quantity
                                if matched_item in do_data["items"]:
                                    do_data["items"][matched_item] += quantity
                                else:
                                    do_data["items"][matched_item] = quantity
                                
                                print(f"         🔍 Row {row_idx}: '{desc_cell[:40]}...' → {matched_item} = {quantity}")
        
        # Extract metadata from text
        combined_text = "\n".join(all_text)
        
        # Extract DO number
        do_match = re.search(r'do[-\s]?(\d{4}[-]?\d+)', combined_text.lower())
        if do_match:
            do_data["metadata"]["do_number"] = do_match.group(0).upper()
            print(f"\n   🔍 Found DO Number: {do_data['metadata']['do_number']}")
        
        # Extract supplier
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
                            print(f"   🔍 Found Supplier: {supplier}")
                elif i + 1 < len(lines):
                    supplier = lines[i + 1].strip()
                    if supplier and len(supplier) > 3 and not supplier[0].isdigit():
                        do_data["metadata"]["supplier"] = supplier
                        print(f"   🔍 Found Supplier: {supplier}")
                break
        
        # Summary
        if do_data["items"]:
            print(f"\n✅ Successfully extracted from PDF DO:")
            print(f"   DO Number: {do_data['metadata']['do_number']}")
            print(f"   Supplier: {do_data['metadata']['supplier']}")
            print(f"   Expected items:")
            for item_type, qty in do_data["items"].items():
                print(f"      - {item_type}: {qty} units")
            print()
            return do_data
        else:
            print("\n⚠️  Could not extract item quantities from PDF")
            print("   Make sure PDF contains:")
            print("   - A table with 'Quantity' column header")
            print("   - Item descriptions (laptop, phone, bottle, etc.)")
            print("   Running in count-only mode\n")
            return None
        
    except Exception as e:
        print(f"⚠️  Error reading PDF: {e}")
        import traceback
        traceback.print_exc()
        print("   Running in count-only mode")
        return None


# ─────────────────────────────────────────────
# HELPER: Check if text is gibberish
# ─────────────────────────────────────────────
def is_gibberish(text):
    """
    Check if OCR text is likely gibberish (FSLS, ISLS, etc.)
    But first check if it's a known OCR error pattern
    """
    # First check if this "gibberish" is actually a brand misread
    from_correction, _ = correct_ocr_errors(text)
    if from_correction:
        return False  # Not gibberish, it's a correctable brand!
    
    text_lower = text.lower().replace(' ', '')
    
    if len(text_lower) < 2:
        return True
    
    # Check vowel ratio - real words have vowels
    vowels = 'aeiou'
    vowel_count = sum(1 for c in text_lower if c in vowels)
    
    # No vowels in words >3 chars = gibberish
    if len(text_lower) > 3 and vowel_count == 0:
        return True
    
    # Too many consonants in a row (>4) is unusual
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


# ─────────────────────────────────────────────
# OCR ERROR CORRECTION HELPER
# ─────────────────────────────────────────────
def correct_ocr_errors(text):
    """
    Map common OCR misreads to likely brands
    Returns (corrected_text, brand_name) or (None, None) if no match
    """
    text_clean = text.lower().replace(' ', '').replace('-', '').replace('_', '')
    
    # HP patterns - check FIRST before ASUS patterns
    hp_patterns = ['hp', '(p', ')p', 'ip', '1p', 'hd', 'np']
    for pattern in hp_patterns:
        if pattern == text_clean:
            return ('HP', 'HP')
    
    # ASUS common misreads
    asus_patterns = [
        'fsls', 'fls', 'asls', 'fsus', 'isls', 'a5u5', 'a5us', 'asu5', 
        'fels', 'felsi'
    ]
    for pattern in asus_patterns:
        if pattern in text_clean or text_clean in pattern:
            return ('ASUS', 'ASUS')
        if len(text_clean) >= 4 and len(pattern) == 4:
            matches = sum(1 for a, b in zip(text_clean[:4], pattern) if a == b)
            if matches >= 3:
                return ('ASUS', 'ASUS')
    
    # Dell common misreads
    dell_patterns = ['vell', 'deil', 'deii', 'de11', 'deli']
    for pattern in dell_patterns:
        if pattern in text_clean or text_clean in pattern:
            return ('Dell', 'Dell')
    
    # Acer common misreads
    acer_patterns = ['aser', 'acar', 'acer', 'aeer']
    for pattern in acer_patterns:
        if pattern in text_clean or text_clean in pattern:
            return ('Acer', 'Acer')
    
    return (None, None)


# ─────────────────────────────────────────────
# PREPROCESSING METHODS
# ─────────────────────────────────────────────
def preprocess_image(image, method='clahe'):
    """Apply various preprocessing methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    elif method == 'threshold':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == 'adaptive':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    elif method == 'sharp':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(gray, -1, kernel)
    elif method == 'denoise':
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)
    else:
        return gray


# ─────────────────────────────────────────────
# RUN OCR WITH MULTIPLE METHODS
# ─────────────────────────────────────────────
def run_ocr_multiple_ways(reader, image):
    """Try OCR with multiple preprocessing methods and regions"""
    methods = ['clahe', 'threshold', 'sharp', 'denoise', 'gray']
    all_texts = []
    h, w = image.shape[:2]
    
    # Scan THREE regions: top 40%, middle 20-60%, and full
    regions = [
        ('top', image[:int(h * 0.4), :], 1.0),
        ('middle', image[int(h * 0.2):int(h * 0.6), :], 0.95),
        ('full', image, 0.8),
    ]
    
    for region_name, region_img, conf_multiplier in regions:
        for method in methods:
            processed = preprocess_image(region_img, method)
            
            # Try OCR WITHOUT allowlist first
            try:
                results1 = reader.readtext(processed, paragraph=False)
                all_texts.extend([(text, conf * conf_multiplier, method, region_name) 
                                for _, text, conf in results1])
            except:
                pass
            
            # Try with letter-only allowlist
            try:
                results2 = reader.readtext(
                    processed,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                )
                all_texts.extend([(text, conf * conf_multiplier, method, region_name) 
                                for _, text, conf in results2])
            except:
                pass
    
    return all_texts


# ─────────────────────────────────────────────
# DETECT LAPTOP LOGOS (IMPROVED)
# ─────────────────────────────────────────────
def detect_laptop_logo(image):
    """
    Detect laptop brand logos visually (for brands without text like Apple)
    Uses multiple detection methods: color analysis, edge detection, and shape matching
    """
    h, w = image.shape[:2]
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detections = {}
    
    # ============================================================================
    # APPLE LOGO DETECTION
    # ============================================================================
    apple_score = 0
    
    # Check 1: Overall color (Apple = silver, space gray, or black)
    avg_color = np.mean(image, axis=(0, 1))
    b, g, r = avg_color
    color_diff = max(abs(r-g), abs(g-b), abs(r-b))
    is_grayscale = color_diff < 20  # Low saturation = metallic/gray
    
    if is_grayscale:
        avg_brightness = np.mean(avg_color)
        if 130 < avg_brightness < 200:  # Silver range
            apple_score += 40
        elif 40 < avg_brightness < 90:  # Space gray/dark range
            apple_score += 35
    
    # Check 2: Look for metallic/silver pixels
    lower_silver = np.array([0, 0, 140])
    upper_silver = np.array([180, 35, 230])
    silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
    silver_ratio = np.count_nonzero(silver_mask) / silver_mask.size
    
    if silver_ratio > 0.25:
        apple_score += int(min(silver_ratio * 80, 40))
    
    # Check 3: Look for the Apple logo shape in center-top
    center_top = gray[:int(h*0.4), int(w*0.35):int(w*0.65)]
    
    # Apply edge detection to find logo outline
    edges = cv2.Canny(center_top, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Apple logo creates a moderate edge density (not too much text)
    if 0.02 < edge_density < 0.08:
        apple_score += 20
    
    detections['Apple'] = apple_score
    
    # ============================================================================
    # HP LOGO DETECTION
    # ============================================================================
    hp_score = 0
    
    # HP blue logo detection
    hp_blue_lower = np.array([100, 100, 50])
    hp_blue_upper = np.array([130, 255, 255])
    hp_blue_mask = cv2.inRange(hsv, hp_blue_lower, hp_blue_upper)
    hp_blue_ratio = np.count_nonzero(hp_blue_mask) / hp_blue_mask.size
    
    if hp_blue_ratio > 0.02:
        hp_score += int(min(hp_blue_ratio * 200, 60))
    
    detections['HP'] = hp_score
    
    # ============================================================================
    # DELL LOGO DETECTION
    # ============================================================================
    dell_score = 0
    
    # Dell blue logo detection (slightly different blue than HP)
    dell_blue_lower = np.array([95, 80, 40])
    dell_blue_upper = np.array([115, 255, 255])
    dell_blue_mask = cv2.inRange(hsv, dell_blue_lower, dell_blue_upper)
    dell_blue_ratio = np.count_nonzero(dell_blue_mask) / dell_blue_mask.size
    
    if dell_blue_ratio > 0.02:
        dell_score += int(min(dell_blue_ratio * 200, 60))
    
    detections['Dell'] = dell_score
    
    # ============================================================================
    # ASUS LOGO DETECTION
    # ============================================================================
    asus_score = 0
    
    # ASUS gold/yellow logo detection
    asus_gold_lower = np.array([15, 100, 100])
    asus_gold_upper = np.array([35, 255, 255])
    asus_gold_mask = cv2.inRange(hsv, asus_gold_lower, asus_gold_upper)
    asus_gold_ratio = np.count_nonzero(asus_gold_mask) / asus_gold_mask.size
    
    if asus_gold_ratio > 0.01:
        asus_score += int(min(asus_gold_ratio * 300, 60))
    
    detections['ASUS'] = asus_score
    
    # ============================================================================
    # Return the highest scoring brand if above threshold
    # ============================================================================
    
    if detections:
        best_brand = max(detections.items(), key=lambda x: x[1])
        brand_name, score = best_brand
        
        # Lower threshold for detection
        if score >= 35:
            return {
                'brand': brand_name,
                'confidence': min(score / 100.0, 0.95),
                'score': score,
                'method': 'Visual (logo detection)'
            }
    
    return None


# ─────────────────────────────────────────────
# LOAD YOLO
# ─────────────────────────────────────────────
def load_yolo():
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        print("📥 Loading YOLO AI model...")
        model = YOLO("yolov8n.pt")
        print("✅ YOLO ready!\n")
        return model
    except ImportError:
        print("❌ Need to install ultralytics first:")
        print("   pip install ultralytics")
        sys.exit(1)


# ─────────────────────────────────────────────
# LOAD EASYOCR
# ─────────────────────────────────────────────
def load_ocr():
    """Load EasyOCR reader"""
    try:
        import easyocr
        print("📥 Loading OCR model (first time may take a minute)...")
        reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have a GPU
        print("✅ OCR ready!\n")
        return reader
    except ImportError:
        print("❌ Need to install easyocr first:")
        print("   pip install easyocr")
        sys.exit(1)


# ─────────────────────────────────────────────
# READ TEXT/BRANDS FROM ISOLATED ITEM CROP
# ─────────────────────────────────────────────
def read_text_from_item_isolated(reader, cropped, item_type, item_num):
    """
    Process a single isolated item crop (already extracted from image).
    This ensures NO contamination from other items.
    
    Args:
        reader: EasyOCR reader
        cropped: Already cropped image containing ONLY this item
        item_type: Type of item (laptop, bottle, etc.)
        item_num: Item number (for logging)
    
    Returns:
        List of brand/text info found
    """
    
    brands = []
    
    # Comprehensive brand dictionary
    known_brands = {
        'Dell': ['dell', 'latitude', 'inspiron', 'xps', 'precision', 'alienware', 'vell', 'deil'],
        'HP': ['hp', 'hewlett', 'packard', 'pavilion', 'elitebook', 'probook', 'envy', 'omen'],
        'ASUS': ['asus', 'rog', 'zenbook', 'vivobook', 'tuf'],
        'Acer': ['acer', 'aspire', 'predator', 'nitro', 'swift'],
        'Lenovo': ['lenovo', 'thinkpad', 'ideapad', 'yoga', 'legion'],
        'Apple': ['apple', 'macbook', 'mac'],
        'MSI': ['msi', 'gaming'],
        'Samsung': ['samsung', 'galaxy'],
        'Razer': ['razer', 'blade'],
        'Microsoft': ['microsoft', 'surface'],
        'Toshiba': ['toshiba'],
        'Sony': ['sony', 'vaio'],
    }
    
    # FOR LAPTOPS: Use improved multi-method OCR
    if item_type == "laptop":
        print(f"      🔍 Analyzing laptop #{item_num} (isolated crop)...")
        
        # Upscale 3x for better OCR
        upscaled = cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Try visual logo detection first
        print(f"      👁️  Running visual logo detection...")
        detected_logo = detect_laptop_logo(upscaled)
        
        if detected_logo:
            print(f"      ✨ Visual: {detected_logo['brand']} (score:{detected_logo['score']}, conf:{detected_logo['confidence']:.0%})")
        
        # Run multi-method OCR
        print(f"      📝 Running OCR...")
        all_texts = run_ocr_multiple_ways(reader, upscaled)
        
        # Process results
        seen_texts = set()
        found_brands = {}
        
        for text, ocr_conf, method, region in all_texts:
            text_clean = text.strip()
            text_lower = text_clean.lower()
            
            if len(text_clean) < 2 or text_lower in seen_texts:
                continue
            
            # Check OCR error correction
            corrected_text, ocr_corrected_brand = correct_ocr_errors(text_clean)
            if ocr_corrected_brand:
                brand_key = f"{ocr_corrected_brand}_{region}"
                if brand_key not in found_brands or ocr_conf > found_brands[brand_key]['conf']:
                    found_brands[brand_key] = {
                        'brand': ocr_corrected_brand,
                        'conf': ocr_conf,
                        'text': text_clean,
                        'region': region,
                        'corrected': True
                    }
                    print(f"         🔧 '{text_clean}' → {ocr_corrected_brand} ({ocr_conf:.0%}, {region})")
                seen_texts.add(text_lower)
                continue
            
            if is_gibberish(text_clean):
                continue
            
            letter_count = sum(c.isalpha() for c in text_clean)
            if letter_count < len(text_clean) * 0.5:
                continue
            
            # Check against known brands
            matched_brand = None
            for brand_name, variations in known_brands.items():
                for variant in variations:
                    if variant in text_lower:
                        matched_brand = brand_name
                        break
                if matched_brand:
                    break
            
            if matched_brand:
                brand_key = f"{matched_brand}_{region}"
                if brand_key not in found_brands or ocr_conf > found_brands[brand_key]['conf']:
                    found_brands[brand_key] = {
                        'brand': matched_brand,
                        'conf': ocr_conf,
                        'text': text_clean,
                        'region': region,
                        'corrected': False
                    }
                    print(f"         ✅ {matched_brand} via '{text_clean}' ({ocr_conf:.0%}, {region})")
                seen_texts.add(text_lower)
        
        has_meaningful_text = len(found_brands) > 0
        
        # Add visual logo detection
        if detected_logo:
            if not has_meaningful_text:
                print(f"         💡 No text - boosting visual confidence")
                detected_logo['confidence'] = min(detected_logo['confidence'] * 1.3, 0.95)
                detected_logo['score'] += 20
            
            found_brands['visual'] = {
                'brand': detected_logo['brand'],
                'conf': detected_logo['confidence'] * 100,
                'text': '(visual)',
                'region': 'visual',
                'corrected': False
            }
        
        # Consolidate brands
        brand_consolidated = {}
        region_priority = {'visual': 4, 'top': 3, 'middle': 2, 'full': 1}
        
        for brand_key, info in found_brands.items():
            brand = info['brand']
            region = info['region']
            
            if brand not in brand_consolidated:
                brand_consolidated[brand] = info
            else:
                existing = brand_consolidated[brand]
                if region_priority[region] > region_priority[existing['region']]:
                    brand_consolidated[brand] = info
                elif region_priority[region] == region_priority[existing['region']] and info['conf'] > existing['conf']:
                    brand_consolidated[brand] = info
        
        print(f"         📊 Consolidated: {list(brand_consolidated.keys())}")
        
        # CONFLICT RESOLUTION: ASUS vs HP
        if 'ASUS' in brand_consolidated and 'HP' in brand_consolidated:
            asus_info = brand_consolidated['ASUS']
            hp_info = brand_consolidated['HP']
            
            if not hp_info.get('corrected', False) and hp_info['conf'] > 20:
                del brand_consolidated['ASUS']
            elif asus_info.get('corrected', False) and asus_info['text'].lower() == 'fls' and asus_info['conf'] < 70:
                del brand_consolidated['ASUS']
            elif asus_info['region'] == 'top' and asus_info['conf'] > 75:
                del brand_consolidated['HP']
            elif not hp_info.get('corrected', False) and asus_info.get('corrected', False):
                del brand_consolidated['ASUS']
            else:
                del brand_consolidated['HP']
        
        # CONFLICT RESOLUTION: Apple vs Others
        if 'Apple' in brand_consolidated and brand_consolidated['Apple']['region'] == 'visual':
            apple_conf = brand_consolidated['Apple']['conf']
            
            if apple_conf > 50:
                other_brands = [b for b in brand_consolidated.keys() if b != 'Apple']
                if other_brands:
                    keep_apple = True
                    for other_brand in other_brands:
                        other_info = brand_consolidated[other_brand]
                        if other_info['region'] == 'top' and other_info['conf'] > 80 and not other_info.get('corrected', False):
                            keep_apple = False
                            break
                    
                    if keep_apple:
                        print(f"         → Keeping only Apple (visual wins)")
                        brand_consolidated = {'Apple': brand_consolidated['Apple']}
        
        # Convert to output format
        print(f"      ✨ FINAL for laptop #{item_num}:")
        for brand, info in brand_consolidated.items():
            print(f"         🏆 {brand} ({info['conf']:.0%}) via {info['region']}")
            brands.append({
                "text": brand,
                "confidence": info['conf'] / 100.0,
                "is_brand": True,
                "detection_method": "visual" if info['region'] == 'visual' else "ocr"
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
        
        seen = set()
        for img in [cropped, enhanced]:
            ocr_results = reader.readtext(img, paragraph=False)
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


# ─────────────────────────────────────────────
# DETECT + READ BRANDS IN A PHOTO
# ─────────────────────────────────────────────
def count_items_in_photo(model, reader, image_path):
    """
    Detect items and read brands/logos on them.
    Each item is processed independently using ONLY its bounding box region.

    Returns:
        count, annotated_image, detections_list
    """

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load: {image_path}")
        return 0, None, []

    # Run YOLO detection
    print(f"🤖 Running AI on: {image_path}")
    results = model(image, verbose=False, imgsz=1280)
    result = results[0]

    count = len(result.boxes)

    # Get annotated image with boxes
    annotated = result.plot()

    # Items that we want to read text/logos from
    text_readable_items = ["bottle", "laptop", "cell phone", "book", "box"]

    # Sort boxes by position (left to right, top to bottom) for consistent ordering
    boxes_with_info = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        boxes_with_info.append({
            'box': box,
            'label': label,
            'conf': conf,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2
        })
    
    # Sort by position (top to bottom, then left to right)
    boxes_with_info.sort(key=lambda b: (b['center_y'], b['center_x']))

    # Extract detections + run OCR/logo detection on readable items
    detections = []
    item_counter = {}
    
    for idx, box_info in enumerate(boxes_with_info, 1):
        box = box_info['box']
        label = box_info['label']
        conf = box_info['conf']
        x1, y1, x2, y2 = box_info['x1'], box_info['y1'], box_info['x2'], box_info['y2']

        # Count this item type
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

        # If it's a text-readable item, process it
        if label in text_readable_items:
            print(f"   🔍 {label} #{item_num} at ({x1},{y1})→({x2},{y2})")
            
            # CRITICAL: Crop ONLY this bounding box (with smart padding)
            pad = 40
            h, w = image.shape[:2]
            
            x1_crop = max(0, x1 - pad)
            y1_crop = max(0, y1 - pad)
            x2_crop = min(w, x2 + pad)
            y2_crop = min(h, y2 + pad)
            
            # Check for overlap with other boxes and adjust
            for other_box in boxes_with_info:
                if other_box == box_info:
                    continue
                
                ox1, oy1, ox2, oy2 = other_box['x1'], other_box['y1'], other_box['x2'], other_box['y2']
                
                # Reduce padding if it would overlap
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
            
            print(f"      📐 Isolated crop: ({x1_crop},{y1_crop})→({x2_crop},{y2_crop})")
            
            # Extract ONLY this item
            cropped_item = image[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if cropped_item.shape[0] < 20 or cropped_item.shape[1] < 20:
                print(f"      ⚠️  Too small")
                detections.append(detection)
                continue
            
            # Process in isolation
            brands = read_text_from_item_isolated(reader, cropped_item, label, item_num)
            detection["brands"] = brands

            # Draw on annotated image
            if brands:
                max_texts = 2 if label == "laptop" else 3
                top_brands = brands[:max_texts]
                brand_text = " | ".join([b["text"] for b in top_brands])
                full_text = f"#{item_num} {brand_text}"
                
                text_color = (255, 0, 0) if label == "laptop" else (0, 0, 255) if label == "bottle" else (0, 255, 255)
                
                cv2.putText(annotated, full_text, (x1, max(y1 - 10, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        detections.append(detection)

    # Add total count
    cv2.putText(annotated, f"TOTAL: {count} items", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return count, annotated, detections


# ─────────────────────────────────────────────
# COMPARE ACTUAL VS EXPECTED
# ─────────────────────────────────────────────
def compare_quantities(actual_counts, do_data):
    """
    Compare actual detected quantities against DO expectations.
    """
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


# ─────────────────────────────────────────────
# EXPORT TO EXCEL
# ─────────────────────────────────────────────
def export_to_excel(results_log, do_data=None, comparison=None):
    """Export detection results to Excel"""
    try:
        import pandas as pd
    except ImportError:
        print("❌ Need pandas & openpyxl: pip install pandas openpyxl")
        return None

    all_detections = []
    summary_data = []
    
    # Define which items we tried to read text/brands from
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
                # Only add text-readable items even if no brand found
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"detection_results_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
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
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    return excel_filename


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ISOLATED BOUNDING BOX DETECTION (NO CROSS-CONTAMINATION)")
    print("=" * 70)
    print("\nEach laptop is processed independently using ONLY its bounding box.\n")

    do_data = extract_pdf_delivery_order()
    model = load_yolo()
    reader = load_ocr()

    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = [f for f in os.listdir(".")
                   if any(f.endswith(ext) for ext in image_extensions)]

    if not image_files:
        print("❌ No images found!")
        sys.exit(1)

    print("📸 Images:")
    for i, img in enumerate(image_files, 1):
        print(f"   {i}. {img}")

    print("\nOptions:")
    print("  1. Process all images")
    print("  2. Choose specific image")

    choice = input("\nChoice (1 or 2): ").strip()

    images_to_process = []

    if choice == "1":
        images_to_process = image_files
    else:
        img_num = int(input(f"Which image (1-{len(image_files)}): ")) - 1
        if 0 <= img_num < len(image_files):
            images_to_process = [image_files[img_num]]
        else:
            print("Invalid choice")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("PROCESSING IMAGES")
    print("=" * 70)

    results_log = []
    actual_counts = {}

    for image_path in images_to_process:
        print(f"\n📸 {image_path}")
        print("─" * 70)

        count, annotated, detections = count_items_in_photo(model, reader, image_path)

        if count == 0:
            print("   ❌ No items detected")
        else:
            print(f"   ✅ Detected {count} item(s):")
            for i, d in enumerate(detections, 1):
                print(f"      {i}. {d['label']} ({d['confidence']:.0%})")
                
                item_type = d['label']
                actual_counts[item_type] = actual_counts.get(item_type, 0) + 1

                if d["brands"]:
                    if d["label"] == "laptop":
                        for b in d["brands"][:2]:
                            marker = "👁️" if b.get("detection_method") == "visual" else "🏷️"
                            print(f"          {marker} {b['text']} ({b['confidence']:.0%})")
                    else:
                        for b in d["brands"][:3]:
                            print(f"          📛 {b['text']} ({b['confidence']:.0%})")

            output_name = f"detected_{image_path}"
            cv2.imwrite(output_name, annotated)
            print(f"   💾 Saved: {output_name}")

            import subprocess
            subprocess.Popen(["start", output_name], shell=True)

        results_log.append({
            "file": image_path,
            "count": count,
            "detections": detections
        })

    comparison = None
    if do_data:
        print("\n" + "=" * 70)
        print("PDF DO COMPARISON")
        print("=" * 70)
        
        comparison = compare_quantities(actual_counts, do_data)
        
        if comparison["matches"]:
            print("\n✅ MATCHING:")
            for match in comparison["matches"]:
                print(f"   {match['item_type']}: {match['actual']}/{match['expected']} ✓")
        
        if comparison["discrepancies"]:
            print("\n⚠️  DISCREPANCIES:")
            for disc in comparison["discrepancies"]:
                print(f"   {disc['item_type']}: Expected {disc['expected']}, Got {disc['actual']} (Diff: {disc['difference']:+d})")
        else:
            print("\n✅ ALL MATCH!")

    print("\n" + "=" * 70)
    print("EXPORTING TO EXCEL")
    print("=" * 70)
    
    excel_file = export_to_excel(results_log, do_data, comparison)
    
    if excel_file:
        print(f"✅ Excel: {excel_file}")
        
        try:
            import subprocess
            subprocess.Popen(["start", excel_file], shell=True)
        except:
            pass

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if do_data and comparison:
        print(f"📄 DO: {do_data['metadata'].get('do_number', 'N/A')}")
        print(f"🏢 Supplier: {do_data['metadata'].get('supplier', 'N/A')}")
        print(f"📊 Status: {'✅ ALL MATCH' if not comparison['has_discrepancy'] else '⚠️ DISCREPANCIES'}")
        print()
    
    for r in results_log:
        print(f"  {r['file']}: {r['count']} items")
        for d in r["detections"]:
            if d["brands"]:
                brands_str = ", ".join([b["text"] for b in d["brands"][:3]])
                print(f"      {d['label']}: {brands_str}")
    print("=" * 70)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()