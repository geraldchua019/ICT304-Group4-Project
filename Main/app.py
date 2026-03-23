"""
app.py
──────────────────────────────────────────────────────────────────────
Warehouse Intelligence System — Integrated Application
Subsystems:
  1. Object Scanner   (YOLO + CLIP brand detection, PDF DO verification)
  2. Auto-Picker      (route optimisation, inventory management)
  3. Email / Courier  (ML shipping selection, FedEx email dispatch)

All three subsystems share a single SharedStore (session state).
Navigation: top-level tabs keep each subsystem's full original UI.
A "Pipeline Overview" tab shows the end-to-end flow.
──────────────────────────────────────────────────────────────────────
"""

import streamlit as st

# ── Page config MUST be first ────────────────────────────────────────
st.set_page_config(
    page_title="Warehouse Intelligence System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stdlib / third-party ─────────────────────────────────────────────
import cv2
import os
import re
import sys
import json
import base64
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import uuid
import time

from PIL import Image as PILImage, ImageDraw, ImageFont

# ── Add parent folder to path so ai_decision_engine can be found
#    regardless of which directory the app is launched from ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Shared state ─────────────────────────────────────────────────────
from shared_state import (
    SharedStore, ScanResult, ScanDetection,
    PickerOrder, ShipmentRecord, get_store,
)
from database import get_db
from db_viewer import render_db_tab

# ═══════════════════════════════════════════════════════════════════════
# SECTION 0 — GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.wis-header{font-size:2rem;font-weight:700;color:#1a1a2e;margin-bottom:.3rem}
.wis-sub  {font-size:.95rem;color:#555;margin-bottom:1.5rem}
.metric-card{
    background:linear-gradient(135deg,#667eea,#764ba2);
    padding:1.1rem;border-radius:12px;color:#fff;text-align:center;
    box-shadow:0 4px 14px rgba(102,126,234,.3);
}
.metric-card h3{font-size:1.8rem;margin:0;font-weight:700}
.metric-card p{font-size:.8rem;margin:0;opacity:.9}
.success-card{background:linear-gradient(135deg,#11998e,#38ef7d)}
.fail-card   {background:linear-gradient(135deg,#eb3349,#f45c43)}
.rate-card   {background:linear-gradient(135deg,#4facfe,#00f2fe)}
.pipeline-badge{
    display:inline-block;padding:.35rem .9rem;border-radius:20px;
    font-size:.82rem;font-weight:600;margin:.2rem;
}
.badge-done  {background:#d4edda;color:#155724}
.badge-active{background:#cce5ff;color:#004085}
.badge-wait  {background:#e2e3e5;color:#383d41}
.email-preview{
    background:#f8f9fa !important;
    border:1px solid #dee2e6;border-radius:8px;
    padding:1.2rem;font-family:monospace;font-size:.8rem;
    white-space:pre-wrap;max-height:480px;overflow-y:auto;
    color:#212529 !important;
}
.email-preview *{color:#212529 !important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — SUBSYSTEM 1: OBJECT SCANNER
# ═══════════════════════════════════════════════════════════════════════

BRAND_LABELS = [
    "a Dell laptop","an HP laptop","an ASUS laptop","an Acer laptop",
    "a Lenovo laptop","an Apple MacBook","a Samsung laptop or phone",
    "an unknown laptop with no visible brand",
]
BRAND_LABEL_MAP = {
    "a Dell laptop":"Dell","an HP laptop":"HP","an ASUS laptop":"ASUS",
    "an Acer laptop":"Acer","a Lenovo laptop":"Lenovo",
    "an Apple MacBook":"Apple","a Samsung laptop or phone":"Samsung",
    "an unknown laptop with no visible brand":"Unknown",
}
PHONE_LABELS = [
    "a Samsung phone","an Apple iPhone","a Google Pixel phone",
    "a Huawei phone","an unknown phone with no visible brand",
]
PHONE_LABEL_MAP = {
    "a Samsung phone":"Samsung","an Apple iPhone":"Apple",
    "a Google Pixel phone":"Google","a Huawei phone":"Huawei",
    "an unknown phone with no visible brand":"Unknown",
}


def classify_brand_clip(clip_model, clip_processor, cropped_image, item_type="laptop"):
    try:
        import torch
        rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        labels = PHONE_LABELS if item_type == "cell phone" else BRAND_LABELS
        lmap   = PHONE_LABEL_MAP if item_type == "cell phone" else BRAND_LABEL_MAP
        inputs = clip_processor(text=labels, images=pil_image,
                                return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        results = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)
        top_label, top_conf = results[0]
        brand_name = lmap[top_label]
        if brand_name == "Unknown" or top_conf < 0.4:
            return []
        return [{"text": brand_name, "confidence": top_conf,
                 "is_brand": True, "detection_method": "clip"}]
    except Exception:
        return []


def extract_pdf_delivery_order(pdf_file):
    try:
        import pdfplumber
    except ImportError:
        st.error("❌ Need pdfplumber: pip install pdfplumber")
        return None
    try:
        do_data = {"items": {}, "metadata": {"supplier":"N/A","do_number":"N/A","delivery_date":"N/A"}}
        item_mappings = {
            'laptop':['laptop','notebook','computer','latitude'],
            'bottle':['bottle','water bottle','drink'],
            'cup':['cup','mug'],
            'cell phone':['phone','cell phone','mobile','smartphone','cellphone','iphone','samsung'],
            'book':['book'],'box':['box','carton','package'],
            'keyboard':['keyboard'],'mouse':['mouse'],
            'backpack':['backpack','bag','rucksack'],'chair':['chair','seat'],
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
                        header_row = qty_col_idx = desc_col_idx = None
                        for row_idx, row in enumerate(table[:3]):
                            if not row: continue
                            row_lower = [str(c).lower() if c else "" for c in row]
                            for ci, cell in enumerate(row_lower):
                                if 'quantity' in cell or 'qty' in cell:
                                    qty_col_idx = ci; header_row = row_idx
                                if 'item' in cell or 'description' in cell:
                                    desc_col_idx = ci
                        if qty_col_idx is None: continue
                        if desc_col_idx is None:
                            desc_col_idx = 1 if len(table[0]) > 1 else 0
                        for row in table[(header_row+1 if header_row else 1):]:
                            if not row or len(row) <= max(qty_col_idx, desc_col_idx): continue
                            qty_cell  = str(row[qty_col_idx] or "").strip()
                            desc_cell = str(row[desc_col_idx] or "").lower().strip()
                            if not qty_cell or not desc_cell: continue
                            qm = re.search(r'\b(\d+)\b', qty_cell)
                            if not qm: continue
                            quantity = int(qm.group(1))
                            if quantity < 1 or quantity > 1000: continue
                            for sname, variations in item_mappings.items():
                                if any(v in desc_cell for v in variations):
                                    do_data["items"][sname] = do_data["items"].get(sname, 0) + quantity
                                    break
        combined = "\n".join(all_text)
        dom = re.search(r'do[-\s]?(\d{4}[-]?\d+)', combined.lower())
        if dom:
            do_data["metadata"]["do_number"] = dom.group(0).upper()
        lines = combined.split('\n')
        for i, line in enumerate(lines):
            ll = line.lower()
            if 'supplier' in ll or 'company name' in ll:
                if ':' in line:
                    p = line.split(':', 1)
                    if len(p) > 1 and p[1].strip():
                        do_data["metadata"]["supplier"] = p[1].strip()
                elif i+1 < len(lines) and lines[i+1].strip():
                    do_data["metadata"]["supplier"] = lines[i+1].strip()
                break
        return do_data if do_data["items"] else None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def calculate_iou(b1, b2):
    ix1 = max(b1['x1'], b2['x1']); iy1 = max(b1['y1'], b2['y1'])
    ix2 = min(b1['x2'], b2['x2']); iy2 = min(b1['y2'], b2['y2'])
    if ix2 < ix1 or iy2 < iy1: return 0.0
    ia = (ix2-ix1)*(iy2-iy1)
    ua = (b1['x2']-b1['x1'])*(b1['y2']-b1['y1']) + \
         (b2['x2']-b2['x1'])*(b2['y2']-b2['y1']) - ia
    return ia/ua if ua else 0.0


def filter_overlapping_detections(boxes, iou_threshold=0.5):
    groups = {}
    for b in boxes:
        groups.setdefault(b['label'], []).append(b)
    out = []
    for label, blist in groups.items():
        blist.sort(key=lambda x: x['conf'], reverse=True)
        keep = []
        while blist:
            cur = blist.pop(0); keep.append(cur)
            blist = [b for b in blist if calculate_iou(cur, b) < iou_threshold]
        out.extend(keep)
    return out


@st.cache_resource
def load_models():
    try:
        from ultralytics import YOLO
        from transformers import CLIPProcessor, CLIPModel
        with st.spinner("Loading AI models…"):
            yolo = YOLO("yolov8n.pt")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()
        return yolo, clip_model, clip_proc
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


@st.cache_resource
def load_ocr_reader():
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        import easyocr
        return easyocr.Reader(["en"], gpu=use_gpu)
    except Exception:
        return None


_BRAND_WHITELIST = [
    "Dell", "HP", "ASUS", "Acer", "Lenovo", "Apple", "Samsung",
    "Google", "Huawei", "Microsoft", "LG", "Sony", "Toshiba",
    "Evian", "Nestle", "Coca-Cola", "Pepsi", "Heinz",
]

_OCR_DIRECT_CORRECTIONS = {
    "FSLS": "ASUS", "F5LS": "ASUS", "FSUS": "ASUS", "A5U5": "ASUS",
    "4PPLE": "Apple", "APPL3": "Apple", "APPL": "Apple",
    "D3LL": "Dell", "DEIL": "Dell", "OELL": "Dell",
    "G00GLE": "Google", "G0OGLE": "Google", "GOGLE": "Google",
    "SAMSUMG": "Samsung", "SAMDUNG": "Samsung", "SAMSNG": "Samsung",
    "LENEVO": "Lenovo", "LENOVO": "Lenovo", "LEN0VO": "Lenovo",
    "MICR0SOFT": "Microsoft", "MICROSOGT": "Microsoft",
    "HUAW3I": "Huawei", "HUWEI": "Huawei",
    "HP0": "HP", "H0": "HP",
}

def _correct_ocr_brand(text: str) -> str:
    import difflib
    text_upper = text.upper().strip()
    if text_upper in _OCR_DIRECT_CORRECTIONS:
        return _OCR_DIRECT_CORRECTIONS[text_upper]
    for brand in _BRAND_WHITELIST:
        if brand.upper() == text_upper:
            return brand
    candidates = [b.upper() for b in _BRAND_WHITELIST]
    matches = difflib.get_close_matches(text_upper, candidates, n=1, cutoff=0.60)
    if matches:
        idx = candidates.index(matches[0])
        return _BRAND_WHITELIST[idx]
    return text


def count_items_in_photo(yolo_model, clip_model, clip_processor, image, exclude_labels=None):
    exclude_labels = exclude_labels or []
    try:
        import torch
        _imgsz = 1280 if torch.cuda.is_available() else 640
    except Exception:
        _imgsz = 640
    results = yolo_model(image, verbose=False, imgsz=_imgsz, conf=0.3)
    result  = results[0]
    clip_items = ["laptop","cell phone"]
    text_items = ["bottle","book","box"]

    boxes_with_info = []
    for box in result.boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        label = yolo_model.names[cls]
        if label in exclude_labels: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        area = (x2-x1)*(y2-y1)
        img_area = image.shape[0]*image.shape[1]
        if label == "laptop" and area < img_area*0.01: continue
        boxes_with_info.append({'label':label,'conf':conf,'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                                 'center_x':(x1+x2)/2,'center_y':(y1+y2)/2,'area':area})

    boxes_with_info = filter_overlapping_detections(boxes_with_info, iou_threshold=0.4)
    annotated = image.copy()
    boxes_with_info.sort(key=lambda b: (b['center_y'], b['center_x']))

    detections = []
    item_counter = {}

    for idx, bi in enumerate(boxes_with_info, 1):
        base_label = bi['label']; conf = bi['conf']
        x1,y1,x2,y2 = bi['x1'],bi['y1'],bi['x2'],bi['y2']

        brands = []
        display_label = base_label

        if base_label in clip_items or base_label in text_items:
            pad = 40; h,w = image.shape[:2]
            crop = image[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
            if crop.shape[0] >= 20 and crop.shape[1] >= 20:
                if base_label in clip_items:
                    brands = classify_brand_clip(clip_model, clip_processor, crop, base_label)
                else:
                    try:
                        reader = load_ocr_reader()
                        if reader:
                            ocr_res = reader.readtext(crop, paragraph=False)
                            raw_brands = [
                                {"text": _correct_ocr_brand(t.strip()),
                                 "raw_text": t.strip(),
                                 "confidence": c,
                                 "is_brand": False,
                                 "detection_method": "ocr"}
                                for (_, t, c) in ocr_res
                                if c > 0.3 and len(t.strip()) >= 2
                            ]
                            brands = raw_brands
                        else:
                            brands = []
                    except Exception:
                        brands = []

            if brands and base_label in clip_items:
                brand_name = brands[0]["text"]
                if base_label == "laptop":
                    display_label = f"{brand_name} Laptop"
                elif base_label == "cell phone":
                    display_label = f"{brand_name} Phone"
                else:
                    display_label = f"{brand_name} {base_label.title()}"

        color = (0,255,0)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
        label_text = f"{display_label} {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(label_text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(annotated,(x1,y1-th-10),(x1+tw+10,y1),color,-1)
        cv2.putText(annotated,label_text,(x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

        item_counter[base_label] = item_counter.get(base_label,0)+1

        detection = {
            "label":      display_label,
            "base_label": base_label,
            "confidence": conf,
            "brands":     brands,
            "position":   idx,
            "bbox":       (x1,y1,x2,y2),
        }
        detections.append(detection)

    cv2.putText(annotated,f"TOTAL: {len(boxes_with_info)} items",(10,50),
                cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)
    return len(boxes_with_info), annotated, detections


_BASE_CATEGORY_MAP = {
    "laptop":     ["laptop"],
    "cell phone": ["phone", "cell phone"],
    "bottle":     ["bottle"],
    "cup":        ["cup"],
    "book":       ["book"],
    "box":        ["box"],
    "keyboard":   ["keyboard"],
    "mouse":      ["mouse"],
    "backpack":   ["backpack"],
    "chair":      ["chair"],
}

def _fold_to_base(actual_counts: dict) -> dict:
    base: dict = {}
    for label, cnt in actual_counts.items():
        label_l = label.lower()
        matched = False
        for base_key, keywords in _BASE_CATEGORY_MAP.items():
            if any(kw in label_l for kw in keywords):
                base[base_key] = base.get(base_key, 0) + cnt
                matched = True
                break
        if not matched:
            base[label_l] = base.get(label_l, 0) + cnt
    return base


def compare_quantities(actual_counts, do_data):
    if not do_data: return None
    base_counts = _fold_to_base(actual_counts)
    comparison = {"matches":[], "discrepancies":[], "has_discrepancy":False}
    for item_type, expected_qty in (do_data.get("items") or {}).items():
        actual_qty = base_counts.get(item_type, 0)
        diff = actual_qty - expected_qty
        s = {"item_type":item_type,"expected":expected_qty,"actual":actual_qty,
             "difference":diff,"status":"✅ MATCH" if diff==0 else "⚠️ DISCREPANCY"}
        if diff != 0:
            comparison["discrepancies"].append(s); comparison["has_discrepancy"] = True
        else:
            comparison["matches"].append(s)
    for item_type, actual_qty in base_counts.items():
        if item_type not in (do_data.get("items") or {}):
            comparison["discrepancies"].append({"item_type":item_type,"expected":0,
                "actual":actual_qty,"difference":actual_qty,"status":"⚠️ UNEXPECTED ITEM"})
            comparison["has_discrepancy"] = True
    return comparison


def create_excel_report(results_log, do_data=None, comparison=None):
    all_det=[]; clip_items=["laptop","cell phone"]
    for result in results_log:
        iname = result['file']
        for i, d in enumerate(result['detections'], 1):
            it = d['label']; conf = d['confidence']
            base_it = d.get('base_label', it)
            if d['brands']:
                for b in d['brands']:
                    all_det.append({'Image':iname,'Item Number':i,'Item Type':it,
                        'Detection Confidence':f"{conf:.1%}",'Brand/Text':b['text'],
                        'Brand Confidence':f"{b['confidence']:.1%}",
                        'Detection Method':'🤖 CLIP' if b.get('detection_method')=='clip' else '📝 OCR'})
            elif base_it in clip_items:
                all_det.append({'Image':iname,'Item Number':i,'Item Type':it,
                    'Detection Confidence':f"{conf:.1%}",'Brand/Text':'N/A',
                    'Brand Confidence':'','Detection Method':''})

    item_counts={}; item_brands={}
    for r in results_log:
        for d in r['detections']:
            it = d['label']
            item_counts[it] = item_counts.get(it,0)+1
            if d['brands']:
                item_brands.setdefault(it,{})
                for b in d['brands']:
                    item_brands[it][b['text']] = item_brands[it].get(b['text'],0)+1

    summary_data = [{'Item Type':k.capitalize(),'Total Count':v} for k,v in sorted(item_counts.items())]
    brand_summary = []
    for it, brands in sorted(item_brands.items()):
        total = item_counts.get(it,0)
        for brand, cnt in sorted(brands.items()):
            brand_summary.append({'Item Type':it.capitalize(),'Brand':brand,
                'Count':cnt,'Percentage':f"{cnt/total*100:.1f}%" if total else "0%"})

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if comparison and do_data:
            cdata = [
                {'Item Type':'PDF DO INFORMATION','Expected Quantity':'','Actual Quantity':'','Difference':'','Status':''},
                {'Item Type':f"DO Number: {do_data['metadata'].get('do_number','N/A')}",
                 'Expected Quantity':'','Actual Quantity':'','Difference':'','Status':''},
                {'Item Type':f"Supplier: {do_data['metadata'].get('supplier','N/A')}",
                 'Expected Quantity':'','Actual Quantity':'','Difference':'','Status':''},
                {'Item Type':'','Expected Quantity':'','Actual Quantity':'','Difference':'','Status':''},
            ]
            for m in comparison["matches"]:
                cdata.append({'Item Type':m["item_type"].upper(),'Expected Quantity':m["expected"],
                    'Actual Quantity':m["actual"],'Difference':m["difference"],'Status':m["status"]})
            for disc in comparison["discrepancies"]:
                cdata.append({'Item Type':disc["item_type"].upper(),'Expected Quantity':disc["expected"],
                    'Actual Quantity':disc["actual"],'Difference':f"{disc['difference']:+d}",'Status':disc["status"]})
            pd.DataFrame(cdata).to_excel(writer, sheet_name='PDF DO Comparison', index=False)
        if all_det:
            pd.DataFrame(all_det).to_excel(writer, sheet_name='All Detections', index=False)
        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Item Summary', index=False)
        if brand_summary:
            pd.DataFrame(brand_summary).to_excel(writer, sheet_name='Brand Summary', index=False)
    output.seek(0)
    return output


def generate_asset_id():
    return f"AST-{datetime.now().strftime('%Y')}-{str(uuid.uuid4()).upper()[:8]}"


def generate_sticker_png(asset_id, image_name, detection, item_num, date_str, cropped_img=None):
    try:
        import barcode
        from barcode.writer import ImageWriter
    except ImportError:
        return None
    W, H = 600, 320
    sticker = PILImage.new("RGB",(W,H),(255,255,255))
    draw = ImageDraw.Draw(sticker)
    primary=(30,30,80); accent=(0,180,120); light_gray=(240,240,245)
    draw.rectangle([(0,0),(W,60)],fill=primary)
    try:
        fl  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",22)
        fm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
        fs  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",12)
        fb  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
    except:
        fl=fm=fs=fb=ImageFont.load_default()
    draw.text((20,10),"📦 ASSET STICKER",font=fl,fill=(255,255,255))
    draw.text((W-160,20),date_str,font=fs,fill=(200,200,220))
    draw.rectangle([(15,75),(W-15,115)],fill=light_gray,outline=primary,width=2)
    draw.text((25,82),"ASSET ID:",font=fb,fill=primary)
    draw.text((120,82),asset_id,font=fb,fill=accent)
    item_label = detection['label'].title()
    brand_name = detection['brands'][0]['text'] if detection['brands'] else "Unknown"
    conf = detection['confidence']
    draw.text((25,125),f"Source: {image_name}  •  Item #{item_num}",font=fs,fill=(100,100,100))
    draw.text((25,148),"Item Details:",font=fb,fill=primary)
    y=170
    draw.text((25,y),    f"  • Type:       {item_label}",font=fm,fill=(50,50,50))
    draw.text((25,y+24), f"  • Brand:      {brand_name}",font=fm,fill=(50,50,50))
    draw.text((25,y+48), f"  • Confidence: {conf:.0%}",font=fm,fill=(50,50,50))
    if cropped_img is not None:
        try:
            rgb = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
            thumb = PILImage.fromarray(rgb).resize((120,90),PILImage.LANCZOS)
            sticker.paste(thumb,(W-145,70))
            draw.rectangle([(W-146,69),(W-24,161)],outline=primary,width=2)
        except:
            pass
    try:
        bio = BytesIO()
        code128 = barcode.get('code128',asset_id,writer=ImageWriter())
        code128.write(bio,options={"module_width":0.8,"module_height":8.0,"font_size":6,
            "text_distance":2,"quiet_zone":2,"write_text":True,"background":"white","foreground":"black"})
        bio.seek(0)
        bc_img = PILImage.open(bio).convert("RGB").resize((200,80),PILImage.LANCZOS)
        sticker.paste(bc_img,(W-215,H-95))
    except:
        draw.text((W-210,H-40),asset_id,font=fs,fill=primary)
    draw.rectangle([(0,H-6),(W,H)],fill=accent)
    out = BytesIO(); sticker.save(out,format="PNG",dpi=(300,300)); out.seek(0)
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — SUBSYSTEM 2: AUTO-PICKER
# ═══════════════════════════════════════════════════════════════════════

WAREHOUSE_CONFIG = {
    "grid_size":(6,24),"aisle_width":3.0,"rack_depth":1.2,
    "max_order_quantity":1000,"max_order_items":50,
    "min_customer_id_length":3,"max_customer_id_length":100,
    "low_stock_threshold":10,"walking_speed":80,
    "picking_time_per_item":0.5,"demo_item_cap":5,
}
VALID_CUSTOMER_TYPES = {"retail","wholesale","repair_shop"}
VALID_CATEGORIES = {"computers","phones","accessories","components"}


class OrderStatus(Enum):
    PENDING="pending"; PROCESSING="processing"; PICKING="picking"
    PARTIAL="partially_fulfilled"; COMPLETE="complete"
    FAILED="failed"; CANCELLED="cancelled"


@dataclass
class GridLocation:
    aisle: int; rack: int; x: float=0.0; y: float=0.0
    def __post_init__(self):
        if not (1<=self.aisle<=WAREHOUSE_CONFIG["grid_size"][0]):
            raise ValueError(f"Aisle must be 1-{WAREHOUSE_CONFIG['grid_size'][0]}")
        if not (1<=self.rack<=WAREHOUSE_CONFIG["grid_size"][1]):
            raise ValueError(f"Rack must be 1-{WAREHOUSE_CONFIG['grid_size'][1]}")
        self.x=(self.aisle-1)*WAREHOUSE_CONFIG["aisle_width"]
        self.y=(self.rack-1)*WAREHOUSE_CONFIG["rack_depth"]
    def __str__(self): return f"A{self.aisle:02d}-R{self.rack:03d}"
    def distance_to(self, other):
        if self.aisle==other.aisle: return abs(self.y-other.y)
        ae=min(self.y,WAREHOUSE_CONFIG["rack_depth"]*12-self.y)
        oe=min(other.y,WAREHOUSE_CONFIG["rack_depth"]*12-other.y)
        return ae+abs(self.aisle-other.aisle)*WAREHOUSE_CONFIG["aisle_width"]+oe+abs(self.y-other.y)


@dataclass
class ElectronicsItem:
    sku:str; name:str; category:str; location:GridLocation
    quantity:int; weight:float; value:float
    dimensions:Tuple[int,int,int]; fragile:bool=False
    def __post_init__(self):
        if not self.sku.startswith("ELEC-"): raise ValueError("SKU must start with 'ELEC-'")
        if self.quantity < 0: raise ValueError("Quantity cannot be negative")
        if self.category not in VALID_CATEGORIES: raise ValueError("Invalid category")


@dataclass
class OrderItemPicker:
    sku:str; quantity:int; picked:int=0; status:str="pending"


@dataclass
class CustomerOrder:
    order_id:str; customer_id:str; customer_type:str
    items:List[OrderItemPicker]; status:OrderStatus
    created_at:datetime; priority:int=3
    completed_at:Optional[datetime]=None
    def fulfillment_rate(self):
        tr=sum(i.quantity for i in self.items)
        tp=sum(i.picked for i in self.items)
        return (tp/tr*100) if tr>0 else 0.0


class GridRouteOptimizer:
    def __init__(self):
        self.start_location=GridLocation(aisle=1,rack=1)
    def generate_grid_route(self, items):
        if not items: return []
        groups={}
        for item in items:
            groups.setdefault(item.location.aisle,[]).append(item)
        route=[]
        for aisle in sorted(groups):
            route.extend(item.location for item in sorted(groups[aisle],key=lambda x:x.location.rack))
        return route
    def calculate_route_metrics(self, route):
        if not route: return {"total_distance_m":0,"total_time_min":0,"items_count":0}
        dist=0; cur=self.start_location
        for loc in route:
            dist+=cur.distance_to(loc); cur=loc
        dist+=cur.distance_to(self.start_location)
        tt=dist/WAREHOUSE_CONFIG["walking_speed"]
        pt=len(route)*WAREHOUSE_CONFIG["picking_time_per_item"]
        return {"total_distance_m":round(dist,2),"travel_time_min":round(tt,2),
                "picking_time_min":round(pt,2),"total_time_min":round(tt+pt,2),"items_count":len(route)}


class StreamlitAutoPicker:
    def __init__(self):
        self.orders: Dict[str,CustomerOrder]={}
        self.inventory: Dict[str,ElectronicsItem]={}
        self.optimizer=GridRouteOptimizer()
        self.order_counter = self._next_order_counter()
        self.metrics={"start_time":datetime.now().isoformat(),"total_orders":0,
                      "total_items_picked":0,"successful_picks":0,"failed_picks":0}
        self.brand_sku_registry: Dict[str,str] = {}
        self._init_inventory()

    def _init_inventory(self):
        from database import get_db as _get_db
        try:
            db_rows = {r["sku"]: r for r in _get_db().get_inventory()}
        except Exception:
            db_rows = {}

        n = 1
        for aisle in range(1, WAREHOUSE_CONFIG["grid_size"][0]+1):
            for rack in range(1, WAREHOUSE_CONFIG["grid_size"][1]+1):
                try:
                    if aisle <= 2:
                        cat="computers"; name=f"Laptop {n}"; val=1200.0; frag=True; qty=10+(rack%5)
                    elif aisle <= 4:
                        cat="phones"; name=f"Smartphone {n}"; val=800.0; frag=True; qty=25+(rack%10)
                    elif rack <= 12:
                        cat="accessories"; name=f"Accessory {n}"; val=50.0; frag=False; qty=100+(rack%20)
                    else:
                        cat="components"; name=f"Component {n}"; val=150.0; frag=True; qty=50+(rack%15)

                    sku = f"ELEC-{n:04d}"

                    if sku in db_rows:
                        row  = db_rows[sku]
                        qty  = row["quantity"]
                        name = row["name"] or name
                        frag = bool(row.get("fragile", frag))

                    self.inventory[sku] = ElectronicsItem(
                        sku=sku, name=name, category=cat,
                        location=GridLocation(aisle=aisle, rack=rack),
                        quantity=qty, weight=0.5, value=val,
                        dimensions=(10, 10, 5), fragile=frag)
                    n += 1
                except Exception:
                    n += 1
                    continue

        try:
            _get_db().seed_inventory([
                {
                    "sku":      sku,
                    "name":     item.name,
                    "category": item.category,
                    "location": {"aisle": item.location.aisle, "rack": item.location.rack},
                    "quantity": item.quantity,
                    "weight":   item.weight,
                    "value":    item.value,
                    "dimensions": item.dimensions,
                    "fragile":  item.fragile,
                }
                for sku, item in self.inventory.items()
            ])
            for sku, item in self.inventory.items():
                if item.category == "computers" and not item.name.startswith("Laptop "):
                    self.brand_sku_registry[item.name] = sku
                elif item.category == "phones" and not item.name.startswith("Smartphone "):
                    self.brand_sku_registry[item.name] = sku
        except Exception:
            pass

    def _next_order_counter(self) -> int:
        try:
            from database import get_db as _get_db
            orders = _get_db().get_picker_orders(limit=9999)
            if not orders:
                return 1
            max_num = 0
            for o in orders:
                oid = o.get("order_id", "")
                if oid.startswith("ORD-"):
                    try:
                        max_num = max(max_num, int(oid[4:]))
                    except ValueError:
                        pass
            return max_num + 1
        except Exception:
            return 1

    def _validate_inventory(self):
        for item in self.inventory.values():
            if item.quantity<0: item.quantity=0

    def register_brand_sku(self, display_label: str, base_label: str) -> str:
        if display_label in self.brand_sku_registry:
            return self.brand_sku_registry[display_label]

        cat_map = {
            "laptop":     "computers",
            "cell phone": "phones",
        }
        target_cat = cat_map.get(base_label.lower(), "")
        if not target_cat:
            return ""

        already_taken = set(self.brand_sku_registry.values())
        candidates = [
            sku for sku, item in sorted(self.inventory.items())
            if item.category == target_cat and sku not in already_taken
        ]
        if not candidates:
            return ""

        chosen_sku = candidates[0]
        self.brand_sku_registry[display_label] = chosen_sku
        self.inventory[chosen_sku].name = display_label
        self.inventory[chosen_sku].fragile = True

        try:
            from database import get_db
            get_db().update_inventory_name(chosen_sku, display_label)
        except Exception:
            pass

        return chosen_sku

    def create_order(self, customer_id, customer_type, items):
        try:
            if not customer_id or len(customer_id.strip())<3:
                return {"success":False,"error":"Customer ID must be at least 3 characters"}
            if customer_type not in VALID_CUSTOMER_TYPES:
                return {"success":False,"error":f"Invalid customer type"}
            if not items:
                return {"success":False,"error":"Order must contain at least one item"}
            cid = customer_id.strip()
            active_statuses = {OrderStatus.PENDING, OrderStatus.PROCESSING, OrderStatus.PICKING}
            duplicate = next(
                (o for o in self.orders.values()
                 if o.customer_id == cid and o.status in active_statuses),
                None
            )
            if duplicate:
                return {
                    "success": False,
                    "error": (f"Customer '{cid}' already has an active order "
                              f"({duplicate.order_id} — {duplicate.status.value}). "
                              f"Complete or cancel it first.")
                }
            order_items=[]; warnings=[]
            for sku,qty in items:
                if sku not in self.inventory: warnings.append(f"SKU {sku} not found"); continue
                if not isinstance(qty,int) or qty<=0: warnings.append(f"Bad qty for {sku}"); continue
                if qty>WAREHOUSE_CONFIG["max_order_quantity"]:
                    qty=WAREHOUSE_CONFIG["max_order_quantity"]; warnings.append(f"Capped qty for {sku}")
                avail=self.inventory[sku].quantity
                if qty>avail: qty=avail; warnings.append(f"Partial stock for {sku}")
                if qty>0: order_items.append(OrderItemPicker(sku=sku,quantity=qty))
            if not order_items: return {"success":False,"error":"No valid items"}
            order_id=f"ORD-{self.order_counter:06d}"; self.order_counter+=1
            priority={"wholesale":1,"repair_shop":2,"retail":3}.get(customer_type,3)
            order=CustomerOrder(order_id=order_id,customer_id=customer_id.strip(),
                customer_type=customer_type,items=order_items,
                status=OrderStatus.PENDING,created_at=datetime.now(),priority=priority)
            self.orders[order_id]=order; self.metrics["total_orders"]+=1
            return {"success":True,"order_id":order_id,"warnings":warnings or None,
                    "message":f"Order {order_id} created"}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def process_order(self, order_id):
        try:
            if order_id not in self.orders: return {"success":False,"error":"Order not found"}
            order=self.orders[order_id]
            if order.status not in [OrderStatus.PENDING,OrderStatus.PROCESSING]:
                return {"success":False,"error":f"Cannot process: {order.status.value}"}
            order.status=OrderStatus.PROCESSING
            route_items=[]; unavailable=[]
            for item in order.items:
                if item.sku in self.inventory:
                    inv=self.inventory[item.sku]
                    for _ in range(min(item.quantity,5)):
                        route_items.append(inv)
                else: unavailable.append(item.sku)
            if not route_items:
                order.status=OrderStatus.FAILED
                return {"success":False,"error":"No items available"}
            route=self.optimizer.generate_grid_route(route_items)
            metrics=self.optimizer.calculate_route_metrics(route)
            return {"success":True,"order_id":order_id,"route":[str(l) for l in route],
                    "metrics":metrics,"unavailable":unavailable or None,
                    "order_value":self._calculate_order_value(order)}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def get_route_preview(self, order_id: str) -> dict:
        try:
            if order_id not in self.orders:
                return {"success": False, "error": "Order not found"}
            order = self.orders[order_id]
            route_items = []
            for item in order.items:
                if item.sku in self.inventory:
                    inv = self.inventory[item.sku]
                    for _ in range(min(item.quantity, 5)):
                        route_items.append(inv)
            if not route_items:
                return {"success": False, "error": "No route items"}
            route   = self.optimizer.generate_grid_route(route_items)
            metrics = self.optimizer.calculate_route_metrics(route)
            return {
                "success":     True,
                "order_id":    order_id,
                "route":       [str(l) for l in route],
                "metrics":     metrics,
                "order_value": self._calculate_order_value(order),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_picking(self, order_id):
        try:
            if order_id not in self.orders: return {"success":False,"error":"Order not found"}
            order=self.orders[order_id]
            if order.status not in [OrderStatus.PROCESSING,OrderStatus.PENDING]:
                return {"success":False,"error":f"Cannot pick: {order.status.value}"}
            order.status=OrderStatus.PICKING; picked=[]; failed=[]
            for item in order.items:
                if item.sku not in self.inventory:
                    failed.append({"sku":item.sku,"reason":"not_found"})
                    self.metrics["failed_picks"]+=1; continue
                inv=self.inventory[item.sku]
                if inv.quantity>=item.quantity:
                    inv.quantity-=item.quantity; item.picked=item.quantity; item.status="picked"
                    picked.append({"sku":item.sku,"quantity":item.quantity,
                        "location":str(inv.location),"value":inv.value*item.quantity})
                    self.metrics["successful_picks"]+=1
                    self.metrics["total_items_picked"]+=item.quantity
                elif inv.quantity>0:
                    pq=inv.quantity; inv.quantity=0; item.picked=pq; item.status="partial"
                    picked.append({"sku":item.sku,"quantity":pq,"location":str(inv.location),
                        "value":inv.value*pq,"partial":True})
                    failed.append({"sku":item.sku,"reason":"insufficient_stock",
                        "requested":item.quantity,"picked":pq})
                    self.metrics["total_items_picked"]+=pq
                else:
                    failed.append({"sku":item.sku,"reason":"out_of_stock"})
                    self.metrics["failed_picks"]+=1
            ful=order.fulfillment_rate()
            if ful==100: order.status=OrderStatus.COMPLETE
            elif ful>0: order.status=OrderStatus.PARTIAL
            else: order.status=OrderStatus.FAILED
            order.completed_at=datetime.now()
            return {"success":True,"order_id":order_id,
                    "status":order.status.value,"fulfillment":ful,
                    "picked_count":len(picked),"failed_count":len(failed),
                    "picked_value":sum(p.get('value',0) for p in picked),
                    "failed_items":failed or None}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def _calculate_order_value(self, order):
        return round(sum(self.inventory[i.sku].value*i.quantity
                         for i in order.items if i.sku in self.inventory),2)

    def get_inventory_dataframe(self):
        return pd.DataFrame([{
            "SKU":sku,"Name":item.name,"Category":item.category,
            "Location":str(item.location),"Aisle":item.location.aisle,
            "Rack":item.location.rack,"Quantity":item.quantity,
            "Value ($)":item.value,"Total Value ($)":item.quantity*item.value,
            "Fragile":"Yes" if item.fragile else "No"}
            for sku,item in self.inventory.items()])

    def get_orders_dataframe(self):
        rows=[]
        for oid,o in self.orders.items():
            rows.append({"Order ID":oid,"Customer":o.customer_id,"Type":o.customer_type,
                "Status":o.status.value,"Items":len(o.items),
                "Fulfillment":f"{o.fulfillment_rate():.1f}%","Priority":o.priority,
                "Created":o.created_at.strftime("%Y-%m-%d %H:%M"),
                "Completed":o.completed_at.strftime("%Y-%m-%d %H:%M") if o.completed_at else "-"})
        return pd.DataFrame(rows)

    def get_grid_heatmap(self):
        grid=np.zeros(WAREHOUSE_CONFIG["grid_size"])
        for item in self.inventory.values():
            try: grid[item.location.aisle-1,item.location.rack-1]=item.quantity
            except: pass
        return grid

    def get_system_stats(self):
        total_items=sum(i.quantity for i in self.inventory.values())
        total_value=sum(i.quantity*i.value for i in self.inventory.values())
        low_stock=sum(1 for i in self.inventory.values()
                      if i.quantity<WAREHOUSE_CONFIG["low_stock_threshold"])
        return {
            "total_skus":len(self.inventory),"total_items":total_items,
            "total_value":total_value,"low_stock_count":low_stock,
            "total_orders":len(self.orders),
            "pending_orders":sum(1 for o in self.orders.values() if o.status==OrderStatus.PENDING),
            "completed_orders":sum(1 for o in self.orders.values() if o.status==OrderStatus.COMPLETE),
            "failed_orders":sum(1 for o in self.orders.values() if o.status==OrderStatus.FAILED),
            "success_rate":(self.metrics["successful_picks"]/
                            max(self.metrics["successful_picks"]+self.metrics["failed_picks"],1))*100,
        }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — SUBSYSTEM 3: COURIER EMAIL
# ═══════════════════════════════════════════════════════════════════════

COURIER = "FedEx"
SHIPPING_METHODS = ["Same-Day Express", "Express", "Standard"]
WAREHOUSE_INFO = {
    "name":"WIS Warehouse","address":"1 Warehouse Road, Singapore 123456",
    "contact":"+65 6000 0000","operating_hours":"Mon-Fri 08:00-18:00",
    "dock_number":"Dock A3",
}

def _stub_ai_select_shipping(weight_kg, volume_cm3, distance_km, priority):
    if priority == "urgent" or distance_km < 100:
        return "Same-Day Express", 0.92
    if weight_kg > 10 or distance_km > 3000:
        return "Standard", 0.85
    return "Express", 0.88


def generate_courier_email(order: PickerOrder, shipping_method: str,
                           is_fragile: bool, confidence: float) -> Dict:
    items_text = "\n".join(
        f"  - {i.get('name') or i.get('sku','Unknown')}  "
        f"x{i.get('quantity',0)}  (SKU: {i.get('sku','N/A')})"
        for i in order.items
    ) if order.items else "  (no items on record)"
    fragile_note = "⚠️ FRAGILE — Handle with care\n" if is_fragile else ""
    body = f"""Dear {COURIER} Operations Team,

Please arrange a pick-up for the following shipment:

ORDER DETAILS
─────────────────────────────────
Order ID      : {order.order_id}
Customer      : {order.customer_id}  ({order.customer_type})
Order Value   : SGD {order.order_value:,.2f}
Shipping      : {shipping_method}  (ML confidence {confidence:.0%})
{fragile_note}
ITEMS TO COLLECT
─────────────────────────────────
{items_text}

PICK-UP DETAILS
─────────────────────────────────
Warehouse     : {WAREHOUSE_INFO['name']}
Address       : {WAREHOUSE_INFO['address']}
Dock          : {WAREHOUSE_INFO['dock_number']}
Contact       : {WAREHOUSE_INFO['contact']}
Operating Hrs : {WAREHOUSE_INFO['operating_hours']}

Please confirm this pick-up request by replying to this email.

Kind regards,
WIS Logistics Team
Warehouse Intelligence System
"""
    subject = f"[{shipping_method}] Pick-up Request — {order.order_id} — {COURIER}"
    email_to = "operations@fedex.com"
    return {"subject": subject, "body": body, "to": email_to}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════

def init_session():
    store = get_store()

    if "auto_picker" not in st.session_state:
        st.session_state.auto_picker = StreamlitAutoPicker()

    store.inventory = st.session_state.auto_picker.inventory

    if "picker_current_order" not in st.session_state:
        st.session_state.picker_current_order = None
    if "picker_errors" not in st.session_state:
        st.session_state.picker_errors = []
    if "picker_success" not in st.session_state:
        st.session_state.picker_success = []

    # SS3 state
    if "courier_results" not in st.session_state:
        st.session_state.courier_results = []
    if "courier_last_email" not in st.session_state:
        st.session_state.courier_last_email = None
    if "pending_shipment_preview" not in st.session_state:
        st.session_state.pending_shipment_preview = None

    # ── NEW: persists pick result across reruns so it doesn't vanish ──
    if "last_pick_result" not in st.session_state:
        st.session_state.last_pick_result = None

    return store


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

def render_sidebar(store: SharedStore):
    with st.sidebar:
        st.markdown("## 🏭 WIS")
        st.caption("Warehouse Intelligence System")
        st.divider()

        st.markdown("### 🔄 Pipeline Progress")
        summary = store.pipeline_summary()

        steps = [
            ("📷 Scan",   summary["scans"] > 0),
            ("🤖 Pick",   summary["completed_picks"] > 0),
            ("📧 Ship",   summary["confirmed_shipments"] > 0),
        ]
        for label, done in steps:
            badge = "badge-done" if done else "badge-wait"
            icon  = "✅" if done else "⏳"
            st.markdown(f'<span class="pipeline-badge {badge}">{icon} {label}</span>',
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 Live Counters")
        col1, col2 = st.columns(2)
        col1.metric("Scans",    summary["scans"])
        col2.metric("Detected", summary["items_detected"])
        col1.metric("Orders",   summary["picker_orders"])
        col2.metric("Picks",    summary["completed_picks"])
        col1.metric("Shipments",summary["shipments"])
        col2.metric("Confirmed",summary["confirmed_shipments"])

        st.divider()
        if st.button("🗑️ Reset Everything", use_container_width=True, type="secondary"):
            from database import reset_db
            reset_db()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — TAB: PIPELINE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

def render_pipeline_tab(store: SharedStore):
    st.markdown("## 🔄 End-to-End Pipeline")
    st.caption("Live status of all three subsystems sharing the same dataset.")

    summary = store.pipeline_summary()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔍 Scan Results",   summary["scans"])
    c2.metric("📦 Items Detected", summary["items_detected"])
    c3.metric("✅ Picks Complete", summary["completed_picks"])
    c4.metric("📧 Emails Sent",    summary["confirmed_shipments"])

    st.markdown("---")

    st.markdown("""
    ```
    ┌─────────────────────┐
    │  📷 Object Scanner  │  ← Upload images + optional PDF DO
    │  (Subsystem 1)      │    YOLO detection · CLIP brand ID
    └──────────┬──────────┘
               │ actual_counts, scan_results
               ▼
    ┌─────────────────────┐
    │  🤖 Auto-Picker     │  ← Maps scanned items → inventory SKUs
    │  (Subsystem 2)      │    Route optimisation · picking execution
    └──────────┬──────────┘
               │ completed PickerOrders
               ▼
    ┌─────────────────────┐
    │  📧 Courier Email   │  ← Selects shipping method (ML)
    │  (Subsystem 3)      │    Generates FedEx pick-up email
    └─────────────────────┘
    ```
    """)

    st.markdown("---")

    st.subheader("🔗 Scan → Picker Bridge")
    if store.actual_counts:
        st.markdown("Items detected by scanner (ready to order from picker):")
        bridge_data = []
        ap_for_bridge: StreamlitAutoPicker = st.session_state.auto_picker
        for item_type, count in store.actual_counts.items():
            reg_sku = ap_for_bridge.brand_sku_registry.get(item_type)
            if reg_sku:
                matching = [reg_sku]
            else:
                matching = [sku for sku, inv in store.inventory.items()
                            if item_type.lower() in inv.name.lower() or
                               item_type.lower() in inv.category.lower()]
            bridge_data.append({
                "Scanned Item": item_type.title(),
                "Count": count,
                "Matched SKU": matching[0] if matching else "N/A",
                "Stock": (store.inventory[matching[0]].quantity
                          if matching and matching[0] in store.inventory else 0),
            })
        st.dataframe(pd.DataFrame(bridge_data), hide_index=True, use_container_width=True)

        if st.button("📦 Auto-Create Picker Order from Scan", type="primary"):
            _auto_create_order_from_scan(store)
    else:
        st.info("Run the Object Scanner first to populate item counts.")

    st.markdown("---")

    st.subheader("🔗 Picker → Email Bridge")
    unshipped = store.get_unshipped_orders()
    if unshipped:
        st.success(f"{len(unshipped)} completed order(s) ready to ship.")
        for o in unshipped:
            with st.expander(f"📦 {o.order_id} — {o.customer_id}"):
                col1, col2 = st.columns(2)
                col1.metric("Items", len(o.items))
                col2.metric("Value", f"${o.order_value:,.2f}")
                if st.button(f"📧 Generate Email for {o.order_id}", key=f"gen_{o.order_id}"):
                    _generate_and_store_email(store, o)
                    st.rerun()
    else:
        st.info("Complete a picker order to enable automatic email generation.")

    if store.shipments:
        st.markdown("---")
        st.subheader("📧 Recent Shipment Emails")
        rows = [{"Shipment":sid,"Order":s.order_id,"Method":s.shipping_method,
                 "Status":s.status,"Fragile":"⚠️" if s.is_fragile else "✓"}
                for sid,s in store.shipments.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _auto_create_order_from_scan(store: SharedStore):
    ap: StreamlitAutoPicker = st.session_state.auto_picker
    items_to_order = []
    for item_type, count in store.actual_counts.items():
        sku = ap.brand_sku_registry.get(item_type)
        if sku and sku in ap.inventory:
            items_to_order.append((sku, count))
            continue
        for sku, inv in ap.inventory.items():
            if (item_type.lower() in inv.name.lower() or
                    item_type.lower() in inv.category.lower()):
                items_to_order.append((sku, count))
                break

    if not items_to_order:
        st.warning("No matching SKUs found for scanned items.")
        return

    result = ap.create_order("SCAN-AUTO", "retail", items_to_order)
    if result["success"]:
        oid = result["order_id"]
        ap.process_order(oid)
        pick_result = ap.execute_picking(oid)
        order = ap.orders[oid]

        po = PickerOrder(
            order_id=oid, customer_id="SCAN-AUTO", customer_type="retail",
            status=order.status.value,
            items=[{"sku":i.sku, "name":ap.inventory[i.sku].name if i.sku in ap.inventory else i.sku,
                    "quantity":i.quantity, "picked":i.picked,
                    "value":ap.inventory[i.sku].value*i.quantity if i.sku in ap.inventory else 0}
                   for i in order.items],
            route=[], route_metrics={},
            fulfillment=order.fulfillment_rate(),
            order_value=ap._calculate_order_value(order),
        )
        store.push_picker_order(po)
        ful_rate = order.fulfillment_rate() or 0.0
        st.success(f"✅ Auto-order {oid} created & picked ({ful_rate:.0f}% fulfilled).")
    else:
        st.error(f"Could not create auto-order: {result['error']}")


def _generate_and_store_email(store: SharedStore, order: PickerOrder):
    weight = sum(i.get("quantity",1)*0.5 for i in order.items)
    try:
        from ai_decision_engine import AIDecisionEngine
        engine = AIDecisionEngine(); engine.train()
        method, conf = engine.predict_shipping(weight_kg=weight, volume_cm3=5000,
                                                distance_km=500, priority="standard")
    except Exception:
        method, conf = _stub_ai_select_shipping(weight, 5000, 500, "standard")

    fragile = any("laptop" in i.get("name","").lower() or "phone" in i.get("name","").lower()
                  for i in order.items)
    email = generate_courier_email(order, method, fragile, conf)

    sid = f"SHP-{uuid.uuid4().hex[:8].upper()}"
    shipment = ShipmentRecord(
        shipment_id=sid, order_id=order.order_id, status="confirmed",
        shipping_method=method, courier=COURIER, is_fragile=fragile,
        email_to=email["to"], email_subject=email["subject"], email_body=email["body"],
    )
    store.push_shipment(shipment)
    st.session_state.courier_last_email = email
    st.success(f"📧 Email generated ({method}) and stored as {sid}.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — TAB: OBJECT SCANNER
# ═══════════════════════════════════════════════════════════════════════

def render_scanner_tab(store: SharedStore):
    st.markdown("## 📷 Object Scanner — Subsystem 1")

    yolo_model, clip_model, clip_processor = load_models()
    if yolo_model is None:
        st.error("Failed to load AI models.")
        return

    with st.expander("⚙️ Upload & Settings", expanded=True):
        col_left, col_right = st.columns([1,1])
        with col_left:
            pdf_file   = st.file_uploader("PDF Delivery Order (optional)", type=['pdf'])
            image_files = st.file_uploader("Images (required)", type=['jpg','jpeg','png'],
                                           accept_multiple_files=True)
        with col_right:
            all_possible = ['chair','couch','dining table','tv','potted plant','clock',
                            'vase','scissors','teddy bear','hair drier','toothbrush','bench']
            exclude_items = st.multiselect("Ignore items:", options=all_possible, default=['chair'])
            process_btn = st.button("🚀 Start Processing", type="primary", use_container_width=True)

    if not process_btn:
        return
    if not image_files:
        st.error("Please upload at least one image!")
        return

    do_data = None
    if pdf_file:
        with st.spinner("📄 Reading PDF…"):
            do_data = extract_pdf_delivery_order(pdf_file)
        if do_data:
            st.success("✅ PDF DO extracted!")
            c1,c2 = st.columns(2)
            c1.info(f"**DO Number:** {do_data['metadata']['do_number']}")
            c2.info(f"**Supplier:** {do_data['metadata']['supplier']}")
        else:
            st.warning("⚠️ Could not extract PDF data — count-only mode.")

    store.clear_scans()
    store.do_data = do_data
    results_log = []; actual_counts = {}
    progress_bar = st.progress(0); status_text = st.empty()

    session_id = store.start_scan_session(
        total_images=len(image_files),
        notes=f"Uploaded: {', '.join(f.name for f in image_files)}"
    )

    do_id = None
    if do_data:
        do_id = store.push_delivery_order(do_data, session_id)

    for idx, image_file in enumerate(image_files):
        status_text.text(f"Processing {image_file.name}… ({idx+1}/{len(image_files)})")
        image_file.seek(0)
        raw_bytes = image_file.read()
        file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.warning(f"⚠️ Could not decode {image_file.name} — skipping.")
            progress_bar.progress((idx+1)/len(image_files))
            continue

        count, annotated, detections = count_items_in_photo(
            yolo_model, clip_model, clip_processor, image, exclude_labels=exclude_items)
        for det in detections:
            actual_counts[det['label']] = actual_counts.get(det['label'], 0) + 1

        scan = ScanResult(
            file_name=image_file.name, count=count,
            session_id=session_id,
            detections=[ScanDetection(label=d['label'], confidence=d['confidence'],
                brands=d['brands'], position=d['position'], bbox=d['bbox'],
                ) for d in detections],
            annotated_image=annotated)
        store.push_scan(scan)

        ap: StreamlitAutoPicker = st.session_state.auto_picker
        for det in detections:
            lbl  = det['label']
            base = det.get('base_label', lbl)
            if lbl != base and det.get('brands'):
                top_brand = det['brands'][0]
                if top_brand.get('detection_method') == 'clip':
                    ap.register_brand_sku(lbl, base)

        results_log.append({"file":image_file.name,"count":count,
                             "detections":detections,"annotated":annotated})
        progress_bar.progress((idx+1)/len(image_files))

    status_text.text("✅ Processing complete!")

    ap: StreamlitAutoPicker = st.session_state.auto_picker
    db = get_db()
    inv_updates = []

    YOLO_TO_CAT = {
        "laptop":     "computers",
        "cell phone": "phones",
        "bottle":     "accessories",
        "book":       "accessories",
        "backpack":   "accessories",
        "keyboard":   "components",
        "mouse":      "components",
        "box":        "components",
        "cup":        "accessories",
    }

    for det in [d for result in results_log for d in result['detections']]:
        lbl  = det['label']
        base = det.get('base_label', lbl)
        cat  = YOLO_TO_CAT.get(base.lower(), base.lower())

        sku = ap.brand_sku_registry.get(lbl)
        if not sku:
            taken_skus = set(ap.brand_sku_registry.values())
            for s, inv_item in sorted(ap.inventory.items()):
                if s in taken_skus:
                    continue
                if (cat in inv_item.category.lower() or
                        base.lower() in inv_item.name.lower()):
                    sku = s
                    break
            if not sku:
                for s, inv_item in sorted(ap.inventory.items()):
                    if (cat in inv_item.category.lower() or
                            base.lower() in inv_item.name.lower()):
                        sku = s
                        break

        if sku and sku in ap.inventory:
            inv_item = ap.inventory[sku]
            before = inv_item.quantity
            inv_item.quantity += 1
            db.update_inventory_quantity(sku, inv_item.quantity)
            existing = next((u for u in inv_updates if u['sku'] == sku), None)
            if existing:
                existing['added']  += 1
                existing['after']   = inv_item.quantity
            else:
                inv_updates.append({
                    'sku':    sku,
                    'name':   inv_item.name,
                    'before': before,
                    'added':  1,
                    'after':  inv_item.quantity,
                })

    total_written = sum(store.actual_counts.values())
    unique_types  = len(store.actual_counts)
    st.success(
        f"✅ Scan complete — **{total_written} item(s)** detected "
        f"({unique_types} type{'s' if unique_types != 1 else ''})"
    )
    if inv_updates:
        st.info(f"📦 **{len(inv_updates)} SKU(s) restocked** based on scan:")
        upd_df = pd.DataFrame(inv_updates)[['sku','name','before','added','after']]
        upd_df.columns = ['SKU','Item','Before','Added','New Stock']
        st.dataframe(upd_df, hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ No matching SKUs found — inventory not updated.")

    st.markdown("---")

    st.header("📊 Detection Results")
    for result in results_log:
        with st.expander(f"📸 {result['file']} — {result['count']} items", expanded=True):
            c1,c2 = st.columns([2,1])
            with c1:
                st.image(cv2.cvtColor(result['annotated'],cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.subheader("Detected Items:")
                for i,det in enumerate(result['detections'],1):
                    st.write(f"**{i}. {det['label'].title()}** ({det['confidence']:.0%})")
                    for b in det['brands'][:3]:
                        method = b.get('detection_method','ocr')
                        if method == 'clip':
                            continue
                        st.write(f"   📝 {b['text']} ({b['confidence']:.0%})")

    st.markdown("---"); st.header("🏷️ Asset Stickers")
    date_str = datetime.now().strftime("%d %b %Y")
    for result in results_log:
        if not result['detections']: continue
        st.subheader(f"📸 {result['file']}")
        cols = st.columns(min(len(result['detections']),3))
        for i, det in enumerate(result['detections']):
            asset_id = generate_asset_id()
            x1,y1,x2,y2 = det['bbox']; pad=20
            h,w = result['annotated'].shape[:2]
            crop = result['annotated'][max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
            sticker_png = generate_sticker_png(asset_id, result['file'], det, i+1, date_str,
                                               crop if crop.size>0 else None)
            with cols[i % len(cols)]:
                brand = det['brands'][0]['text'] if det['brands'] else "Unknown"
                st.caption(f"Item {i+1}: {det['label'].title()} — {brand}")
                if sticker_png:
                    st.image(PILImage.open(BytesIO(sticker_png.getvalue())), use_container_width=True)
                    safe_fname = re.sub(r'[^A-Za-z0-9_.-]', '_', result['file'])
                    st.download_button("⬇️ Download", data=sticker_png,
                        file_name=f"sticker_{asset_id}.png", mime="image/png",
                        key=f"stk_{safe_fname}_{i}_{asset_id[:8]}",
                        use_container_width=True)
                det_id = None
                for sr in store.scan_results:
                    if sr.file_name == result['file']:
                        for sd in sr.detections:
                            if sd.position == det['position']:
                                det_id = sd.detection_id
                                break
                        break
                store.push_sticker(
                    asset_id=asset_id,
                    detection_id=det_id,
                    session_id=session_id,
                    item_label=det['label'],
                    brand=brand if brand != "Unknown" else None,
                    confidence=det['confidence'],
                    file_name=result['file'],
                )

    st.markdown("---")
    comparison = None
    if do_data:
        st.header("📋 PDF DO Comparison")
        comparison = compare_quantities(actual_counts, do_data)
        store.do_comparison = comparison
        if comparison and do_id:
            store.push_do_comparison(do_id, session_id, comparison)
        if comparison:
            if comparison["matches"]:
                st.success("✅ **Matching Items:**")
                for m in comparison["matches"]:
                    st.write(f"- {m['item_type'].title()}: {m['actual']}/{m['expected']} ✓")
            if comparison["discrepancies"]:
                st.error("⚠️ **Discrepancies:**")
                for d in comparison["discrepancies"]:
                    st.write(f"- {d['item_type'].title()}: expected {d['expected']}, got {d['actual']}"
                             f" (diff {d['difference']:+d})")
            else:
                st.success("🎉 ALL ITEMS MATCH!")

    st.markdown("---"); st.header("📥 Download Report")
    excel_data = create_excel_report(results_log, do_data, comparison)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("📊 Download Excel Report", data=excel_data,
        file_name=f"detection_results_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"excel_dl_{ts}",
        use_container_width=True)

    st.info("✅ Scan results saved to shared dataset — proceed to **Auto-Picker** tab.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — TAB: AUTO-PICKER
# ═══════════════════════════════════════════════════════════════════════

def render_picker_tab(store: SharedStore):
    ap: StreamlitAutoPicker = st.session_state.auto_picker

    st.markdown("## 🤖 Auto-Picker — Subsystem 2")

    sub_tabs = st.tabs(["📊 Dashboard","📝 Create Order","🗺️ Process Order",
                        "📋 Order Management","ℹ️ Documentation"])

    with sub_tabs[0]:   _picker_dashboard(ap, store)
    with sub_tabs[1]:   _picker_create_order(ap, store)
    with sub_tabs[2]:   _picker_process_order(ap, store)
    with sub_tabs[3]:   _picker_order_management(ap)
    with sub_tabs[4]:   _picker_documentation()


def _picker_dashboard(ap: StreamlitAutoPicker, store: SharedStore):
    stats = ap.get_system_stats()
    c1,c2 = st.columns([2,1])
    with c1:
        st.subheader("🗺️ Warehouse Grid Heatmap")
        grid = ap.get_grid_heatmap()
        fig = px.imshow(grid,labels=dict(x="Rack",y="Aisle",color="Qty"),
            x=[f"R{i:02d}" for i in range(1,25)],
            y=[f"Aisle {i}" for i in range(1,7)],
            color_continuous_scale="Viridis",aspect="auto")
        fig.update_layout(height=380,title="Inventory Distribution",title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📊 Key Metrics")
        fig2 = make_subplots(rows=3,cols=1,
            subplot_titles=("Success Rate","Utilization","Orders Complete"),
            specs=[[{"type":"indicator"}],[{"type":"indicator"}],[{"type":"indicator"}]])
        fig2.add_trace(go.Indicator(mode="gauge+number",value=stats['success_rate'],
            gauge={'axis':{'range':[None,100]},'bar':{'color':'#00cc96'}}),row=1,col=1)
        _grid_cap = WAREHOUSE_CONFIG['grid_size'][0] * WAREHOUSE_CONFIG['grid_size'][1] * 100
        util = (stats['total_items'] / _grid_cap * 100) if _grid_cap else 0
        fig2.add_trace(go.Indicator(mode="gauge+number",value=min(util,100),
            gauge={'axis':{'range':[None,100]},'bar':{'color':'#636efa'}}),row=2,col=1)
        ful = (stats['completed_orders']/max(stats['total_orders'],1))*100
        fig2.add_trace(go.Indicator(mode="gauge+number",value=ful,
            gauge={'axis':{'range':[None,100]},'bar':{'color':'#ff7f0e'}}),row=3,col=1)
        fig2.update_layout(height=480,showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c1b,c2b = st.columns(2)
    with c1b:
        st.subheader("📋 Recent Orders")
        odf = ap.get_orders_dataframe()
        if not odf.empty:
            st.dataframe(odf.sort_values("Created",ascending=False).head(5),
                         use_container_width=True,hide_index=True)
        else: st.info("No orders yet")
    with c2b:
        st.subheader("⚠️ Low Stock Alert")
        idf = ap.get_inventory_dataframe()
        ls  = idf[idf["Quantity"]<WAREHOUSE_CONFIG["low_stock_threshold"]]
        if not ls.empty:
            st.dataframe(ls[["SKU","Name","Location","Quantity"]].head(5),
                         use_container_width=True,hide_index=True)
        else: st.success("No low stock items")

    if store.actual_counts:
        st.markdown("---")
        st.subheader("🔗 Suggested Orders from Latest Scan")
        rows=[]
        for item_type, count in store.actual_counts.items():
            reg_sku = ap.brand_sku_registry.get(item_type)
            if reg_sku and reg_sku in ap.inventory:
                matching = [reg_sku]
            else:
                matching = [sku for sku,inv in ap.inventory.items()
                            if item_type.lower() in inv.name.lower() or
                               item_type.lower() in inv.category.lower()]
            rows.append({
                "Scanned Item": item_type.title(),
                "Count": count,
                "SKU": matching[0] if matching else "—",
                "Name": ap.inventory[matching[0]].name if matching else "—",
                "In Stock": ap.inventory[matching[0]].quantity if matching else 0,
            })
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)


def _picker_create_order(ap: StreamlitAutoPicker, store: SharedStore):
    st.subheader("📝 Create New Order")

    if "order_cart" not in st.session_state:
        st.session_state.order_cart = {}

    cart: dict = st.session_state.order_cart

    c1, c2 = st.columns(2)
    with c1:
        customer_id   = st.text_input("Customer ID", placeholder="e.g., CUST001",
                                      key="co_customer_id")
        customer_type = st.selectbox("Customer Type", list(VALID_CUSTOMER_TYPES),
                                     format_func=str.title, key="co_customer_type")
    with c2:
        st.info("**Priority:** Wholesale=1 · Repair=2 · Retail=3")
        if cart:
            n_skus  = len(cart)
            n_units = sum(cart.values())
            st.success(f"🛒 Cart: **{n_units}** unit(s) across **{n_skus}** SKU(s)")

    st.divider()

    idf = ap.get_inventory_dataframe()
    cats = ["All"] + list(idf["Category"].unique())
    col_f, col_s = st.columns([1, 2])
    cat_sel     = col_f.selectbox("Category", cats, key="co_cat")
    search_term = col_s.text_input("🔎 Search SKU / name", "",
                                   placeholder="e.g. ELEC-0042 or Laptop",
                                   key="co_search")

    fdf = idf if cat_sel == "All" else idf[idf["Category"] == cat_sel]
    if search_term.strip():
        mask = (fdf["SKU"].str.contains(search_term, case=False, na=False) |
                fdf["Name"].str.contains(search_term, case=False, na=False))
        fdf = fdf[mask]

    total_items = len(fdf)
    page_size   = 15
    total_pages = max(1, -(-total_items // page_size))

    if total_pages > 1:
        page = st.number_input(
            f"Page (1 – {total_pages})", min_value=1, max_value=total_pages,
            value=1, step=1, key="co_page"
        ) - 1
    else:
        page = 0

    start    = page * page_size
    page_df  = fdf.iloc[start : start + page_size]
    st.caption(
        f"Showing {start+1}–{min(start+page_size, total_items)} "
        f"of {total_items} item(s)"
        + (f"  ·  Page {page+1}/{total_pages}" if total_pages > 1 else "")
    )

    for _, row in page_df.iterrows():
        sku        = row["SKU"]
        max_qty    = min(int(row["Quantity"]), 100)
        saved_qty  = cart.get(sku, 0)

        c1b, c2b, c3b = st.columns([3, 1, 1])
        c1b.write(f"**{sku}** — {row['Name']}")
        c1b.caption(f"{row['Category']} | {row['Location']}")
        c2b.write(f"Stock: {row['Quantity']}")
        c2b.write(f"${row['Value ($)']}")

        new_qty = c3b.number_input(
            "Qty",
            min_value=0,
            max_value=max_qty,
            value=saved_qty,
            step=1,
            key=f"co_qty_{sku}",
            label_visibility="collapsed",
        )

        if new_qty != saved_qty:
            if new_qty == 0:
                cart.pop(sku, None)
            else:
                cart[sku] = new_qty

    st.divider()
    if cart:
        details = []
        tv = 0.0
        for sku, qty in sorted(cart.items()):
            if sku not in ap.inventory:
                continue
            it  = ap.inventory[sku]
            v   = it.value * qty
            tv += v
            details.append({
                "SKU":   sku,
                "Name":  it.name,
                "Qty":   qty,
                "Unit":  f"${it.value:,.2f}",
                "Total": f"${v:,.2f}",
            })

        if details:
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
            st.metric("Order Total", f"${tv:,.2f}")

        col_btn, col_clr = st.columns([3, 1])
        with col_btn:
            if st.button("✅ Create Order", type="primary", use_container_width=True):
                items_to_order = [(sku, qty) for sku, qty in cart.items()
                                  if sku in ap.inventory]
                res = ap.create_order(
                    customer_id   = st.session_state.get("co_customer_id", ""),
                    customer_type = st.session_state.get("co_customer_type", "retail"),
                    items         = items_to_order,
                )
                if res["success"]:
                    oid = res["order_id"]
                    st.session_state.picker_current_order = oid
                    st.session_state.picker_success.append(f"Order {oid} created")
                    st.session_state.order_cart = {}
                    st.success(f"✅ {oid} created! Go to the Process Order tab to view the route.")
                    st.rerun()
                else:
                    st.error(res["error"])
        with col_clr:
            if st.button("🗑️ Clear Cart", use_container_width=True):
                st.session_state.order_cart = {}
                st.rerun()
    else:
        st.info("Select items and quantities to build an order.")


def _picker_process_order(ap: StreamlitAutoPicker, store: SharedStore):
    st.subheader("🗺️ Process & Pick Orders")
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("### 📋 Orders")

        # Show BOTH pending AND processing orders so they don't vanish after clicking View Route
        actionable = [
            (oid, o) for oid, o in ap.orders.items()
            if o.status in [OrderStatus.PENDING, OrderStatus.PROCESSING]
        ]

        if actionable:
            for oid, order in actionable:
                st.markdown(f"**{oid}**")
                st.caption(f"Customer: {order.customer_id} | Status: {order.status.value}")
                cb1, cb2 = st.columns(2)

                if cb1.button("🗺️ View Route", key=f"proc_{oid}", use_container_width=True):
                    # Move to PROCESSING only if still PENDING
                    if order.status == OrderStatus.PENDING:
                        ap.process_order(oid)
                    st.session_state.picker_current_order = oid
                    st.session_state.last_pick_result = None
                    st.rerun()

                if cb2.button("✅ Pick", key=f"pick_{oid}", use_container_width=True):
                    if order.status == OrderStatus.PENDING:
                        ap.process_order(oid)
                    r = ap.execute_picking(oid)
                    if r.get("success"):
                        _sync_order_to_store(ap, oid, store)
                        ful = r.get("fulfillment") or 0.0
                        st.session_state.last_pick_result = {
                            "order_id":    oid,
                            "fulfillment": ful,
                            "picked":      r.get("picked_count", 0),
                            "failed":      r.get("failed_count", 0),
                            "value":       r.get("picked_value", 0.0),
                            "failed_items": r.get("failed_items") or [],
                        }
                        st.session_state.picker_current_order = None
                        st.rerun()
                    else:
                        st.error(r.get("error", "Picking failed"))

                st.divider()
        else:
            st.info("No pending or processing orders")

    with c2:
        # ── Show last pick result persistently (survives the rerun) ──
        if st.session_state.get("last_pick_result"):
            r = st.session_state.last_pick_result
            st.success(f"✅ Order **{r['order_id']}** complete — {r['fulfillment']:.1f}% fulfilled")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Items Picked", r["picked"])
            rc2.metric("Items Failed", r["failed"])
            rc3.metric("Value Picked", f"${r['value']:,.2f}")
            if r["failed_items"]:
                st.warning("Failed items: " +
                           ", ".join(i.get("sku", "?") for i in r["failed_items"]))
            if st.button("🗑️ Clear Result"):
                st.session_state.last_pick_result = None
                st.rerun()
            return  # don't render the route map at the same time

        current_oid = st.session_state.picker_current_order

        # Auto-select first actionable order if none is selected yet
        if not current_oid:
            actionable_ids = [
                oid for oid, o in ap.orders.items()
                if o.status in [OrderStatus.PENDING, OrderStatus.PROCESSING]
            ]
            if actionable_ids:
                current_oid = actionable_ids[0]
                st.session_state.picker_current_order = current_oid

        if current_oid and current_oid in ap.orders:
            res = ap.get_route_preview(current_oid)
            if res.get("success"):
                st.markdown("### 🗺️ Optimised Picking Route")
                m = res["metrics"]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("📍 Stops",    m['items_count'])
                mc2.metric("📏 Distance", f"{m['total_distance_m']}m")
                mc3.metric("⏱️ Time",     f"{m['total_time_min']}min")
                mc4.metric("💰 Value",    f"${res.get('order_value') or 0:,.2f}")

                rdf = pd.DataFrame({
                    "Stop":     range(1, len(res['route']) + 1),
                    "Location": res['route']
                })
                st.dataframe(rdf, hide_index=True, use_container_width=True)

                # ── Route map ─────────────────────────────────────────
                coords = []
                for ls in res['route']:
                    parts = ls.split('-')
                    coords.append((int(parts[1][1:]), int(parts[0][1:])))

                _MAX_RACK = 24
                _FAR_END  = _MAX_RACK   # rack 24 — stays within chart
                _ENTRANCE = 1           # rack 1  — stays within chart

                def _aisle_cross(x1, y1, x2, y2):
                    cost_here     = abs(x2 - x1)
                    cost_entrance = abs(x1 - _ENTRANCE) + abs(x2 - _ENTRANCE)
                    cost_far_end  = abs(x1 - _FAR_END)  + abs(x2 - _FAR_END)
                    best = min(cost_here, cost_entrance, cost_far_end)
                    if best == cost_here:
                        return [(x1, y1), (x1, y2), (x2, y2)]
                    elif best == cost_entrance:
                        return [(x1, y1), (_ENTRANCE, y1), (_ENTRANCE, y2), (x2, y2)]
                    else:
                        return [(x1, y1), (_FAR_END, y1), (_FAR_END, y2), (x2, y2)]

                def _build_route(depot_pt, stop_coords):
                    all_pts = [depot_pt] + list(stop_coords) + [depot_pt]
                    wp = []
                    for i in range(len(all_pts) - 1):
                        x1, y1 = all_pts[i]
                        x2, y2 = all_pts[i + 1]
                        if y1 == y2:
                            wp += [(x1, y1), (x2, y2)]
                        else:
                            wp += _aisle_cross(x1, y1, x2, y2)
                    out = [wp[0]] if wp else []
                    for pt in wp[1:]:
                        if pt != out[-1]:
                            out.append(pt)
                    return out

                depot_pt  = (1, 1)
                waypoints = _build_route(depot_pt, coords) if coords else []

                fig = go.Figure()

                # Aisle shading
                for a in range(1, 7):
                    fig.add_hrect(
                        y0=a - 0.4, y1=a + 0.4,
                        fillcolor='rgba(220,230,255,0.3)',
                        line_width=0, layer='below')

                # Corridor shading
                for cx_col in [_ENTRANCE, _FAR_END]:
                    fig.add_vrect(
                        x0=cx_col - 0.4, x1=cx_col + 0.4,
                        fillcolor='rgba(255,220,100,0.2)',
                        line_width=0, layer='below')

                # Rack grid
                gx = [r for a in range(1, 7) for r in range(1, _MAX_RACK + 1)]
                gy = [a for a in range(1, 7) for r in range(1, _MAX_RACK + 1)]
                fig.add_trace(go.Scatter(
                    x=gx, y=gy, mode='markers',
                    marker=dict(size=7, color='#d0d4e8', symbol='square'),
                    showlegend=False, hoverinfo='none', name='grid'))

                # Corridor labels
                for cx_col, lbl in [(_ENTRANCE, 'Entrance'), (_FAR_END, 'Far end')]:
                    fig.add_annotation(
                        x=cx_col, y=6.8, text=lbl,
                        showarrow=False, font=dict(size=10, color='#996600'),
                        xanchor='center')

                # Route path
                if waypoints:
                    wx = [w[0] for w in waypoints]
                    wy = [w[1] for w in waypoints]
                    fig.add_trace(go.Scatter(
                        x=wx, y=wy, mode='lines',
                        line=dict(width=2.5, color='#e74c3c'),
                        showlegend=True, name='Route',
                        hoverinfo='skip'))

                # Stop markers
                if coords:
                    sx = [c[0] for c in coords]
                    sy = [c[1] for c in coords]
                    slabels = [str(i + 1) for i in range(len(coords))]
                    fig.add_trace(go.Scatter(
                        x=sx, y=sy, mode='markers+text',
                        marker=dict(size=13, color='#3498db',
                                    line=dict(width=1.5, color='white')),
                        text=slabels, textposition='top center',
                        showlegend=True, name='Stops',
                        hovertemplate='Stop %{text} — Rack %{x}, Aisle %{y}<extra></extra>'))

                # Depot marker
                fig.add_trace(go.Scatter(
                    x=[depot_pt[0]], y=[depot_pt[1]],
                    mode='markers+text',
                    marker=dict(size=16, color='#2ecc71', symbol='star',
                                line=dict(width=1.5, color='white')),
                    text=['Depot'], textposition='top right',
                    showlegend=True, name='Depot'))

                fig.update_layout(
                    title="Picking Route — Warehouse Floor Plan",
                    xaxis=dict(
                        title='Rack #', range=[0, 25], dtick=2,
                        tickvals=list(range(0, 26)),
                        ticktext=['↔'] + [str(r) for r in range(1, 25)] + ['↔'],
                        gridcolor='#eeeeee', zeroline=False),
                    yaxis=dict(
                        title='Aisle', range=[0.3, 7.1], dtick=1,
                        tickvals=list(range(1, 7)),
                        ticktext=[f'Aisle {i}' for i in range(1, 7)],
                        gridcolor='#eeeeee', zeroline=False),
                    height=440, plot_bgcolor='white', paper_bgcolor='white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                    margin=dict(l=60, r=20, t=70, b=40))

                st.plotly_chart(fig, use_container_width=True)

                if st.button("🚀 Execute Picking", type="primary", use_container_width=True):
                    pr = ap.execute_picking(current_oid)
                    if pr.get("success"):
                        _sync_order_to_store(ap, current_oid, store)
                        ful = pr.get("fulfillment") or 0.0
                        st.session_state.last_pick_result = {
                            "order_id":    current_oid,
                            "fulfillment": ful,
                            "picked":      pr.get("picked_count", 0),
                            "failed":      pr.get("failed_count", 0),
                            "value":       pr.get("picked_value", 0.0),
                            "failed_items": pr.get("failed_items") or [],
                        }
                        st.session_state.picker_current_order = None
                        st.rerun()
                    else:
                        st.error(pr.get("error", "Picking failed"))
            else:
                st.error(f"Route error: {res.get('error')}")
        else:
            st.info("Select an order from the left to view its route.")


def _sync_order_to_store(ap: StreamlitAutoPicker, order_id: str, store: SharedStore):
    if order_id not in ap.orders: return
    o = ap.orders[order_id]
    items = [
        {
            "sku":      i.sku,
            "name":     ap.inventory[i.sku].name if i.sku in ap.inventory else i.sku,
            "quantity": i.quantity,
            "picked":   i.picked,
            "status":   i.status,
            "value":    ap.inventory[i.sku].value * i.quantity if i.sku in ap.inventory else 0,
        }
        for i in o.items
    ]
    po = PickerOrder(
        order_id=order_id, customer_id=o.customer_id, customer_type=o.customer_type,
        status=o.status.value,
        items=items,
        route=[], route_metrics={},
        fulfillment=o.fulfillment_rate(),
        order_value=ap._calculate_order_value(o),
        completed_at=o.completed_at.isoformat() if o.completed_at else None,
    )
    store.push_picker_order(po)
    store.update_picker_order_in_db(po)


def _picker_order_management(ap: StreamlitAutoPicker):
    st.subheader("📋 All Orders")
    odf = ap.get_orders_dataframe()
    if odf.empty: st.info("No orders yet"); return
    c1,c2 = st.columns(2)
    sf = c1.multiselect("Filter Status", odf["Status"].unique())
    tf = c2.multiselect("Filter Type",   odf["Type"].unique())
    fdf = odf.copy()
    if sf: fdf=fdf[fdf["Status"].isin(sf)]
    if tf: fdf=fdf[fdf["Type"].isin(tf)]
    fdf_disp = fdf.copy()
    fdf_disp["Fulfillment"] = fdf_disp["Fulfillment"].str.replace("%","").astype(float)
    st.dataframe(fdf_disp.sort_values("Created",ascending=False),
                 use_container_width=True,hide_index=True,
                 column_config={"Fulfillment":st.column_config.ProgressColumn(
                     format="%.1f%%",min_value=0,max_value=100)})


def _picker_documentation():
    st.subheader("📚 Documentation")
    with st.expander("System Architecture"):
        st.markdown("""
**Grid:** 6 aisles × 24 racks | **Route optimisation:** A-star / nearest-neighbour
**Categories:** computers · phones · accessories · components

Error handling: input validation → partial fulfilment → detailed logging
        """)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — TAB: COURIER EMAIL
# ═══════════════════════════════════════════════════════════════════════

def render_courier_tab(store: SharedStore):
    st.markdown("## 📧 Courier Email — Subsystem 3")

    total   = len(store.shipments)
    conf    = sum(1 for s in store.shipments.values() if s.status=="confirmed")
    failed  = sum(1 for s in store.shipments.values() if s.status=="failed")
    rate    = f"{conf/total*100:.0f}" if total else "0"
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h3>{total}</h3><p>Total Processed</p></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card success-card"><h3>{conf}</h3><p>Confirmed</p></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card fail-card"><h3>{failed}</h3><p>Failed</p></div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card rate-card"><h3>{rate}%</h3><p>Success Rate</p></div>',unsafe_allow_html=True)
    st.markdown("")

    sub_tabs = st.tabs(["📦 Generate Shipment","📋 Activity Log","📧 Email Preview","📊 AI Info"])

    with sub_tabs[0]:
        st.subheader("📦 Generate Courier Email")

        completed_orders = store.get_completed_orders()
        if not completed_orders:
            st.info("No completed picker orders available. Finish picking in the Auto-Picker tab first.")
            ap: StreamlitAutoPicker = st.session_state.auto_picker
            completed_picker = [o for o in ap.orders.values() if o.status==OrderStatus.COMPLETE]
            if completed_picker:
                st.warning(f"Found {len(completed_picker)} completed orders in picker — syncing…")
                for o in completed_picker:
                    _sync_order_to_store(ap, o.order_id, store)
                st.rerun()
            return

        unshipped = store.get_unshipped_orders()
        if not unshipped:
            st.success("All completed orders have been shipped! 🎉")
            return

        order_options = {f"{o.order_id} — {o.customer_id} (${o.order_value:,.2f})": o
                         for o in unshipped}
        sel_label = st.selectbox("Select completed order to ship", list(order_options.keys()))
        sel_order = order_options[sel_label]

        c1b,c2b = st.columns(2)
        with c1b:
            st.markdown("**Order Details**")
            st.write(f"Customer: {sel_order.customer_id} ({sel_order.customer_type})")
            st.write(f"Items: {len(sel_order.items)}")
            st.write(f"Value: SGD {sel_order.order_value:,.2f}")
            st.write(f"Fulfillment: {sel_order.fulfillment:.1f}%")
        with c2b:
            st.markdown("**Items**")
            for it in sel_order.items[:5]:
                st.write(f"• {it.get('name') or it.get('sku','?')} x{it.get('quantity',0)}")
            if len(sel_order.items)>5:
                st.caption(f"…and {len(sel_order.items)-5} more")

        st.markdown("---")
        weight = sum(i.get("quantity",1)*0.5 for i in sel_order.items)
        fragile = any("laptop" in i.get("name","").lower() or
                      "phone" in i.get("name","").lower()
                      for i in sel_order.items)
        try:
            from ai_decision_engine import AIDecisionEngine
            engine = AIDecisionEngine(); engine.train()
            method, conf_score = engine.predict_shipping(weight_kg=weight, volume_cm3=5000,
                                                          distance_km=500, priority="standard")
        except Exception:
            method, conf_score = _stub_ai_select_shipping(weight, 5000, 500, "standard")

        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("📦 Shipping Method", method)
        mc2.metric("🤖 ML Confidence",   f"{conf_score:.0%}")
        mc3.metric("🔍 Fragile",         "⚠️ Yes" if fragile else "✅ No")

        if st.button("📧 Preview Email", use_container_width=True):
            email = generate_courier_email(sel_order, method, fragile, conf_score)
            st.session_state.pending_shipment_preview = {"order": sel_order, "email": email,
                "method": method, "conf": conf_score, "fragile": fragile}

        if st.session_state.pending_shipment_preview:
            prev = st.session_state.pending_shipment_preview
            if prev["order"].order_id == sel_order.order_id:
                em = prev["email"]
                st.markdown(f"**To:** `{em['to']}`")
                st.markdown(f"**Subject:** {em['subject']}")
                st.markdown(f'<div class="email-preview">{em["body"]}</div>', unsafe_allow_html=True)
                st.markdown("")
                if st.button("✅ Confirm & Send", type="primary", use_container_width=True):
                    sid = f"SHP-{uuid.uuid4().hex[:8].upper()}"
                    shipment = ShipmentRecord(
                        shipment_id=sid, order_id=sel_order.order_id,
                        status="confirmed", shipping_method=method, courier=COURIER,
                        is_fragile=fragile, email_to=em["to"],
                        email_subject=em["subject"], email_body=em["body"],
                    )
                    store.push_shipment(shipment)
                    st.session_state.courier_last_email = em
                    st.session_state.pending_shipment_preview = None
                    st.success(f"✅ Shipment {sid} confirmed! Email generated.")
                    st.rerun()

    with sub_tabs[1]:
        st.subheader("📋 Processed Shipments")
        if store.shipments:
            rows=[{"Shipment":sid,"Order":s.order_id,"Status":s.status,
                   "Shipping":s.shipping_method,"Courier":s.courier,
                   "Fragile":"⚠️" if s.is_fragile else "✓","Time":s.created_at[:19]}
                  for sid,s in sorted(store.shipments.items(),
                                      key=lambda x:x[1].created_at,reverse=True)]
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            method_counts={}
            for s in store.shipments.values():
                method_counts[s.shipping_method]=method_counts.get(s.shipping_method,0)+1
            st.markdown("---"); st.markdown("**By Shipping Method:**")
            st.bar_chart(pd.DataFrame({"Count":method_counts}))
        else:
            st.info("No shipments processed yet.")

    with sub_tabs[2]:
        st.subheader("📧 Last Generated Email")
        last = st.session_state.courier_last_email
        if not last and store.shipments:
            sid = sorted(store.shipments, key=lambda x: store.shipments[x].created_at)[-1]
            s   = store.shipments[sid]
            last = {"to":s.email_to,"subject":s.email_subject,"body":s.email_body,"order_id":s.order_id}
        if last:
            st.markdown(f"**To:** `{last['to']}`")
            st.markdown(f"**Subject:** {last['subject']}")
            st.markdown(f'<div class="email-preview">{last["body"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No emails generated yet.")

    with sub_tabs[3]:
        st.subheader("📊 AI Shipping Model")
        st.markdown("""
**Algorithm:** Random Forest Classifier (stub rule-based if RF unavailable)

| Feature        | Role |
|----------------|------|
| Weight (kg)    | Predicts if Standard is feasible |
| Volume (cm³)   | Size-based routing |
| Distance (km)  | Same-Day vs Express vs Standard |
| Priority       | Overrides when urgent |

**Methods:** Same-Day Express · Express · Standard  
**Courier:** FedEx (fixed partner for all shipments)
        """)
        st.code("""
IF priority == 'urgent' OR distance < 100 km:
    → Same-Day Express  (~92% confidence)
ELIF weight > 10 kg OR distance > 3000 km:
    → Standard          (~85% confidence)
ELSE:
    → Express           (~88% confidence)
        """, language="text")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    store = init_session()

    render_sidebar(store)

    st.markdown('<p class="wis-header">🏭 Warehouse Intelligence System</p>', unsafe_allow_html=True)
    st.markdown('<p class="wis-sub">Integrated pipeline: Object Scanning → Auto-Picking → Courier Email</p>',
                unsafe_allow_html=True)

    tabs = st.tabs([
        "🔄 Pipeline Overview",
        "📷 Object Scanner",
        "🤖 Auto-Picker",
        "📧 Courier Email",
        "🗄️ Database",
    ])

    with tabs[0]: render_pipeline_tab(store)
    with tabs[1]: render_scanner_tab(store)
    with tabs[2]: render_picker_tab(store)
    with tabs[3]: render_courier_tab(store)
    with tabs[4]: render_db_tab()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.code(traceback.format_exc())