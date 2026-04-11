from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import re, json, os, base64, time
from openai import OpenAI
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import hashlib
import concurrent.futures
import threading
import os

try:
    import fitz
    PDF_SUPPORT = True
    print("✅ PyMuPDF loaded")
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyMuPDF not installed — run: pip install pymupdf")

app = Flask(__name__)
CORS(app)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
UPLOAD_FOLDER = r"C:\Users\Dell\OneDrive\Desktop\govt-miniproject\pan_card_assistant\backend\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mongo        = MongoClient("mongodb://localhost:27017/")
db           = mongo["gov_doc_ai"]
collection   = db["applications"]
training_col = db["training_data"]
users_col    = db["users"]

grok_client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.x.ai/v1",
)
GROK_MODEL = "grok-2-vision-1212"

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ── Store last extraction result for debugging ───────────────
_last_debug = {}


def make_empty():
    return {k: "" for k in [
        "first_name", "middle_name", "last_name", "dob", "gender",
        "aadhaar_number", "father_first_name", "father_last_name",
        "flat_no", "building", "road", "locality", "city", "district",
        "residentState", "pincode", "address", "email", "mobile",
        "areaCode", "aoType", "rangeCode", "aoNumber"
    ]}

def hash_pw(pwd): return hashlib.sha256(pwd.encode()).hexdigest()

def is_bad_name(v):
    if not v or len(v.strip()) < 2 or len(v.strip()) > 25: return True
    if not re.match(r"^[A-Za-z][A-Za-z\s\-\.]*$", v.strip()): return True
    for w in v.strip().split():
        if len(w) > 12: return True
        if re.search(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}", w): return True
    return False


# ════════════════════════════════════════════════════════════
# GENDER + DOB  ← dedicated fast extractors
# ════════════════════════════════════════════════════════════

def extract_gender(text: str) -> str:
    """
    Returns 'Male', 'Female', or '' by scanning the raw OCR text.
    Handles all common Aadhaar OCR variants:
      FEMALE / Female / female / MALE / Male / male
      'Gender: Female' / 'Sex: Male'
      Partial matches like 'FEMAL', 'EMALE', 'MAL E'
    """
    t = text.upper().replace(" ", "").replace("\n", "")
    # Try exact match first (most reliable)
    if "FEMALE" in t:
        return "Female"
    if re.search(r'\bMALE\b', text, re.IGNORECASE):
        return "Male"
    # Labeled variants
    m = re.search(r'(?:Gender|Sex)\s*[:\-]\s*(Male|Female|MALE|FEMALE)', text, re.IGNORECASE)
    if m:
        v = m.group(1).strip().capitalize()
        return "Female" if v.lower() == "female" else "Male"
    # Partial OCR corruption fallback
    if re.search(r'FEM[A-Z]{0,3}LE|FE[A-Z]{0,2}ALE', t):
        return "Female"
    if re.search(r'\bM[A-Z]{0,2}LE\b', t):
        return "Male"
    return ""


def extract_dob(text: str) -> str:
    """
    Returns DOB as 'YYYY-MM-DD' or '' by scanning raw OCR text.
    Handles:
      DOB: 15/02/2005
      Date of Birth: 15-02-2005
      15/02/2005  (bare date near DOB keyword)
      Year of Birth: 2005  (returns '2005-01-01' as best guess)
    """
    # Pattern 1: labeled DOB with full date
    m = re.search(
        r'(?:DOB|Date\s*of\s*Birth|D\.O\.B)\s*[:\-]?\s*'
        r'(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{4})',
        text, re.IGNORECASE
    )
    if m:
        dd, mm, yyyy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        print(f"  ✅ DOB (labeled): {yyyy}-{mm}-{dd}")
        return f"{yyyy}-{mm}-{dd}"

    # Pattern 2: unlabeled date — any DD/MM/YYYY or DD-MM-YYYY in plausible range
    for pat in [
        r'\b(\d{2})[/\-\.](\d{2})[/\-\.]((?:19[4-9]\d|200[0-9]|201[0-9]|202[0-5]))\b',
        r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.]((?:19|20)\d{2})\b',
    ]:
        m = re.search(pat, text)
        if m:
            dd, mm, yyyy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
            # Sanity: month 1-12, day 1-31
            if 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                print(f"  ✅ DOB (unlabeled): {yyyy}-{mm}-{dd}")
                return f"{yyyy}-{mm}-{dd}"

    # Pattern 3: Year of Birth only
    m = re.search(r'(?:Year\s*of\s*Birth|YOB)\s*[:\-]?\s*(\d{4})', text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-01-01"

    return ""


# ════════════════════════════════════════════════════════════
# AO CODE DB
# ════════════════════════════════════════════════════════════
AO_CODE_DB = {
    "ernakulam":{"areaCode":"KRL","aoType":"W","rangeCode":"15","aoNumber":"1"},
    "kochi":{"areaCode":"KRL","aoType":"W","rangeCode":"15","aoNumber":"2"},
    "kakkanad":{"areaCode":"KRL","aoType":"W","rangeCode":"16","aoNumber":"1"},
    "kalamassery":{"areaCode":"KRL","aoType":"W","rangeCode":"16","aoNumber":"2"},
    "aluva":{"areaCode":"KRL","aoType":"W","rangeCode":"16","aoNumber":"3"},
    "edappally":{"areaCode":"KRL","aoType":"W","rangeCode":"15","aoNumber":"4"},
    "chethipuzha":{"areaCode":"KRL","aoType":"W","rangeCode":"6","aoNumber":"2"},
    "thiruvananthapuram":{"areaCode":"KRL","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "trivandrum":{"areaCode":"KRL","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "thrissur":{"areaCode":"KRL","aoType":"W","rangeCode":"5","aoNumber":"1"},
    "kozhikode":{"areaCode":"KRL","aoType":"W","rangeCode":"8","aoNumber":"1"},
    "kannur":{"areaCode":"KRL","aoType":"W","rangeCode":"10","aoNumber":"1"},
    "kollam":{"areaCode":"KRL","aoType":"W","rangeCode":"3","aoNumber":"1"},
    "alappuzha":{"areaCode":"KRL","aoType":"W","rangeCode":"4","aoNumber":"1"},
    "kottayam":{"areaCode":"KRL","aoType":"W","rangeCode":"6","aoNumber":"1"},
    "palakkad":{"areaCode":"KRL","aoType":"W","rangeCode":"7","aoNumber":"1"},
    "malappuram":{"areaCode":"KRL","aoType":"W","rangeCode":"9","aoNumber":"1"},
    "kasaragod":{"areaCode":"KRL","aoType":"W","rangeCode":"11","aoNumber":"1"},
    "pathanamthitta":{"areaCode":"KRL","aoType":"W","rangeCode":"2","aoNumber":"1"},
    "idukki":{"areaCode":"KRL","aoType":"W","rangeCode":"12","aoNumber":"1"},
    "wayanad":{"areaCode":"KRL","aoType":"W","rangeCode":"13","aoNumber":"1"},
    "kerala":{"areaCode":"KRL","aoType":"W","rangeCode":"14","aoNumber":"1"},
    "chennai":{"areaCode":"TND","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "coimbatore":{"areaCode":"TND","aoType":"W","rangeCode":"5","aoNumber":"1"},
    "bengaluru":{"areaCode":"KTK","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "bangalore":{"areaCode":"KTK","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "mumbai":{"areaCode":"MUM","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "pune":{"areaCode":"MUM","aoType":"W","rangeCode":"8","aoNumber":"1"},
    "delhi":{"areaCode":"DLC","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "hyderabad":{"areaCode":"HYD","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "kolkata":{"areaCode":"WBN","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "lucknow":{"areaCode":"UPL","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "ahmedabad":{"areaCode":"GUJ","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "jaipur":{"areaCode":"RAJ","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "chandigarh":{"areaCode":"CHG","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "bhopal":{"areaCode":"MPR","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "patna":{"areaCode":"BIH","aoType":"W","rangeCode":"1","aoNumber":"1"},
    "guwahati":{"areaCode":"ASM","aoType":"W","rangeCode":"1","aoNumber":"1"},
}

def lookup_ao_code(locality, city, district, state):
    empty = {"areaCode":"","aoType":"","rangeCode":"","aoNumber":""}
    candidates = [(locality or "").lower().strip(),(city or "").lower().strip(),(district or "").lower().strip(),(state or "").lower().strip()]
    for key in candidates:
        if key and key in AO_CODE_DB: return AO_CODE_DB[key]
    for key in candidates:
        if not key: continue
        for db_key in AO_CODE_DB:
            if db_key in key or key in db_key: return AO_CODE_DB[db_key]
    return empty

KNOWN_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra",
    "Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim",
    "Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"
]
STATES = KNOWN_STATES


def parse_aadhaar_address(address_block):
    result = {"flat_no":"","building":"","road":"","locality":"","city":"","district":"","residentState":"","pincode":""}
    if not address_block: return result
    raw_lines = re.split(r"[\n,]+", address_block)
    lines = [l.strip() for l in raw_lines if l.strip()]
    unmatched = []
    for line in lines:
        m = re.search(r"PIN\s*(?:Code)?[:\s]+(\d{6})", line, re.IGNORECASE)
        if m: result["pincode"] = m.group(1); continue
        m = re.match(r"State\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if m: result["residentState"] = m.group(1).strip().title(); continue
        m = re.match(r"District\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if m:
            result["district"] = m.group(1).strip().title()
            if not result["city"]: result["city"] = result["district"]
            continue
        m = re.match(r"VTC\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if m:
            vtc = re.sub(r"\s*[,.]?\s*S\.?\s*O\.?\.?\s*$","",m.group(1),flags=re.IGNORECASE).strip()
            vtc = re.sub(r"\s*[,.]?\s*P\.?\s*O\.?\.?\s*$","",vtc,flags=re.IGNORECASE).strip()
            if not result["locality"]: result["locality"] = vtc.title()
            if not result["city"]: result["city"] = vtc.title()
            continue
        unmatched.append(line)
    oneline = re.sub(r"\s*\n\s*", ", ", address_block.strip())
    compact = re.search(r"([A-Za-z][A-Za-z\s]{1,30}?),\s*([A-Za-z][A-Za-z\s]{1,25}?),\s*([A-Za-z][A-Za-z\s]{3,25}?)\s*[-–]\s*(\d{6})", oneline)
    if compact:
        loc_part = compact.group(1).strip().title()
        dist_part = compact.group(2).strip().title()
        state_part = compact.group(3).strip().title()
        pin_part = compact.group(4).strip()
        if not result["locality"]: result["locality"] = loc_part
        if not result["city"]: result["city"] = dist_part
        if not result["district"]: result["district"] = dist_part
        if not result["residentState"] and state_part in KNOWN_STATES: result["residentState"] = state_part
        if not result["pincode"]: result["pincode"] = pin_part
    if not result["locality"] and result["city"]: result["locality"] = result["city"]
    return result


# ════════════════════════════════════════
# AUTH
# ════════════════════════════════════════
@app.route("/register", methods=["POST"])
def register():
    d = request.json
    email = d.get("email","").strip().lower()
    phone = d.get("phone","").strip()
    pw = d.get("password","").strip()
    name = d.get("name","").strip()
    if not email or not phone or not pw:
        return jsonify({"error":"Email, phone and password are required"}),400
    ex = users_col.find_one({"$or":[{"email":email},{"phone":phone}]})
    if ex:
        field = "Email" if ex.get("email")==email else "Phone"
        return jsonify({"error":f"{field} already registered. Please login."}),409
    r = users_col.insert_one({"name":name,"email":email,"phone":phone,"password":hash_pw(pw),"created_at":datetime.now(timezone.utc),"last_login":None,"login_count":0,"applications":[]})
    return jsonify({"status":"registered","userId":str(r.inserted_id),"name":name,"email":email,"phone":phone,"message":"Account created successfully!"})


@app.route("/login", methods=["POST"])
def login():
    d = request.json
    email = d.get("email","").strip().lower()
    phone = d.get("phone","").strip()
    pw = d.get("password","").strip()
    if not pw: return jsonify({"error":"Password is required"}),400
    q = {"password":hash_pw(pw)}
    if email and phone: q["$or"]=[{"email":email},{"phone":phone}]
    elif email: q["email"]=email
    elif phone: q["phone"]=phone
    else: return jsonify({"error":"Enter email or phone"}),400
    u = users_col.find_one(q)
    if not u: return jsonify({"error":"Invalid credentials."}),401
    users_col.update_one({"_id":u["_id"]},{"$set":{"last_login":datetime.now(timezone.utc)},"$inc":{"login_count":1}})
    cnt = u.get("login_count",0)
    return jsonify({"status":"success","userId":str(u["_id"]),"name":u.get("name",""),"email":u.get("email",""),"phone":u.get("phone",""),"isReturning":cnt>0,"loginCount":cnt,"message":f"Welcome back, {u.get('name','')}!" if cnt>0 else f"Welcome, {u.get('name','')}!"})


# ════════════════════════════════════════
# DEBUG endpoint — see exactly what was extracted
# ════════════════════════════════════════
@app.route("/debug-last", methods=["GET"])
def debug_last():
    return jsonify(_last_debug)


# ════════════════════════════════════════
# PROCESS
# ════════════════════════════════════════
@app.route("/process", methods=["POST"])
def process_aadhaar():
    global _last_debug
    t0 = time.time()
    af = request.files.get("aadhaar")
    pf = request.files.get("photo")
    sf = request.files.get("signature")
    if not af: return jsonify({"error":"No aadhaar file"}),400

    login_name = request.form.get("login_name","").strip()
    mime = af.mimetype or ""
    ext = ".pdf" if "pdf" in mime else (".png" if "png" in mime else ".jpg")
    ap = os.path.join(UPLOAD_FOLDER, f"aadhaar{ext}")
    pp = os.path.join(UPLOAD_FOLDER, "photo.jpg")
    sp = os.path.join(UPLOAD_FOLDER, "signature.jpg")
    af.save(ap)
    if pf: pf.save(pp)
    if sf: sf.save(sp)

    data, method = fast_extract(ap, mime)

    # ── Name resolution ───────────────────────────────────
    if login_name:
        parts = login_name.strip().split()
        data["first_name"]  = parts[0].title() if len(parts) >= 1 else ""
        data["last_name"]   = parts[-1].title() if len(parts) > 1 else ""
        data["middle_name"] = parts[1].title() if len(parts) == 3 else ""
    else:
        for field in ["first_name","last_name","middle_name"]:
            if is_bad_name(data.get(field,"")): data[field] = ""

    # ── AO Code ───────────────────────────────────────────
    ao = lookup_ao_code(data.get("locality",""), data.get("city",""), data.get("district",""), data.get("residentState",""))
    data.update(ao)

    elapsed = round(time.time() - t0, 2)
    print(f"\n{'='*60}")
    print(f"✅ TOTAL TIME : {elapsed}s | Method: {method}")
    print(f"   gender     : '{data.get('gender')}'")
    print(f"   dob        : '{data.get('dob')}'")
    print(f"   confidence : {calc_confidence(data)}")
    print(f"{'='*60}\n")

    _last_debug = {
        "elapsed": elapsed, "method": method,
        "gender": data.get("gender",""), "dob": data.get("dob",""),
        "raw_ocr_snippet": data.get("raw_ocr_text","")[:500],
        "all_fields": {k: v for k, v in data.items() if k != "raw_ocr_text" and v},
    }

    data.update({
        "aadhaar_path": ap, "photo_path": pp if pf else "", "signature_path": sp if sf else "",
        "extraction_method": method, "confidence": calc_confidence(data),
        "verified": False, "created_at": datetime.now(timezone.utc)
    })
    ins = collection.insert_one(data)
    data["_id"] = str(ins.inserted_id)
    return jsonify(data)


# ════════════════════════════════════════
# FAST EXTRACTION PIPELINE
# ════════════════════════════════════════
def fast_extract(file_path, mime):
    print(f"\n🚀 Fast parallel extraction starting...")
    ocr_text = ""

    def run_ocr_fast():
        nonlocal ocr_text
        t = time.time()
        try:
            ocr_text = quick_ocr(file_path, mime)
            print(f"  ⚡ OCR done in {round(time.time()-t,2)}s ({len(ocr_text)} chars)")
            # Print first 300 chars so we can see what OCR got
            print(f"  📄 OCR preview: {repr(ocr_text[:300])}")
        except Exception as e:
            print(f"  ⚠️ OCR error: {e}")

    def run_grok_vision():
        t = time.time()
        try:
            result = grok_extract_with_vision(file_path, mime)
            print(f"  ⚡ Grok done in {round(time.time()-t,2)}s")
            return result
        except Exception as e:
            print(f"  ⚠️ Grok error: {e}")
            return None

    ocr_future = _executor.submit(run_ocr_fast)
    grok_future = _executor.submit(run_grok_vision)

    ocr_future.result()   # wait for OCR (needed for fallback)
    grok_data = grok_future.result()

    # ── Grok result ───────────────────────────────────────
    if grok_data:
        # ── ALWAYS patch gender + DOB from OCR if Grok missed them ──
        if not grok_data.get("gender"):
            grok_data["gender"] = extract_gender(ocr_text)
            if grok_data["gender"]:
                print(f"  🔧 Gender patched from OCR: {grok_data['gender']}")

        if not grok_data.get("dob"):
            grok_data["dob"] = extract_dob(ocr_text)
            if grok_data["dob"]:
                print(f"  🔧 DOB patched from OCR: {grok_data['dob']}")

        # Patch address fields
        if not grok_data.get("district") or not grok_data.get("locality"):
            addr_block = extract_address_block_from_ocr(ocr_text)
            if addr_block:
                parsed = parse_aadhaar_address(addr_block)
                for k, v in parsed.items():
                    if v and not grok_data.get(k,"").strip():
                        grok_data[k] = v

        grok_data["raw_ocr_text"] = ocr_text
        conf = calc_confidence(grok_data)

        if conf >= 0.4:
            return grok_data, "grok_vision+ocr_patch"
        else:
            print(f"  ⚠️ Low Grok confidence ({conf}) — filling gaps from OCR")
            base = grok_data
            return regex_fill_gaps(base, ocr_text), "grok_vision+ocr_fallback"

    # ── Pure OCR fallback ─────────────────────────────────
    print("\n🔍 Pure OCR+regex fallback...")
    base = make_empty()
    base["raw_ocr_text"] = ocr_text

    # Apply dedicated extractors first
    base["gender"] = extract_gender(ocr_text)
    base["dob"]    = extract_dob(ocr_text)

    addr_block = extract_address_block_from_ocr(ocr_text)
    if addr_block:
        parsed = parse_aadhaar_address(addr_block)
        for k, v in parsed.items():
            if v: base[k] = v

    result = regex_fill_gaps(base, ocr_text)
    return result, "ocr_regex_fallback"


# ════════════════════════════════════════
# FAST OCR
# ════════════════════════════════════════
_OCR_TARGET_WIDTH = 1200

def quick_ocr(file_path, mime):
    is_pdf = "pdf" in mime or file_path.lower().endswith(".pdf")
    if is_pdf:
        if not PDF_SUPPORT: raise Exception("Run: pip install pymupdf")
        import io
        doc = fitz.open(file_path)
        texts = []
        for page in doc:
            direct = page.get_text("text").strip()
            if len(direct) > 20:
                texts.append(direct)
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                texts.append(_ocr_image_fast(img))
        doc.close()
        return "\n\n".join(texts)
    else:
        img = Image.open(file_path).convert("RGB")
        return _ocr_image_fast(img)


def _ocr_image_fast(pil_image):
    w, h = pil_image.size
    if w < _OCR_TARGET_WIDTH:
        scale = _OCR_TARGET_WIDTH / w
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr  = np.array(pil_image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run TWO configs and merge — psm 4 for layout, psm 6 as backup
    cfg_fast = "--oem 1 --psm 4 -l eng"
    cfg_back = "--oem 1 --psm 6 -l eng"
    try:
        t1 = pytesseract.image_to_string(binary, config=cfg_fast)
        t2 = pytesseract.image_to_string(binary, config=cfg_back)
        # Merge: use whichever has more content, append unique lines from the other
        lines1 = set(t1.strip().split("\n"))
        lines2 = set(t2.strip().split("\n"))
        # Pick t1 as base (psm 4 usually better), add any unique non-empty lines from t2
        extra = [l for l in t2.split("\n") if l.strip() and l not in lines1]
        merged = t1.strip() + ("\n" + "\n".join(extra) if extra else "")
        return merged
    except Exception as e:
        print(f"    ⚠️ OCR error: {e}")
        return ""


def extract_address_block_from_ocr(ocr_text):
    lines = ocr_text.split("\n")
    start_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"Address\s*:", line, re.IGNORECASE):
            start_idx = i; break
    if start_idx == -1:
        for i, line in enumerate(lines):
            if re.search(r"VTC\s*[:\-]|District\s*[:\-]|PIN\s*Code", line, re.IGNORECASE):
                start_idx = max(0, i - 4); break
    if start_idx == -1: return ""
    block = []
    for line in lines[start_idx:start_idx+12]:
        line = line.strip()
        if not line: continue
        if re.search(r'[\u0900-\u097F]', line): continue
        block.append(line)
    return "\n".join(block)


# ════════════════════════════════════════
# GROK VISION — improved prompt for gender/DOB
# ════════════════════════════════════════
def grok_extract_with_vision(file_path, mime):
    import io
    is_pdf = "pdf" in mime or file_path.lower().endswith(".pdf")
    if is_pdf:
        if not PDF_SUPPORT: raise Exception("Run: pip install pymupdf")
        doc = fitz.open(file_path)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        doc.close()
        img_mime = "image/png"
    else:
        with open(file_path, "rb") as f: img_bytes = f.read()
        img_mime = "image/jpeg" if file_path.lower().endswith(".jpg") else "image/png"

    img_b64  = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{img_mime};base64,{img_b64}"

    prompt = """You are reading an Indian Aadhaar card image. Extract ALL visible fields carefully.

Return ONLY valid JSON (no markdown, no explanation):
{
  "first_name": "",
  "middle_name": "",
  "last_name": "",
  "dob": "",
  "gender": "",
  "aadhaar_number": "",
  "father_first_name": "",
  "father_last_name": "",
  "flat_no": "",
  "building": "",
  "road": "",
  "locality": "",
  "city": "",
  "district": "",
  "residentState": "",
  "pincode": "",
  "address": "",
  "email": "",
  "mobile": ""
}

CRITICAL RULES — read carefully:

GENDER:
- Look for the word "MALE" or "FEMALE" printed on the card (usually near DOB)
- Also check "Gender: Male/Female" or "Sex: Male/Female"  
- Return exactly "Male" or "Female" (capitalised) — never ""  if you can see it
- If the card shows "FEMALE", return "Female". If "MALE", return "Male".

DOB (Date of Birth):
- Look for "DOB:", "Date of Birth:", or a date near those words
- Format printed on Aadhaar is usually DD/MM/YYYY (e.g. 15/02/2005)
- Return in YYYY-MM-DD format (e.g. "2005-02-15")
- Never return "" if a date is visible

AADHAAR NUMBER:
- 12-digit number, usually shown as "XXXX XXXX XXXX"
- NOT the 16-digit VID number

NAME:
- The cardholder's name is the large bold text in English near the top
- Father/guardian name follows S/O (Son Of), D/O (Daughter Of), W/O (Wife Of), or C/O
- Example: "D/O: Jose Jacob," → father_first_name="Jose", father_last_name="Jacob"
- Example: "S/O Rahul Kumar" → father_first_name="Rahul", father_last_name="Kumar"
- DO NOT include the comma or address text after the name
- father_first_name and father_last_name are DIFFERENT people from first_name/last_name

ADDRESS:
- Include full English address in "address" field
- Parse into: flat_no, building, road, locality, city, district, residentState, pincode
- Pincode is 6 digits

Return "" only for fields truly not visible on the card."""

    response = grok_client.chat.completions.create(
        model=GROK_MODEL,
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":data_url,"detail":"high"}},
            {"type":"text","text":prompt},
        ]}],
        max_tokens=1000,
        temperature=0,
    )

    raw     = response.choices[0].message.content.strip()
    cleaned = re.sub(r"```(?:json)?\s*|\s*```","",raw).strip()
    result  = json.loads(cleaned)

    print(f"  📋 Grok raw: gender='{result.get('gender')}' dob='{result.get('dob')}'")

    # Title-case ALL-CAPS names
    for field in ["first_name","middle_name","last_name","father_first_name","father_last_name"]:
        v = result.get(field,"")
        if v and v.isupper(): result[field] = v.title()

    # Discard garbage names
    for nf in ["first_name","last_name","middle_name"]:
        if is_bad_name(result.get(nf,"")): result[nf] = ""

    # Normalise gender capitalisation
    g = result.get("gender","").strip()
    if g.lower() == "female":   result["gender"] = "Female"
    elif g.lower() == "male":   result["gender"] = "Male"
    else:                        result["gender"] = ""

    # Normalise DOB to YYYY-MM-DD
    dob = result.get("dob","").strip()
    m = re.match(r'^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$', dob)
    if m:
        result["dob"] = f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    # Parse address
    raw_addr = result.get("address","")
    if raw_addr:
        parsed = parse_aadhaar_address(raw_addr)
        for field, val in parsed.items():
            if val and not result.get(field,"").strip(): result[field] = val

    return result


# ════════════════════════════════════════
# REGEX FALLBACK
# ════════════════════════════════════════
NAME_SKIP = {"GOVERNMENT","OF","INDIA","AADHAAR","UNIQUE","IDENTIFICATION","AUTHORITY","DOWNLOAD",
             "ENROLMENT","ISSUED","VID","VIRTUAL","ADDRESS","BIRTH","HELPLINE","BACK","SIDE","DATE",
             "DOB","MALE","FEMALE","GENDER","MOBILE","LINKED","THIS","CARD","DIGITALLY","VERIFIABLE",
             "DOCUMENT","RESIDENT","AADHAR","SCAN","FOR","DETAILS","AND","THE","BY","AN","IS"}


def regex_fill_gaps(data, text):
    def missing(key): return not data.get(key,"").strip()

    # ── Gender + DOB — use dedicated extractors ───────────
    if missing("gender"):
        data["gender"] = extract_gender(text)

    if missing("dob"):
        data["dob"] = extract_dob(text)

    # ── Aadhaar number ─────────────────────────────────────
    if missing("aadhaar_number"):
        for pattern in [r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b", r"\b(\d{4})[-–](\d{4})[-–](\d{4})\b"]:
            for m in re.finditer(pattern, text):
                after = text[m.end():m.end()+8].strip()
                if re.match(r"^\d{4}\b", after): continue
                data["aadhaar_number"] = f"{m.group(1)} {m.group(2)} {m.group(3)}"
                break
            if not missing("aadhaar_number"): break

    # ── PIN ────────────────────────────────────────────────
    if missing("pincode"):
        m = re.search(r"PIN\s*(?:Code)?[:\s]+(\d{6})", text, re.IGNORECASE)
        if not m: m = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
        if m: data["pincode"] = m.group(1)

    # ── District ───────────────────────────────────────────
    if missing("district"):
        m = re.search(r"District\s*[:\-]\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)", text, re.IGNORECASE)
        if m: data["district"] = m.group(1).strip().title()

    # ── State ──────────────────────────────────────────────
    if missing("residentState"):
        for s in STATES:
            if re.search(r"\b" + re.escape(s) + r"\b", text, re.IGNORECASE):
                data["residentState"] = s; break

    # ── Locality ───────────────────────────────────────────
    if missing("locality"):
        m = re.search(r"VTC\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]+?)(?:\s+S\.?O\.?)?(?:,|\n|$)", text, re.IGNORECASE)
        if m: data["locality"] = re.sub(r"\s*S\.?O\.?\s*$","",m.group(1)).strip().title()

    # ── City ───────────────────────────────────────────────
    if missing("city"):
        if data.get("locality"): data["city"] = data["locality"]
        elif data.get("district"): data["city"] = data["district"]

    # ── Compact address scan ───────────────────────────────
    if missing("locality") or missing("city"):
        compact = re.search(r"([A-Za-z][A-Za-z\s]{1,30}?),\s*([A-Za-z][A-Za-z\s]{1,25}?),\s*([A-Za-z][A-Za-z\s]{3,25}?)\s*[-–]\s*(\d{6})", text)
        if compact:
            loc_p = compact.group(1).strip().title(); dist_p = compact.group(2).strip().title()
            state_p = compact.group(3).strip().title(); pin_p = compact.group(4).strip()
            if missing("locality"):      data["locality"]      = loc_p
            if missing("city"):          data["city"]          = dist_p
            if missing("district"):      data["district"]      = dist_p
            if missing("residentState") and state_p in STATES: data["residentState"] = state_p
            if missing("pincode"):       data["pincode"]       = pin_p

    # ── Father name ────────────────────────────────────────
    if missing("father_first_name"):
        # Strategy: find D/O, S/O, W/O followed by a REAL name (>=3 chars per word)
        # The OCR produces garbage like "CSO/oes" — we reject single short words
        # "D/O: Jose Jacob, Poovathussery House..." → stop at comma, take "Jose Jacob"
        father_m = re.search(
            r"(?:D\s*/\s*O|S\s*/\s*O|W\s*/\s*O|C\s*/\s*O)"  # prefix
            r"\s*[:\-]?\s*"                                     # optional separator
            r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{1,})*)",        # Name: each word Title-case, min 3 chars
            text
        )
        if not father_m:
            # Fallback: case-insensitive, name words must be >= 3 chars
            father_m = re.search(
                r"(?:D\.?/O\.?|S\.?/O\.?|W\.?/O\.?|C\.?/O\.?)"
                r"\s*[:\-]?\s*"
                r"([A-Za-z]{3,}(?:\s+[A-Za-z]{2,})*)",
                text, re.IGNORECASE
            )
        if father_m:
            raw = father_m.group(1).strip()
            # Cut off at comma (address follows) or known address keywords
            raw = raw.split(",")[0].strip()
            raw = re.sub(
                r"\s+(House|Flat|Poova|Nagar|Village|Post|Road|Street|Industrial).*$",
                "", raw, flags=re.IGNORECASE
            ).strip()
            parts = [p for p in raw.split() if len(p) >= 2 and p.replace(".", "").isalpha()][:3]
            if parts and len(parts[0]) >= 3:   # reject garbage like "Oes"
                data["father_first_name"] = parts[0].title()
                data["father_last_name"]  = " ".join(p.title() for p in parts[1:]) if len(parts) > 1 else ""
                print(f"  ✅ Father: {data['father_first_name']} {data['father_last_name']}")
    # ── Address fallback ───────────────────────────────────
    if missing("address"):
        pts = [data.get(k,"") for k in ["flat_no","building","road","locality","city","district","residentState","pincode"]]
        data["address"] = ", ".join(p for p in pts if p)

    return data


# ════════════════════════════════════════
# CONFIRM
# ════════════════════════════════════════
@app.route("/confirm/<id>", methods=["PUT"])
def confirm_data(id):
    upd = request.json
    uid = upd.pop("userId", None)
    upd.pop("_id", None); upd.pop("created_at", None)
    orig = collection.find_one({"_id": ObjectId(id)})
    if orig:
        corrections = {}
        for f in ["first_name","last_name","middle_name","dob","gender","aadhaar_number","locality","city","district","residentState","pincode","father_first_name","father_last_name","flat_no","building","road","areaCode","aoType","rangeCode","aoNumber"]:
            o, u = orig.get(f,""), upd.get(f,"")
            if o != u and u: corrections[f] = {"original":o,"corrected":u}
        if corrections:
            training_col.insert_one({"ocr_text":orig.get("raw_ocr_text",""),"extraction_method":orig.get("extraction_method","unknown"),"corrections":corrections,"created_at":datetime.now(timezone.utc)})
    collection.update_one({"_id":ObjectId(id)},{"$set":{**upd,"verified":True,"confirmed_at":datetime.now(timezone.utc)}})
    if uid:
        try: users_col.update_one({"_id":ObjectId(uid)},{"$push":{"applications":id}})
        except: pass
    return jsonify({"status":"confirmed"})


def calc_confidence(d):
    s = 0
    if d.get("aadhaar_number"): s += 0.25
    if d.get("dob"):            s += 0.20
    if d.get("first_name"):     s += 0.15
    if d.get("last_name"):      s += 0.10
    if d.get("gender"):         s += 0.10
    if d.get("city") or d.get("locality"): s += 0.10
    if d.get("pincode"):        s += 0.05
    if d.get("residentState"):  s += 0.05
    return round(s, 2)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)