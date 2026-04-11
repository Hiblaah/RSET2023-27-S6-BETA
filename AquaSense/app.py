"""
AquaSense Pro — Flask Backend v4
==================================
Place in same folder as bod_model.pkl, bod_scaler.pkl, model_meta.json, index.html
Run: python app.py  →  Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory, Response
import numpy as np
import joblib
import json
import os
import csv
import io
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='.')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')
USERS_FILE   = os.path.join(BASE_DIR, 'users.json')

# ── Load model ────────────────────────────────────────────────────────────
try:
    model  = joblib.load(os.path.join(BASE_DIR, 'bod_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'bod_scaler.pkl'))
    with open(os.path.join(BASE_DIR, 'model_meta.json')) as f:
        meta = json.load(f)
    USE_LOG  = meta['use_log']
    FEATURES = meta['features']
    print(f"✅ Model loaded  |  R²={meta['r2_score']}  |  Log transform: {USE_LOG}")
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    model = scaler = None

# ── Parameter limits ──────────────────────────────────────────────────────
PARAM_LIMITS = {
    'pH'          : {'impossible': (0,   14),   'normal': (4.0,  10.0),  'unit': ''},
    'Temperature' : {'impossible': (0,   50),   'normal': (5.0,  40.0),  'unit': '°C'},
    'Conductivity': {'impossible': (0,   50000),'normal': (10.0, 5000.0),'unit': 'µS/cm'},
    'DO'          : {'impossible': (0,   18),   'normal': (0.5,  14.0),  'unit': 'mg/L'},
    'Nitrate'     : {'impossible': (0,   1000), 'normal': (0.0,  50.0),  'unit': 'mg/L'},
}

# ── User helpers ──────────────────────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# ── History helpers ───────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# ── Consecutive day alert check ───────────────────────────────────────────
def check_consecutive_day_alert(history, location):
    loc_history = [h for h in history if h.get('location') == location and h.get('level', 0) >= 3]
    if len(loc_history) < 2:
        return None
    bad_dates = sorted(set(h['date'] for h in loc_history), reverse=True)
    if len(bad_dates) < 2:
        return None
    consecutive_days = 1
    for i in range(1, len(bad_dates)):
        d1 = datetime.strptime(bad_dates[i-1], '%Y-%m-%d')
        d2 = datetime.strptime(bad_dates[i],   '%Y-%m-%d')
        if (d1 - d2).days == 1:
            consecutive_days += 1
        else:
            break
    if consecutive_days >= 2:
        worst_level   = max(h['level'] for h in loc_history if h['date'] in bad_dates[:consecutive_days])
        quality_map   = {3:'Fair', 4:'Poor', 5:'Very Poor'}
        worst_quality = quality_map.get(worst_level, 'Fair')
        return {
            'message'         : f'⚠️ {location} has had {worst_quality} or worse water quality for {consecutive_days} consecutive day(s). Investigation recommended.',
            'consecutive_days': consecutive_days,
            'worst_quality'   : worst_quality,
            'level'           : worst_level,
            'location'        : location,
        }
    return None


# ── Serve GUI ─────────────────────────────────────────────────────────────
@app.route('/')
def serve_gui():
    return send_from_directory(BASE_DIR, 'index.html')


# ── Register ──────────────────────────────────────────────────────────────
@app.route('/register', methods=['POST'])
def register():
    data  = request.get_json()
    uname = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    pwd   = data.get('password') or ''

    if not uname or not email or not pwd:
        return jsonify({'error': 'All fields are required'}), 400

    users = load_users()
    if any(u['username'] == uname for u in users):
        return jsonify({'error': 'Username already exists'}), 400
    if any(u['email'] == email for u in users):
        return jsonify({'error': 'Email already registered'}), 400

    users.append({'username': uname, 'email': email, 'password': pwd})
    save_users(users)
    print(f"✅ New user registered: {uname}")
    return jsonify({'success': True, 'message': 'Account created successfully'})


# ── Login ─────────────────────────────────────────────────────────────────
@app.route('/login', methods=['POST'])
def login():
    data  = request.get_json()
    uname = (data.get('username') or '').strip()
    pwd   = data.get('password') or ''

    users = load_users()
    user  = next((u for u in users if u['username'] == uname and u['password'] == pwd), None)
    if not user:
        return jsonify({'error': 'Invalid username or password'}), 401

    print(f"✅ User logged in: {uname}")
    return jsonify({'success': True, 'username': user['username'], 'email': user['email']})


# ── Get all users (for email picker — passwords never sent) ───────────────
@app.route('/users', methods=['GET'])
def get_users():
    users = load_users()
    return jsonify([{'username': u['username'], 'email': u['email']} for u in users])


# ── Predict ───────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server console.'}), 500

    data = request.get_json()
    required = ['pH', 'Temperature', 'Conductivity', 'DO', 'Nitrate']
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        values = {
            'pH'          : float(data['pH']),
            'Temperature' : float(data['Temperature']),
            'Conductivity': float(data['Conductivity']),
            'DO'          : float(data['DO']),
            'Nitrate'     : float(data['Nitrate']),
        }
    except (ValueError, TypeError):
        return jsonify({'error': 'All inputs must be valid numbers'}), 400

    # Block impossible values
    impossible_errors = []
    for field, val in values.items():
        lo, hi = PARAM_LIMITS[field]['impossible']
        if not (lo <= val <= hi):
            impossible_errors.append(
                f'{field} = {val} is physically impossible (valid range: {lo}–{hi} {PARAM_LIMITS[field]["unit"]})'
            )
    if impossible_errors:
        return jsonify({'error': 'Invalid sensor values:\n' + '\n'.join(impossible_errors)}), 400

    # Warn for outside normal range
    warnings = []
    for field, val in values.items():
        lo, hi = PARAM_LIMITS[field]['normal']
        if not (lo <= val <= hi):
            warnings.append(
                f'{field} = {val} is outside normal river range ({lo}–{hi} {PARAM_LIMITS[field]["unit"]}). '
                f'Prediction may be less accurate.'
            )

    # Predict
    X_input  = np.array([[values['pH'], values['Temperature'], values['Conductivity'], values['DO'], values['Nitrate']]])
    X_scaled = scaler.transform(X_input)
    pred     = model.predict(X_scaled)[0]
    if USE_LOG:
        pred = np.expm1(pred)
    pred = max(0.0, float(pred))

    # Classify
    if pred < 2:
        quality, level = 'Excellent', 1
    elif pred < 4:
        quality, level = 'Good', 2
    elif pred < 8:
        quality, level = 'Fair', 3
    elif pred < 20:
        quality, level = 'Poor', 4
    else:
        quality, level = 'Very Poor', 5

    # Save to history
    now   = datetime.now()
    entry = {
        'id'          : now.strftime('%Y%m%d%H%M%S%f'),
        'date'        : now.strftime('%Y-%m-%d'),
        'time'        : now.strftime('%H:%M:%S'),
        'datetime'    : now.strftime('%Y-%m-%d %H:%M:%S'),
        'username'    : data.get('username', 'unknown'),
        'location'    : data.get('location', 'Chitrapuzha River — Near College'),
        'pH'          : values['pH'],
        'Temperature' : values['Temperature'],
        'Conductivity': values['Conductivity'],
        'DO'          : values['DO'],
        'Nitrate'     : values['Nitrate'],
        'Turbidity'   : data.get('turbidity'),
        'BOD'         : round(pred, 2),
        'quality'     : quality,
        'level'       : level,
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    consec_alert = check_consecutive_day_alert(history, entry['location'])

    return jsonify({
        'bod'              : round(pred, 2),
        'quality'          : quality,
        'level'            : level,
        'model_r2'         : meta['r2_score'],
        'unit'             : 'mg/L',
        'warnings'         : warnings,
        'consecutive_alert': consec_alert,
        'entry'            : entry,
    })


# ── History ───────────────────────────────────────────────────────────────
@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(load_history())


# ── Download CSV ──────────────────────────────────────────────────────────
@app.route('/download', methods=['GET'])
def download_csv():
    start_dt = request.args.get('start')
    end_dt   = request.args.get('end')
    history  = load_history()

    if start_dt or end_dt:
        filtered = []
        for h in history:
            hdt = h.get('datetime', '')
            if start_dt and hdt < start_dt:
                continue
            if end_dt and hdt > end_dt + ':59':
                continue
            filtered.append(h)
    else:
        filtered = history

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Time', 'User', 'Location', 'pH', 'Temperature (°C)',
                     'Conductivity (µS/cm)', 'DO (mg/L)', 'Nitrate (mg/L)',
                     'Turbidity (NTU)', 'BOD (mg/L)', 'Water Quality'])
    for h in filtered:
        writer.writerow([
            h.get('date',''), h.get('time',''), h.get('username',''),
            h.get('location',''), h.get('pH',''), h.get('Temperature',''),
            h.get('Conductivity',''), h.get('DO',''), h.get('Nitrate',''),
            h.get('Turbidity',''), h.get('BOD',''), h.get('quality','')
        ])

    output.seek(0)
    fname = f"aquasense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={fname}'})


# ── Status ────────────────────────────────────────────────────────────────
@app.route('/status')
def status():
    history = load_history()
    return jsonify({
        'model_loaded' : model is not None,
        'r2_score'     : meta.get('r2_score') if model else None,
        'total_records': len(history),
    })


if __name__ == '__main__':
    print("\n🌊 AquaSense Pro — Starting server...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
