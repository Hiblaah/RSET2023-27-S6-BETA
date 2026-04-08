import os, random, csv
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin,
                         login_user, logout_user, current_user, login_required)
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()


class Config:
    SECRET_KEY              = os.getenv("SECRET_KEY", "nutricalc-secret-2024")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///nutricalc.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Required for Flutter app cookies to work cross-origin
    SESSION_COOKIE_SAMESITE = "None"
    SESSION_COOKIE_SECURE   = False   # Set True if using HTTPS


db            = SQLAlchemy()
login_manager = LoginManager()

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foods.csv")
DAYS     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── MODELS ──────────────────────────────────────────────────────────────────

class User(db.Model, UserMixin):
    __tablename__        = "users"
    user_id              = db.Column(db.Integer, primary_key=True)
    full_name            = db.Column(db.String(255))
    email                = db.Column(db.String(255), unique=True)
    password_hash        = db.Column(db.String(255))
    food_type            = db.Column(db.String(10), default="nonveg")   # 'veg' or 'nonveg'
    age                  = db.Column(db.Integer)
    gender               = db.Column(db.String(10))
    height               = db.Column(db.Float)
    weight               = db.Column(db.Float)
    activity_level       = db.Column(db.String(50))
    body_goal            = db.Column(db.String(50))
    maintenance_calories = db.Column(db.Float)
    target_calories      = db.Column(db.Float)
    status               = db.Column(db.String(20), default="active")
    last_login           = db.Column(db.DateTime, nullable=True)

    def get_id(self):            return f"u_{self.user_id}"
    def set_password(self, p):   self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)

    @property
    def is_admin(self): return False


class Admin(db.Model, UserMixin):
    __tablename__ = "admins"
    admin_id      = db.Column(db.Integer, primary_key=True)
    full_name     = db.Column(db.String(255))
    email         = db.Column(db.String(255), unique=True)
    password_hash = db.Column(db.String(255))
    last_login    = db.Column(db.DateTime, nullable=True)

    def get_id(self):            return f"a_{self.admin_id}"
    def set_password(self, p):   self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)

    @property
    def is_admin(self): return True


class MealPlan(db.Model):
    __tablename__        = "meal_plans"
    plan_id              = db.Column(db.Integer, primary_key=True)
    user_id              = db.Column(db.Integer)
    day_of_week          = db.Column(db.String(10))
    breakfast_name       = db.Column(db.String(255))
    breakfast_calories   = db.Column(db.Float)
    breakfast_protein    = db.Column(db.Float)
    breakfast_carbs      = db.Column(db.Float)
    breakfast_fat        = db.Column(db.Float)
    lunch_name           = db.Column(db.String(255))
    lunch_calories       = db.Column(db.Float)
    lunch_protein        = db.Column(db.Float)
    lunch_carbs          = db.Column(db.Float)
    lunch_fat            = db.Column(db.Float)
    dinner_name          = db.Column(db.String(255))
    dinner_calories      = db.Column(db.Float)
    dinner_protein       = db.Column(db.Float)
    dinner_carbs         = db.Column(db.Float)
    dinner_fat           = db.Column(db.Float)
    daily_total_calories = db.Column(db.Float)


# ── FOOD LOADING ─────────────────────────────────────────────────────────────

def load_foods_from_csv(food_type="nonveg"):
    """
    Load foods from foods.csv.
    Expects columns: Dish Name, Calories (kcal), Protein (g),
                     Carbohydrates (g), Fats (g), Fibre (g), Sodium (mg)

    Optional column: Type  →  'veg' or 'nonveg'
    If the column exists, filter by food_type.
    VEG users: also exclude mushrooms.
    NON-VEG users: also exclude seafood keywords.
    """
    if not os.path.exists(CSV_PATH):
        return []

    SEAFOOD_KEYWORDS = ["prawn", "fish", "crab", "lobster", "shrimp",
                        "salmon", "tuna", "squid", "mackerel", "sardine",
                        "pomfret", "hilsa", "tilapia", "catfish", "anchovy",
                        "clam", "mussel", "oyster", "scallop", "kingfish"]
    MUSHROOM_KEYWORDS = ["mushroom"]

    foods = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        has_type_col = "Type" in headers

        for row in reader:
            name = row.get("Dish Name", "").strip()
            if not name:
                continue

            # Filter by CSV Type column if present
            if has_type_col:
                row_type = row.get("Type", "").strip().lower()
                if row_type and row_type != food_type:
                    continue

            name_lower = name.lower()

            # Apply dietary restriction rules
            if food_type == "veg":
                if any(kw in name_lower for kw in MUSHROOM_KEYWORDS):
                    continue
            else:  # nonveg
                if any(kw in name_lower for kw in SEAFOOD_KEYWORDS):
                    continue

            def g(col):
                try:
                    return float(str(row.get(col, "0") or "0").strip())
                except Exception:
                    return 0.0

            foods.append({
                "name":     name,
                "calories": g("Calories (kcal)"),
                "protein":  g("Protein (g)"),
                "carbs":    g("Carbohydrates (g)"),
                "fat":      g("Fats (g)"),
                "fibre":    g("Fibre (g)"),
                "sodium":   g("Sodium (mg)"),
            })

    return foods


def pick_best_fit(foods, target_kcal, exclude_names, tolerance=0.55):
    lo = target_kcal * (1 - tolerance)
    hi = target_kcal * (1 + tolerance)
    candidates = [f for f in foods if lo <= f["calories"] <= hi
                  and f["name"] not in exclude_names]
    if not candidates:
        candidates = [f for f in foods if f["name"] not in exclude_names]
    if not candidates:
        candidates = foods
    return random.choice(candidates)


def generate_weekly_plan(user):
    MealPlan.query.filter_by(user_id=user.user_id).delete()
    foods = load_foods_from_csv(getattr(user, "food_type", "nonveg") or "nonveg")
    if not foods:
        db.session.commit()
        return

    target   = user.target_calories or 2000
    b_target = target * 0.30
    l_target = target * 0.40
    d_target = target * 0.30

    used_breakfast, used_lunch, used_dinner = [], [], []

    for day in DAYS:
        bf = pick_best_fit(foods, b_target, used_breakfast)
        lu = pick_best_fit(foods, l_target, used_lunch + [bf["name"]])
        di = pick_best_fit(foods, d_target, used_dinner + [bf["name"], lu["name"]])
        used_breakfast.append(bf["name"])
        used_lunch.append(lu["name"])
        used_dinner.append(di["name"])
        total = bf["calories"] + lu["calories"] + di["calories"]
        db.session.add(MealPlan(
            user_id=user.user_id, day_of_week=day,
            breakfast_name=bf["name"], breakfast_calories=bf["calories"],
            breakfast_protein=bf["protein"], breakfast_carbs=bf["carbs"],
            breakfast_fat=bf["fat"],
            lunch_name=lu["name"], lunch_calories=lu["calories"],
            lunch_protein=lu["protein"], lunch_carbs=lu["carbs"],
            lunch_fat=lu["fat"],
            dinner_name=di["name"], dinner_calories=di["calories"],
            dinner_protein=di["protein"], dinner_carbs=di["carbs"],
            dinner_fat=di["fat"],
            daily_total_calories=round(total, 2),
        ))
    db.session.commit()


# ── CALORIE CALC ─────────────────────────────────────────────────────────────

class CalorieCalculator:
    ACTIVITY_MULTIPLIERS = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Active": 1.725,
    }
    GOAL_ADJUSTMENT = {
        "Fat Loss": -400,
        "Maintain Weight": 0,
        "Muscle Gain": 400,
    }

    @staticmethod
    def calculate_bmr(age, gender, height, weight):
        base = (10 * weight) + (6.25 * height) - (5 * age)
        return base + 5 if gender == "Male" else base - 161

    @staticmethod
    def calculate_target(bmr, activity, goal):
        m = bmr * CalorieCalculator.ACTIVITY_MULTIPLIERS.get(activity, 1.2)
        a = CalorieCalculator.GOAL_ADJUSTMENT.get(goal, 0)
        return m, m + a


# ── APP FACTORY ───────────────────────────────────────────────────────────────

def create_app():
    app = Flask(__name__, static_folder="static")
    app.config.from_object(Config)

    # Allow Flutter app (running on phone) to send cookies
    CORS(app, supports_credentials=True, origins="*")

    db.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        db.create_all()
        # ── MIGRATION: add food_type column if upgrading from old DB ──
        try:
            db.session.execute(db.text("SELECT food_type FROM users LIMIT 1"))
        except Exception:
            try:
                db.session.execute(db.text(
                    "ALTER TABLE users ADD COLUMN food_type VARCHAR(10) DEFAULT 'nonveg'"
                ))
                db.session.commit()
                print("Migration applied: added food_type column to users table.")
            except Exception as e:
                db.session.rollback()
                print(f"Migration warning: {e}")

    @login_manager.user_loader
    def load_user(uid):
        if uid.startswith("a_"):
            return db.session.get(Admin, int(uid[2:]))
        if uid.startswith("u_"):
            return db.session.get(User, int(uid[2:]))
        return None

    @login_manager.unauthorized_handler
    def unauthorized():
        return jsonify({"message": "Not authenticated"}), 401

    # ── STATIC PAGES ────────────────────────────────────────────────────────

    @app.route("/")
    def serve_index():
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        return send_from_directory(static_dir, "index.html")

    @app.route("/admin")
    @app.route("/admin/")
    def serve_admin():
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        return send_from_directory(static_dir, "admin.html")

    # ── AUTH ─────────────────────────────────────────────────────────────────

    @app.route("/api/register", methods=["POST"])
    def register():
        d = request.json or {}
        required = ["full_name", "email", "password", "confirm_password"]
        if not all(k in d for k in required):
            return jsonify({"message": "Missing required fields"}), 400
        if d["password"] != d["confirm_password"]:
            return jsonify({"message": "Passwords do not match"}), 400
        if len(d["password"]) < 6:
            return jsonify({"message": "Password must be at least 6 characters"}), 400
        if User.query.filter_by(email=d["email"].strip().lower()).first():
            return jsonify({"message": "Email already registered"}), 409

        food_type = d.get("food_type", "nonveg")
        if food_type not in ("veg", "nonveg"):
            food_type = "nonveg"

        u = User(
            full_name=d["full_name"].strip(),
            email=d["email"].strip().lower(),
            food_type=food_type,
        )
        u.set_password(d["password"])
        db.session.add(u)
        db.session.commit()
        return jsonify({"message": "Registered successfully"}), 201

    @app.route("/api/login", methods=["POST"])
    def login():
        d        = request.json or {}
        email    = d.get("email", "").strip().lower()
        password = d.get("password", "")

        # Check admin first
        a = Admin.query.filter_by(email=email).first()
        if a and a.check_password(password):
            a.last_login = datetime.now(timezone.utc)
            db.session.commit()
            login_user(a)
            return jsonify({
                "message": "Login successful",
                "is_admin": True,
                "redirect": "/admin",
                "user_id": a.admin_id,
                "name": a.full_name,
                "email": a.email,
            })

        u = User.query.filter_by(email=email).first()
        if u and u.check_password(password):
            if u.status == "restricted":
                return jsonify({
                    "message": "Your account has been restricted. Please contact the administrator."
                }), 403
            u.last_login = datetime.now(timezone.utc)
            if u.status == "inactive":
                u.status = "active"
            db.session.commit()
            login_user(u)
            return jsonify({
                "message": "Login successful",
                "is_admin": False,
                "user_id": u.user_id,
                "name": u.full_name,
                "email": u.email,
                "has_profile": u.age is not None,
                "food_type": u.food_type,
            })

        return jsonify({"message": "Invalid email or password"}), 401

    @app.route("/api/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        return jsonify({"message": "Logged out"})

    @app.route("/api/profile", methods=["GET"])
    @login_required
    def get_profile():
        u = current_user
        base = {
            "user_id": u.admin_id if u.is_admin else u.user_id,
            "name": u.full_name,
            "email": u.email,
            "is_admin": u.is_admin,
        }
        if not u.is_admin:
            base.update({
                "age": u.age,
                "gender": u.gender,
                "height": u.height,
                "weight": u.weight,
                "activity_level": u.activity_level,
                "body_goal": u.body_goal,
                "maintenance_calories": u.maintenance_calories,
                "target_calories": u.target_calories,
                "food_type": u.food_type,
                "has_profile": u.age is not None,
            })
        return jsonify(base)

    # ── CALORIE CALCULATION ───────────────────────────────────────────────────

    @app.route("/api/calculate-calories", methods=["POST"])
    @login_required
    def calculate_calories():
        d = request.json or {}
        try:
            age      = int(d["age"])
            gender   = d["gender"]
            height   = float(d["height"])
            weight   = float(d["weight"])
            activity = d["activity_level"]
            goal     = d["body_goal"]
        except (KeyError, ValueError) as e:
            return jsonify({"message": f"Invalid input: {e}"}), 400

        if activity not in CalorieCalculator.ACTIVITY_MULTIPLIERS:
            return jsonify({"message": "Invalid activity level"}), 400
        if goal not in CalorieCalculator.GOAL_ADJUSTMENT:
            return jsonify({"message": "Invalid goal"}), 400

        bmr = CalorieCalculator.calculate_bmr(age, gender, height, weight)
        maintenance, target = CalorieCalculator.calculate_target(bmr, activity, goal)
        adj = CalorieCalculator.GOAL_ADJUSTMENT[goal]

        u = current_user
        u.age = age
        u.gender = gender
        u.height = height
        u.weight = weight
        u.activity_level = activity
        u.body_goal = goal
        u.maintenance_calories = round(maintenance, 2)
        u.target_calories      = round(target, 2)
        db.session.commit()

        return jsonify({
            "bmr": round(bmr, 2),
            "maintenance_calories": round(maintenance, 2),
            "target_calories": round(target, 2),
            "adjustment": adj,
            "goal": goal,
            "meal_distribution": {
                "breakfast": round(target * 0.30, 2),
                "lunch":     round(target * 0.40, 2),
                "dinner":    round(target * 0.30, 2),
            },
        })

    # ── MEAL PLAN ─────────────────────────────────────────────────────────────

    @app.route("/api/meal-plan/generate", methods=["POST"])
    @login_required
    def gen_plan():
        u = current_user
        if not u.target_calories:
            return jsonify({"message": "Calculate calories first"}), 400
        generate_weekly_plan(u)
        return jsonify({"message": "Meal plan generated"})

    @app.route("/api/meal-plan", methods=["GET"])
    @login_required
    def get_plan():
        plans = (MealPlan.query
                 .filter_by(user_id=current_user.user_id)
                 .order_by(MealPlan.plan_id).all())
        if not plans:
            return jsonify([])
        return jsonify([{
            "day": p.day_of_week,
            "breakfast": {
                "name": p.breakfast_name, "calories": p.breakfast_calories,
                "protein": p.breakfast_protein, "carbs": p.breakfast_carbs,
                "fat": p.breakfast_fat,
            },
            "lunch": {
                "name": p.lunch_name, "calories": p.lunch_calories,
                "protein": p.lunch_protein, "carbs": p.lunch_carbs,
                "fat": p.lunch_fat,
            },
            "dinner": {
                "name": p.dinner_name, "calories": p.dinner_calories,
                "protein": p.dinner_protein, "carbs": p.dinner_carbs,
                "fat": p.dinner_fat,
            },
            "total_calories": p.daily_total_calories,
        } for p in plans])

    # ── ADMIN HELPERS ─────────────────────────────────────────────────────────

    def admin_required(f):
        from functools import wraps
        @wraps(f)
        @login_required
        def decorated(*args, **kwargs):
            if not current_user.is_admin:
                return jsonify({"message": "Admin access required"}), 403
            return f(*args, **kwargs)
        return decorated

    def flag_inactive_users():
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        stale = User.query.filter(
            User.status == "active",
            db.or_(User.last_login.is_(None), User.last_login < cutoff)
        ).all()
        for u in stale:
            u.status = "inactive"
        if stale:
            db.session.commit()

    # ── ADMIN ROUTES ──────────────────────────────────────────────────────────

    @app.route("/api/admin/create", methods=["POST"])
    def create_admin():
        """One-time bootstrap endpoint — only works when no admin exists yet."""
        if Admin.query.first():
            return jsonify({"message": "Admin already exists. Use dashboard to add more."}), 403
        d        = request.json or {}
        name     = d.get("full_name", "").strip()
        email    = d.get("email", "").strip().lower()
        password = d.get("password", "")
        if not name or not email or not password:
            return jsonify({"message": "full_name, email and password are required"}), 400
        if not email.endswith("@nutricalc.com"):
            return jsonify({"message": "Admin email must end with @nutricalc.com"}), 400
        if len(password) < 8:
            return jsonify({"message": "Password must be at least 8 characters"}), 400
        if Admin.query.filter_by(email=email).first():
            return jsonify({"message": "Email already registered"}), 409
        a = Admin(full_name=name, email=email)
        a.set_password(password)
        db.session.add(a)
        db.session.commit()
        return jsonify({"message": f"Admin '{name}' created successfully!"}), 201

    @app.route("/api/admin/add-admin", methods=["POST"])
    @admin_required
    def add_admin():
        d        = request.json or {}
        name     = d.get("full_name", "").strip()
        email    = d.get("email", "").strip().lower()
        password = d.get("password", "")
        if not name or not email or not password:
            return jsonify({"message": "All fields are required"}), 400
        if not email.endswith("@nutricalc.com"):
            return jsonify({"message": "Admin email must end with @nutricalc.com"}), 400
        if len(password) < 8:
            return jsonify({"message": "Password must be at least 8 characters"}), 400
        if Admin.query.filter_by(email=email).first():
            return jsonify({"message": "Email already registered"}), 409
        a = Admin(full_name=name, email=email)
        a.set_password(password)
        db.session.add(a)
        db.session.commit()
        return jsonify({"message": f"Admin '{name}' added successfully!"}), 201

    @app.route("/api/admin/users", methods=["GET"])
    @admin_required
    def admin_get_users():
        flag_inactive_users()
        users = User.query.order_by(User.user_id).all()
        return jsonify([{
            "user_id": u.user_id,
            "full_name": u.full_name,
            "email": u.email,
            "status": u.status or "active",
            "has_profile": u.age is not None,
            "last_login": u.last_login.strftime("%d %b %Y, %I:%M %p") if u.last_login else "Never",
            "body_goal": u.body_goal or "—",
            "target_calories": u.target_calories or 0,
            "food_type": u.food_type or "nonveg",
        } for u in users])

    @app.route("/api/admin/users/<int:uid>", methods=["GET"])
    @admin_required
    def admin_get_user(uid):
        u = db.session.get(User, uid)
        if not u:
            return jsonify({"message": "User not found"}), 404
        return jsonify({
            "user_id": u.user_id,
            "full_name": u.full_name,
            "email": u.email,
            "status": u.status or "active",
            "food_type": u.food_type or "nonveg",
            "last_login": u.last_login.strftime("%d %b %Y, %I:%M %p") if u.last_login else "Never",
            "age": u.age, "gender": u.gender,
            "height": u.height, "weight": u.weight,
            "activity_level": u.activity_level, "body_goal": u.body_goal,
            "maintenance_calories": u.maintenance_calories,
            "target_calories": u.target_calories,
        })

    @app.route("/api/admin/users/<int:uid>/restrict", methods=["POST"])
    @admin_required
    def admin_restrict(uid):
        u = db.session.get(User, uid)
        if not u:
            return jsonify({"message": "User not found"}), 404
        u.status = "restricted"
        db.session.commit()
        return jsonify({"message": f"{u.full_name} has been restricted."})

    @app.route("/api/admin/users/<int:uid>/reactivate", methods=["POST"])
    @admin_required
    def admin_reactivate(uid):
        u = db.session.get(User, uid)
        if not u:
            return jsonify({"message": "User not found"}), 404
        u.status = "active"
        db.session.commit()
        return jsonify({"message": f"{u.full_name} has been reactivated."})

    @app.route("/api/admin/users/<int:uid>/delete", methods=["POST"])
    @admin_required
    def admin_delete(uid):
        u = db.session.get(User, uid)
        if not u:
            return jsonify({"message": "User not found"}), 404
        MealPlan.query.filter_by(user_id=uid).delete()
        db.session.delete(u)
        db.session.commit()
        return jsonify({"message": "User deleted."})

    @app.route("/api/admin/users/<int:uid>/meal-plan", methods=["GET"])
    @admin_required
    def admin_user_mealplan(uid):
        plans = (MealPlan.query
                 .filter_by(user_id=uid)
                 .order_by(MealPlan.plan_id).all())
        if not plans:
            return jsonify([])
        return jsonify([{
            "day": p.day_of_week,
            "breakfast": {"name": p.breakfast_name, "calories": p.breakfast_calories},
            "lunch":     {"name": p.lunch_name,     "calories": p.lunch_calories},
            "dinner":    {"name": p.dinner_name,    "calories": p.dinner_calories},
            "total_calories": p.daily_total_calories,
        } for p in plans])

    @app.route("/api/admin/list", methods=["GET"])
    @admin_required
    def admin_list_admins():
        admins = Admin.query.order_by(Admin.admin_id).all()
        return jsonify([{
            "user_id": a.admin_id,
            "full_name": a.full_name,
            "email": a.email,
            "last_login": a.last_login.strftime("%d %b %Y, %I:%M %p") if a.last_login else "Never",
        } for a in admins])

    return app


if __name__ == "__main__":
    print("=" * 55)
    print("  NutriCalc Flask Server")
    print("=" * 55)
    print()
    print("  API running at:  http://0.0.0.0:5000")
    print()
    print("  Flutter app:  Set BASE_URL in main.dart to")
    print("  your PC's local IP, e.g.:")
    print("  const String BASE_URL = 'http://192.168.x.x:5000';")
    print()
    print("  To find your IP:")
    print("  Windows: ipconfig")
    print("  Mac/Linux: ifconfig or ip addr")
    print()
    print("  To create the first admin account, POST to:")
    print("  http://localhost:5000/api/admin/create")
    print("  Body: { full_name, email (@nutricalc.com), password }")
    print("=" * 55)
    create_app().run(host="0.0.0.0", port=5000, debug=True)