import React, { useState, useEffect, useRef } from "react";

/* ─── Google Fonts ─────────────────────────────────────────── */
const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href =
  "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap";
document.head.appendChild(fontLink);

/* ─── Design Tokens ─────────────────────────────────────────── */
const C = {
  navy:     "#0d1b2e",
  navyDark: "#081525",
  navyMid:  "#0f2035",
  orange:   "#f37321",
  orangeD:  "#dc5c00",
  orangeL:  "#ff9944",
  cream:    "#f5f0e8",
  white:    "#ffffff",
  gray50:   "#f9fafb",
  gray100:  "#f3f4f6",
  gray200:  "#e5e7eb",
  gray400:  "#9ca3af",
  gray600:  "#4b5563",
  gray800:  "#1f2937",
  green:    "#16a34a",
  greenL:   "#dcfce7",
  red:      "#dc2626",
  redL:     "#fef2f2",
  blue:     "#3b82f6",
  blueL:    "#eff6ff",
};

const NAV_H = 60;

const T = {
  nav: {
    position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
    height: NAV_H, background: C.navyDark,
    borderBottom: `3px solid transparent`, backgroundClip: "padding-box",
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0 28px", boxShadow: "0 2px 20px rgba(0,0,0,0.4)",
  },
  navBar: {
    position: "fixed", top: 0, left: 0, right: 0, height: 3, zIndex: 101,
    background: `linear-gradient(90deg, ${C.orange}, ${C.orangeL}, ${C.green})`,
  },
  logo: { display: "flex", alignItems: "center", gap: 10, fontFamily: "'DM Sans', sans-serif", textDecoration: "none" },
  logoBox: {
    width: 34, height: 34, borderRadius: 8,
    background: `linear-gradient(135deg, ${C.orange}, ${C.orangeD})`,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 18, fontWeight: 800, color: C.white, fontFamily: "'DM Sans', sans-serif",
  },
  logoText: { fontSize: 18, fontWeight: 700, color: C.white, fontFamily: "'DM Sans', sans-serif" },
  navRight: { display: "flex", alignItems: "center", gap: 12 },
  userPill: {
    display: "flex", alignItems: "center", gap: 8,
    background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 24, padding: "5px 14px 5px 6px",
    color: C.white, fontSize: 14, fontWeight: 500, fontFamily: "'DM Sans', sans-serif",
  },
  userAvatar: {
    width: 28, height: 28, borderRadius: "50%",
    background: `linear-gradient(135deg, ${C.orange}, ${C.orangeD})`,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 13, fontWeight: 700, color: C.white,
  },
  logoutBtn: {
    padding: "7px 16px", borderRadius: 8, background: "transparent",
    border: "1px solid rgba(255,255,255,0.2)", color: "rgba(255,255,255,0.7)",
    fontSize: 13, cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
    fontWeight: 500, transition: "all .2s",
  },
  wrap: { minHeight: "100vh", background: C.cream, fontFamily: "'DM Sans', sans-serif", paddingTop: NAV_H },
  hero: {
    background: `linear-gradient(135deg, ${C.navyDark} 0%, ${C.navyMid} 60%, #122040 100%)`,
    padding: "48px 32px 40px", color: C.white,
  },
  heroInner: { maxWidth: 960, margin: "0 auto" },
  heroSub: { fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: C.orange, fontWeight: 700, marginBottom: 8 },
  heroTitle: { fontFamily: "'Playfair Display', serif", fontSize: 36, fontWeight: 700, color: C.white, marginBottom: 4, lineHeight: 1.2 },
  heroTitle2: { color: C.orange },
  heroSub2: { fontSize: 14, color: "rgba(255,255,255,0.55)", marginTop: 6 },
  stepBar: { background: C.navyDark, padding: "0 32px", borderBottom: "1px solid rgba(255,255,255,0.06)" },
  stepInner: { maxWidth: 960, margin: "0 auto", display: "flex", alignItems: "center", gap: 0 },
  stepItem: (active, done) => ({
    display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
    padding: "14px 20px", borderBottom: active ? `2px solid ${C.orange}` : "2px solid transparent",
    cursor: "default", flex: 1,
  }),
  stepCircle: (active, done) => ({
    width: 28, height: 28, borderRadius: "50%",
    background: done ? C.orange : active ? C.orange : "rgba(255,255,255,0.1)",
    border: done || active ? "none" : "1px solid rgba(255,255,255,0.2)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 12, fontWeight: 700,
    color: done || active ? C.white : "rgba(255,255,255,0.4)",
  }),
  stepLabel: (active, done) => ({
    fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase",
    color: done || active ? C.orange : "rgba(255,255,255,0.35)",
    fontWeight: done || active ? 700 : 400,
  }),
  stepLine: { flex: 0, width: 1, height: 30, background: "rgba(255,255,255,0.08)" },
  content: { maxWidth: 960, margin: "0 auto", padding: "32px 16px" },
  card: {
    background: C.white, borderRadius: 16, border: `1px solid ${C.gray200}`,
    boxShadow: "0 1px 3px rgba(0,0,0,0.06), 0 4px 20px rgba(0,0,0,0.04)", padding: "28px 32px",
  },
  sideCard: {
    background: C.white, borderRadius: 14, border: `1px solid ${C.gray200}`,
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)", padding: "20px 24px",
  },
  primaryBtn: {
    width: "100%", padding: "14px 24px",
    background: `linear-gradient(135deg, ${C.orange}, ${C.orangeD})`,
    color: C.white, border: "none", borderRadius: 10,
    fontSize: 15, fontWeight: 700, cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
    boxShadow: `0 4px 20px rgba(243,115,33,0.3)`,
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8, transition: "all .2s",
  },
  secondaryBtn: {
    padding: "11px 20px", background: C.gray50, color: C.gray800,
    border: `1px solid ${C.gray200}`, borderRadius: 8, fontSize: 14,
    fontWeight: 600, cursor: "pointer", fontFamily: "'DM Sans', sans-serif", transition: "all .2s",
  },
  uploadItem: {
    background: C.gray50, borderRadius: 12, border: `1px solid ${C.gray200}`,
    padding: "18px 20px", marginBottom: 12, display: "flex", alignItems: "flex-start", gap: 16,
  },
  uploadIcon: {
    width: 44, height: 44, borderRadius: 10,
    background: `linear-gradient(135deg, #fff7ed, #ffe8cc)`, border: `1px solid #fed7aa`,
    display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, flexShrink: 0,
  },
  uploadBtn: {
    display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 8,
    background: C.navyDark, color: C.white, fontSize: 12, fontWeight: 600,
    cursor: "pointer", fontFamily: "'DM Sans', sans-serif", border: "none", marginTop: 8, transition: "all .2s",
  },
  section: {
    background: C.gray50, borderRadius: 12, border: `1px solid ${C.gray100}`,
    padding: "18px 22px", marginBottom: 16,
  },
  secHead: {
    display: "flex", alignItems: "center", gap: 8, marginBottom: 16,
    paddingBottom: 12, borderBottom: `1px solid ${C.gray200}`,
  },
  secTitle: { fontSize: 11, fontWeight: 700, color: C.gray600, letterSpacing: 1.2, textTransform: "uppercase" },
  fGrid: { display: "grid", gridTemplateColumns: "180px 1fr", gap: "8px 16px" },
  fLabel: { fontSize: 12, color: C.gray400, paddingTop: 9, fontWeight: 500 },
  fInput: {
    padding: "8px 10px", border: `1px solid ${C.gray200}`, borderRadius: 6,
    fontSize: 13, width: "100%", background: C.white, color: C.gray800,
    fontFamily: "'DM Sans', sans-serif", boxSizing: "border-box", outline: "none",
  },
  fDisabled: { background: C.gray50, color: C.gray600, border: `1px solid ${C.gray100}` },
  errBox: {
    background: C.redL, border: `1px solid #fecaca`, borderRadius: 8,
    padding: "10px 14px", fontSize: 13, color: C.red, marginBottom: 12,
  },
  okBox: {
    background: C.greenL, border: `1px solid #bbf7d0`, borderRadius: 8,
    padding: "12px 16px", fontSize: 13, color: C.green, marginBottom: 16, fontWeight: 600,
  },
  infoBox: {
    background: C.blueL, border: `1px solid #bfdbfe`, borderRadius: 8,
    padding: "10px 14px", fontSize: 12, color: C.blue, marginBottom: 12,
  },
  badge: (color, bg, border) => ({
    display: "inline-flex", alignItems: "center", padding: "3px 10px", borderRadius: 20,
    fontSize: 11, fontWeight: 600, color, background: bg, border: `1px solid ${border}`,
  }),
};

/* ─── Navbar ─────────────────────────────────────────────────── */
function Navbar({ userName, onLogout }) {
  return (
    <>
      <div style={T.navBar} />
      <nav style={T.nav}>
        <div style={T.logo}>
          <div style={T.logoBox}>D</div>
          <span style={T.logoText}>DocAssist</span>
        </div>
        <div style={T.navRight}>
          {userName && (
            <div style={T.userPill}>
              <div style={T.userAvatar}>{userName[0].toUpperCase()}</div>
              {userName}
            </div>
          )}
          {onLogout && (
            <button style={T.logoutBtn} onClick={onLogout}
              onMouseEnter={e => { e.target.style.background = "rgba(255,255,255,0.1)"; e.target.style.color = C.white; }}
              onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.color = "rgba(255,255,255,0.7)"; }}>
              Logout
            </button>
          )}
        </div>
      </nav>
    </>
  );
}

/* ─── Stepper ────────────────────────────────────────────────── */
const STEPS = ["Login", "Upload", "Verify", "Autofill", "Payment"];
function Stepper({ active }) {
  return (
    <div style={T.stepBar}>
      <div style={T.stepInner}>
        {STEPS.map((s, i) => (
          <React.Fragment key={s}>
            <div style={T.stepItem(i + 1 === active, i + 1 < active)}>
              <div style={T.stepCircle(i + 1 === active, i + 1 < active)}>
                {i + 1 < active ? "✓" : i + 1}
              </div>
              <span style={T.stepLabel(i + 1 === active, i + 1 < active)}>{s}</span>
            </div>
            {i < STEPS.length - 1 && <div style={T.stepLine} />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/* ─── Login Field ────────────────────────────────────────────── */
function LField({ label, placeholder, type = "text", value, onChange, icon, onEnter }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: "block", fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 6, letterSpacing: 0.5, fontWeight: 500 }}>
        {label}
      </label>
      <div style={{ position: "relative" }}>
        <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", fontSize: 15 }}>{icon}</span>
        <input
          type={type} placeholder={placeholder} value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={e => e.key === "Enter" && onEnter && onEnter()}
          style={{
            width: "100%", padding: "11px 12px 11px 38px",
            background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.14)",
            borderRadius: 9, color: C.white, fontSize: 14,
            fontFamily: "'DM Sans', sans-serif", outline: "none", boxSizing: "border-box",
          }}
          onFocus={e => e.target.style.borderColor = `${C.orange}99`}
          onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.14)"}
        />
      </div>
    </div>
  );
}

/* ─── Review Field / Select ──────────────────────────────────── */
function RF({ label, v, ch, editable, type = "text", err }) {
  return (
    <>
      <div style={T.fLabel}>{label}</div>
      <div>
        <input type={type} disabled={!editable} value={v || ""} onChange={e => ch && ch(e.target.value)}
          style={{ ...T.fInput, ...(err ? { borderColor: "#ef4444", background: C.redL } : {}), ...(!editable ? T.fDisabled : {}) }} />
        {err && <div style={{ fontSize: 11, color: C.red, marginTop: 2 }}>{err}</div>}
      </div>
    </>
  );
}

/* ─── FIX: RS (Select) — properly handles pre-filled values ──── */
function RS({ label, v, opts, ch, editable, err }) {
  return (
    <>
      <div style={T.fLabel}>{label}</div>
      <div>
        <select
          disabled={!editable}
          value={v || ""}
          onChange={e => ch(e.target.value)}
          style={{
            ...T.fInput,
            ...(err ? { borderColor: "#ef4444", background: C.redL } : {}),
            ...(!editable ? T.fDisabled : {}),
            // Show orange highlight when a value is auto-filled and not editable
            ...(v && !editable ? { color: C.gray800, fontWeight: 500 } : {}),
          }}
        >
          <option value="">Select</option>
          {opts.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        {err && <div style={{ fontSize: 11, color: C.red, marginTop: 2 }}>{err}</div>}
      </div>
    </>
  );
}

function RSection({ title, icon, children }) {
  return (
    <div style={T.section}>
      <div style={T.secHead}>
        <span style={{ fontSize: 16 }}>{icon}</span>
        <span style={T.secTitle}>{title}</span>
      </div>
      <div style={T.fGrid}>{children}</div>
    </div>
  );
}

function UCard({ icon, title, required, hint, subhint, accept, onChange, status, preview, previewStyle }) {
  return (
    <div style={T.uploadItem}>
      <div style={T.uploadIcon}>{icon}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: C.gray800, marginBottom: 2 }}>
          {title}{required && <span style={{ color: C.red, marginLeft: 3 }}>*</span>}
        </div>
        <div style={{ fontSize: 12, color: C.gray400 }}>{hint}</div>
        {subhint && <div style={{ fontSize: 12, color: C.orange, marginTop: 1 }}>{subhint}</div>}
        <label style={T.uploadBtn}
          onMouseEnter={e => e.currentTarget.style.background = "#1a2e45"}
          onMouseLeave={e => e.currentTarget.style.background = C.navyDark}>
          <input type="file" accept={accept} onChange={onChange} style={{ display: "none" }} />
          ↑ Upload
        </label>
        {preview && <img src={preview} alt="Preview" style={previewStyle} />}
        {status && (
          <div style={{ fontSize: 12, marginTop: 6, fontWeight: 500, color: status.ok === true ? C.green : status.ok === false ? C.red : C.orange }}>
            {status.msg}
          </div>
        )}
      </div>
    </div>
  );
}

function ConfChip({ v }) {
  const pct = Math.round(v * 100);
  const color = v >= .8 ? C.green : v >= .5 ? C.orange : C.red;
  const bg = v >= .8 ? C.greenL : v >= .5 ? "#fff7ed" : C.redL;
  const border = v >= .8 ? "#86efac" : v >= .5 ? "#fed7aa" : "#fecaca";
  return <div style={T.badge(color, bg, border)}>OCR {pct}%</div>;
}

/* ═══════════════════════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════════════════════ */
export default function App() {
  const [page, setPage] = useState("login");
  const [authMode, setAuthMode] = useState("login");
  const [editable, setEditable] = useState(false);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [showRedirectMsg, setShowRedirectMsg] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [loadingPct, setLoadingPct] = useState(0);
  const [paymentRef, setPaymentRef] = useState("");
  const [applicantName, setApplicantName] = useState("");

  const [authName, setAuthName] = useState("");
  const [authIdentifier, setAuthIdentifier] = useState(""); // ← single email-or-phone field
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  // Register-specific fields (still need both for account creation)
  const [regEmail, setRegEmail] = useState("");
  const [regPhone, setRegPhone] = useState("");

  const [welcomeMsg, setWelcomeMsg] = useState("");
  const [userId, setUserId] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [userPhone, setUserPhone] = useState("");
  const [userName, setUserName] = useState("");
  const [isReturning, setIsReturning] = useState(false);

  const [aadhaar, setAadhaar] = useState(null);
  const [photo, setPhoto] = useState(null);
  const [signature, setSignature] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [sigPreview, setSigPreview] = useState(null);
  const [photoStatus, setPhotoStatus] = useState(null);
  const [sigStatus, setSigStatus] = useState(null);

  const canvasRef = useRef(null);
  const cardRef = useRef(null);
  const bgRef = useRef(null);

  const [form, setForm] = useState({
    income_salary: false, income_capital_gains: false,
    income_house_property: false, income_other_sources: false,
    income_business: false, income_no_income: false,
  });

  const requiredFields = [
    { key: "first_name", label: "First Name" },
    { key: "last_name", label: "Last Name" },
    { key: "dob", label: "Date of Birth" },
    { key: "gender", label: "Gender" },
    { key: "aadhaar_number", label: "Aadhaar Number" },
    { key: "mobile", label: "Mobile Number" },
    { key: "email", label: "Email" },
    { key: "locality", label: "Area / Locality" },
    { key: "city", label: "Town / City" },
    { key: "residentState", label: "Resident State" },
    { key: "pincode", label: "Pincode" },
  ];

  /* ── Particles ── */
  useEffect(() => {
    if (page !== "login") return;
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize(); window.addEventListener("resize", resize);
    const pts = Array.from({ length: 60 }, () => ({
      x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight,
      r: Math.random() * 2 + .5, dx: (Math.random() - .5) * .35, dy: (Math.random() - .5) * .35,
      o: Math.random() * .3 + .1,
    }));
    let id;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pts.forEach(p => {
        p.x += p.dx; p.y += p.dy;
        if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(243,115,33,${p.o})`; ctx.fill();
      });
      id = requestAnimationFrame(draw);
    };
    draw();
    const move = e => {
      const x = (window.innerWidth / 2 - e.pageX) / 40, y = (window.innerHeight / 2 - e.pageY) / 40;
      if (cardRef.current) cardRef.current.style.transform = `rotateY(${x}deg) rotateX(${y}deg)`;
      if (bgRef.current) bgRef.current.style.transform = `translate(${x * 2}px,${y * 2}px) scale(1.05)`;
    };
    document.addEventListener("mousemove", move);
    return () => { cancelAnimationFrame(id); document.removeEventListener("mousemove", move); window.removeEventListener("resize", resize); };
  }, [page]);

  /* ════════════════════════════════════════════════════════════
     AUTH — LOGIN with single identifier field (email OR phone)
  ════════════════════════════════════════════════════════════ */
  const handleLogin = async () => {
    if (!authIdentifier) { setAuthError("Enter your email or phone number."); return; }
    if (!authPassword) { setAuthError("Password is required."); return; }
    setAuthLoading(true); setAuthError("");

    // Detect whether identifier is email or phone
    const isEmail = authIdentifier.includes("@");
    const payload = {
      password: authPassword,
      ...(isEmail ? { email: authIdentifier } : { phone: authIdentifier }),
    };

    try {
      const res = await fetch("http://127.0.0.1:5000/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const d = await res.json();
      if (!res.ok) { setAuthError(d.error); return; }
      setUserId(d.userId); setUserEmail(d.email); setUserPhone(d.phone);
      setUserName(d.name); setIsReturning(d.isReturning);
      setWelcomeMsg(d.message); setPage("dashboard");
    } catch { setAuthError("Server error. Is Flask running?"); }
    finally { setAuthLoading(false); }
  };

  const handleRegister = async () => {
    if (!authName || !regEmail || !regPhone || !authPassword) { setAuthError("All fields are required."); return; }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(regEmail)) { setAuthError("Enter a valid email."); return; }
    if (!/^\d{10}$/.test(regPhone)) { setAuthError("Phone must be 10 digits."); return; }
    if (authPassword.length < 6) { setAuthError("Password must be at least 6 characters."); return; }
    setAuthLoading(true); setAuthError("");
    try {
      const res = await fetch("http://127.0.0.1:5000/register", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: authName, email: regEmail, phone: regPhone, password: authPassword }),
      });
      const d = await res.json();
      if (!res.ok) { setAuthError(d.error); return; }
      setUserId(d.userId); setUserEmail(d.email); setUserPhone(d.phone);
      setUserName(d.name); setIsReturning(false);
      setWelcomeMsg(`Welcome, ${d.name}! Your account has been created.`); setPage("dashboard");
    } catch { setAuthError("Server error. Is Flask running?"); }
    finally { setAuthLoading(false); }
  };

  /* ── Image resize ── */
  const resizeImage = (file, maxW, maxH, maxKB) => new Promise(resolve => {
    const url = URL.createObjectURL(file); const img = new Image();
    img.onload = () => {
      let { naturalWidth: w, naturalHeight: h } = img; const r = Math.min(maxW / w, maxH / h, 1); w = Math.round(w * r); h = Math.round(h * r);
      const c = document.createElement("canvas"); c.width = w; c.height = h; c.getContext("2d").drawImage(img, 0, 0, w, h);
      let q = 0.92; const go = () => c.toBlob(b => {
        if (b.size <= maxKB * 1024 || q <= 0.1) { URL.revokeObjectURL(url); resolve({ file: new File([b], file.name, { type: "image/jpeg" }), width: w, height: h, sizeKB: (b.size / 1024).toFixed(1) }); }
        else { q = parseFloat((q - .08).toFixed(2)); go(); }
      }, "image/jpeg", q); go();
    }; img.src = url;
  });

  const handlePhotoUpload = async file => {
    if (!file) return;
    if (!["image/jpeg", "image/png"].includes(file.type)) { setPhotoStatus({ ok: false, msg: "❌ Use JPG or PNG." }); setPhoto(null); setPhotoPreview(null); return; }
    setPhotoStatus({ ok: null, msg: "⏳ Checking..." });
    const url = URL.createObjectURL(file); const img = new Image();
    img.onload = async () => {
      const w = img.naturalWidth, h = img.naturalHeight; URL.revokeObjectURL(url);
      if (w < 50 || h < 50) { setPhotoStatus({ ok: false, msg: `❌ Too small: ${w}×${h}px.` }); setPhoto(null); setPhotoPreview(null); return; }
      let f = file, note = `${w}×${h}px, ${(file.size / 1024).toFixed(1)}KB`;
      if (file.size > 50 * 1024) { setPhotoStatus({ ok: null, msg: "📐 Resizing..." }); const r = await resizeImage(file, 213, 213, 50); f = r.file; note = `resized to ${r.width}×${r.height}px, ${r.sizeKB}KB`; }
      setPhotoPreview(URL.createObjectURL(f)); setPhoto(f); setPhotoStatus({ ok: true, msg: `✅ ${note}` });
    }; img.src = url;
  };

  const handleSignatureUpload = async file => {
    if (!file) return;
    if (!["image/jpeg", "image/png"].includes(file.type)) { setSigStatus({ ok: false, msg: "❌ Use JPG or PNG." }); setSignature(null); setSigPreview(null); return; }
    setSigStatus({ ok: null, msg: "⏳ Checking..." });
    const url = URL.createObjectURL(file); const img = new Image();
    img.onload = async () => {
      const w = img.naturalWidth, h = img.naturalHeight; URL.revokeObjectURL(url);
      if (w < 50 || h < 10) { setSigStatus({ ok: false, msg: `❌ Too small: ${w}×${h}px.` }); setSignature(null); setSigPreview(null); return; }
      let f = file, note = `${w}×${h}px, ${(file.size / 1024).toFixed(1)}KB`;
      if (file.size > 60 * 1024) { setSigStatus({ ok: null, msg: "📐 Resizing..." }); const r = await resizeImage(file, 500, 200, 60); f = r.file; note = `resized to ${r.width}×${r.height}px, ${r.sizeKB}KB`; }
      setSigPreview(URL.createObjectURL(f)); setSignature(f); setSigStatus({ ok: true, msg: `✅ ${note}` });
    }; img.src = url;
  };

  const startPaymentPolling = () => {
    const iv = setInterval(async () => {
      try {
        const r = await fetch("http://127.0.0.1:4000/payment-status"); const d = await r.json();
        if (d.status === "paid") { clearInterval(iv); setPaymentRef(d.ref); setApplicantName(d.applicant); setPage("success"); }
      } catch { }
    }, 3000);
  };

  /* ── Submit ── */
  const handleSubmit = async () => {
    if (!aadhaar) { setError("Please upload your Aadhaar card."); return; }
    if (!photo) { setError("Please upload a photograph."); return; }
    if (!signature) { setError("Please upload your signature."); return; }
    if (photoStatus && !photoStatus.ok) { setError("Fix photo issue first."); return; }
    if (sigStatus && !sigStatus.ok) { setError("Fix signature issue first."); return; }
    setLoading(true); setError("");

    const fd = new FormData();
    fd.append("aadhaar", aadhaar); fd.append("photo", photo); fd.append("signature", signature);
    if (userName) fd.append("login_name", userName);

    const steps = [
      { label: "📤 Uploading documents...", pct: 10 },
      { label: "🔍 Running OCR on Aadhaar...", pct: 30 },
      { label: "🤖 AI extracting fields...", pct: 60 },
      { label: "✅ Finalising...", pct: 90 },
    ];
    let stepIdx = 0;
    setLoadingStep(steps[0].label); setLoadingPct(steps[0].pct);
    const animTimer = setInterval(() => {
      stepIdx = Math.min(stepIdx + 1, steps.length - 1);
      setLoadingStep(steps[stepIdx].label); setLoadingPct(steps[stepIdx].pct);
    }, 2200);

    try {
      const res = await fetch("http://127.0.0.1:5000/process", { method: "POST", body: fd });
      clearInterval(animTimer);
      setLoadingPct(100); setLoadingStep("✅ Done!");
      await new Promise(r => setTimeout(r, 300));
      const data = await res.json();

      /* ── FIX: Normalize gender to exactly "Male" / "Female" ── */
      const rawGender = (data.gender || "").trim();
      const normalizedGender =
        rawGender.toLowerCase() === "female" ? "Female" :
        rawGender.toLowerCase() === "male"   ? "Male"   : rawGender;

      /* ── FIX: Normalize DOB — backend returns YYYY-MM-DD which is
         correct for <input type="date">, but if it comes as DD/MM/YYYY
         or DD-MM-YYYY, convert it here ── */
      let normalizedDob = (data.dob || "").trim();
      const dobMatch = normalizedDob.match(/^(\d{2})[-/.](\d{2})[-/.](\d{4})$/);
      if (dobMatch) {
        normalizedDob = `${dobMatch[3]}-${dobMatch[2]}-${dobMatch[1]}`;
      }

      setForm(prev => ({
        ...prev,
        _id: data._id || "",
        first_name: data.first_name || "",
        middle_name: data.middle_name || "",
        last_name: data.last_name || "",
        dob: normalizedDob,
        gender: normalizedGender,           // ← normalized value
        aadhaar_number: data.aadhaar_number || "",
        confidence: data.confidence,
        email: data.email || userEmail || "",
        mobile: data.mobile || userPhone || "",
        fatherFirstName: data.father_first_name || "",
        fatherLastName: data.father_last_name || "",
        flat_no: data.flat_no || "",
        building: data.building || "",
        road: data.road || "",
        locality: data.locality || "",
        city: data.city || "",
        district: data.district || "",
        residentState: data.residentState || "",
        pincode: data.pincode || "",
        address: data.address || "",
        aadhaar_path: data.aadhaar_path || "",
        photo_path: data.photo_path || "",
        signature_path: data.signature_path || "",
        areaCode: data.areaCode || "",
        aoType: data.aoType || "",
        rangeCode: data.rangeCode || "",
        aoNumber: data.aoNumber || "",
        rep_first: data.father_first_name || "",
        rep_last: data.father_last_name || "",
        income_salary: false, income_capital_gains: false,
        income_house_property: false, income_other_sources: false,
        income_business: false, income_no_income: false,
      }));
      setPage("review");
    } catch {
      clearInterval(animTimer);
      setError("Processing failed. Is Flask running?");
    } finally {
      setLoading(false); setLoadingStep(""); setLoadingPct(0);
    }
  };

  const validateReviewForm = () => {
    const errs = {};
    requiredFields.forEach(({ key, label }) => { if (!form[key] || !String(form[key]).trim()) errs[key] = `${label} is required`; });
    if (form.aadhaar_number && !/^\d{4}\s\d{4}\s\d{4}$/.test(form.aadhaar_number)) errs.aadhaar_number = "Format: XXXX XXXX XXXX";
    if (form.mobile && !/^\d{10}$/.test(form.mobile.replace(/\s/g, ""))) errs.mobile = "Must be 10 digits";
    if (form.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) errs.email = "Enter a valid email";
    if (form.pincode && !/^\d{6}$/.test(form.pincode)) errs.pincode = "Must be 6 digits";
    if (form.dob && new Date(form.dob) >= new Date()) errs.dob = "Must be in the past";
    if (!["income_salary", "income_capital_gains", "income_house_property", "income_other_sources", "income_business", "income_no_income"].some(f => form[f])) errs.income = "Select at least one";
    setFieldErrors(errs); return Object.keys(errs).length === 0;
  };

  const handleConfirm = async () => {
    if (!validateReviewForm()) { setError("Fix the highlighted errors before proceeding."); return; }
    setLoading(true); setError("");
    try {
      await fetch(`http://127.0.0.1:5000/confirm/${form._id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...form, userId }) });
      const res = await fetch("http://127.0.0.1:4000/autofill", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(form) });
      const r = await res.json();
      if (r.status === "success") { setShowRedirectMsg(true); startPaymentPolling(); }
    } catch { setError("Autofill failed. Is Puppeteer running on port 4000?"); }
    finally { setLoading(false); }
  };

  const update = (k, v) => setForm(p => ({ ...p, [k]: v }));
  const updateCheck = k => setForm(p => ({ ...p, [k]: !p[k] }));

  const resetAll = () => {
    setPage("login"); setAuthMode("login"); setShowRedirectMsg(false); setFieldErrors({});
    setPaymentRef(""); setApplicantName(""); setError(""); setWelcomeMsg("");
    setAadhaar(null); setPhoto(null); setSignature(null);
    setPhotoPreview(null); setSigPreview(null); setPhotoStatus(null); setSigStatus(null);
    setAuthName(""); setAuthIdentifier(""); setAuthPassword(""); setAuthError("");
    setRegEmail(""); setRegPhone("");
    setUserEmail(""); setUserPhone(""); setUserName(""); setUserId("");
    setForm({ income_salary: false, income_capital_gains: false, income_house_property: false, income_other_sources: false, income_business: false, income_no_income: false });
    fetch("http://127.0.0.1:4000/payment-reset", { method: "POST" }).catch(() => { });
  };

  /* ════════════════════════════════════════════════════════════
     LOGIN PAGE
  ════════════════════════════════════════════════════════════ */
  if (page === "login") return (
    <div style={{
      position: "relative", width: "100vw", height: "100vh",
      display: "flex", justifyContent: "center", alignItems: "center",
      overflow: "hidden", perspective: "1200px", fontFamily: "'DM Sans', sans-serif",
    }}>
      <div ref={bgRef} style={{
        position: "absolute", width: "115%", height: "115%",
        backgroundImage: `url("https://t4.ftcdn.net/jpg/15/03/46/79/360_F_1503467946_oLfcObbuJaCLxQZoO032Q0QmyeK5qe5o.jpg")`,
        backgroundSize: "cover", backgroundPosition: "center",
        filter: "brightness(0.3) saturate(0.6)", zIndex: -1, transition: "transform 0.15s ease",
      }} />
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(160deg,rgba(8,21,37,0.7),rgba(20,6,0,0.6))", zIndex: 0 }} />
      <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, zIndex: 0, pointerEvents: "none" }} />
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg,${C.orange},${C.orangeL},${C.green})`, zIndex: 10 }} />
      <div style={{ position: "absolute", top: 20, left: 28, zIndex: 10, display: "flex", alignItems: "center", gap: 10 }}>
        <div style={T.logoBox}>D</div>
        <span style={{ ...T.logoText, fontSize: 20 }}>DocAssist</span>
      </div>

      <div style={{ position: "relative", zIndex: 2, display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
        <div ref={cardRef} style={{
          width: 400, borderRadius: 18, overflow: "hidden",
          background: "rgba(8,18,40,0.9)", backdropFilter: "blur(28px)",
          border: "1px solid rgba(255,255,255,0.1)", boxShadow: "0 30px 80px rgba(0,0,0,0.6)",
          transition: "transform 0.12s ease",
        }}>
          <div style={{ height: 4, background: `linear-gradient(90deg,${C.orange},${C.orangeL})` }} />

          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
            {["login", "register"].map(m => (
              <button key={m} style={{
                flex: 1, padding: "15px", border: "none",
                background: authMode === m ? "rgba(243,115,33,0.1)" : "transparent",
                color: authMode === m ? C.orange : "rgba(255,255,255,0.35)",
                fontSize: 14, fontWeight: 700, cursor: "pointer",
                borderBottom: authMode === m ? `2px solid ${C.orange}` : "2px solid transparent",
                fontFamily: "'DM Sans', sans-serif", textTransform: "capitalize",
              }} onClick={() => { setAuthMode(m); setAuthError(""); }}>
                {m === "login" ? "Login" : "Register"}
              </button>
            ))}
          </div>

          <div style={{ padding: "24px 28px 28px" }}>
            {/* ── REGISTER form ── */}
            {authMode === "register" && (
              <>
                <LField label="Full Name" placeholder="Prakash Ranjan" value={authName} onChange={setAuthName} icon="👤" />
                <LField label="Email Address" placeholder="you@email.com" type="email" value={regEmail} onChange={setRegEmail} icon="✉️" />
                <LField label="Phone Number" placeholder="10-digit mobile" type="tel" value={regPhone} onChange={setRegPhone} icon="📱" />
              </>
            )}

            {/* ── LOGIN form — single identifier field ── */}
            {authMode === "login" && (
              <LField
                label="Email or Phone Number"
                placeholder="you@email.com or 9876543210"
                value={authIdentifier}
                onChange={setAuthIdentifier}
                icon="👤"
                onEnter={handleLogin}
              />
            )}

            <LField
              label="Password"
              placeholder={authMode === "register" ? "Min 6 characters" : "Your password"}
              type="password"
              value={authPassword}
              onChange={setAuthPassword}
              icon="🔒"
              onEnter={authMode === "login" ? handleLogin : handleRegister}
            />

            {authError && (
              <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "9px 13px", fontSize: 12, color: "#fca5a5", marginBottom: 14 }}>
                {authError}
              </div>
            )}

            <button style={{ ...T.primaryBtn, opacity: authLoading ? .7 : 1 }}
              onClick={authMode === "login" ? handleLogin : handleRegister}
              disabled={authLoading}>
              {authLoading ? "Please wait..." : authMode === "login" ? "Login →" : "Create Account →"}
            </button>

            <div style={{ textAlign: "center", marginTop: 16, fontSize: 12, color: "rgba(255,255,255,0.35)" }}>
              {authMode === "login"
                ? <>New here? <span style={{ color: C.orangeL, cursor: "pointer" }} onClick={() => { setAuthMode("register"); setAuthError(""); }}>Register</span></>
                : <>Have an account? <span style={{ color: C.orangeL, cursor: "pointer" }} onClick={() => { setAuthMode("login"); setAuthError(""); }}>Login</span></>
              }
            </div>
          </div>
        </div>

        <div style={{ marginTop: 24, fontSize: 11, color: "rgba(255,255,255,0.2)", letterSpacing: 2, textTransform: "uppercase" }}>
          Government of India · Income Tax Department
        </div>
      </div>
    </div>
  );

  /* ════════════════════════════════════════════════════════════
     DASHBOARD
  ════════════════════════════════════════════════════════════ */
  if (page === "dashboard") return (
    <div style={T.wrap}>
      <Navbar userName={userName} onLogout={resetAll} />
      <div style={T.hero}>
        <div style={T.heroInner}>
          <div style={T.heroSub}>WELCOME BACK</div>
          <h1 style={T.heroTitle}>Hello, <span style={T.heroTitle2}>{userName}</span> 👋</h1>
          <div style={T.heroSub2}>Choose a government service to get started</div>
        </div>
      </div>
      <div style={T.content}>
        {welcomeMsg && <div style={{ ...T.okBox, marginBottom: 24 }}>{isReturning ? "👋" : "🎉"} {welcomeMsg}</div>}
        <div style={{ fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: C.gray400, fontWeight: 700, marginBottom: 16 }}>AVAILABLE SERVICES</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 36 }}>
          <div style={{ background: C.white, borderRadius: 14, border: `2px solid ${C.orange}`, boxShadow: `0 0 0 4px rgba(243,115,33,0.08), 0 4px 20px rgba(0,0,0,0.06)`, padding: "24px 22px", cursor: "pointer", transition: "all .2s" }}
            onClick={() => setPage("upload")}
            onMouseEnter={e => e.currentTarget.style.transform = "translateY(-2px)"}
            onMouseLeave={e => e.currentTarget.style.transform = "translateY(0)"}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>🪪</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: C.gray800, marginBottom: 6 }}>PAN Card</div>
            <div style={{ fontSize: 12, color: C.gray400, marginBottom: 14, lineHeight: 1.5 }}>Apply for a new PAN card using your Aadhaar. AI‑assisted autofill up to payment stage.</div>
            <div style={T.badge(C.green, C.greenL, "#86efac")}>✓ Available Now</div>
          </div>
          {[{ icon: "🗳️", title: "Voter ID", desc: "Register for a Voter ID card via the NVSP portal." }, { icon: "📘", title: "Passport", desc: "Apply for a fresh Indian passport through the Passport Seva portal." }, { icon: "🏥", title: "Ayushman Bharat", desc: "Enrol for PM‑JAY health insurance scheme." }].map(s => (
            <div key={s.title} style={{ background: C.gray50, borderRadius: 14, border: `1px solid ${C.gray200}`, padding: "24px 22px", opacity: 0.7 }}>
              <div style={{ fontSize: 32, marginBottom: 12, filter: "grayscale(1)" }}>{s.icon}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: C.gray600, marginBottom: 6 }}>{s.title}</div>
              <div style={{ fontSize: 12, color: C.gray400, marginBottom: 14 }}>{s.desc}</div>
              <div style={T.badge(C.gray400, C.gray100, C.gray200)}>Coming Soon</div>
            </div>
          ))}
        </div>
        <div style={{ background: `linear-gradient(135deg, ${C.orange}, #d4600a)`, borderRadius: 16, padding: "28px 36px", display: "flex", alignItems: "center", justifyContent: "space-between", boxShadow: `0 8px 32px rgba(243,115,33,0.3)` }}>
          <div>
            <div style={{ fontSize: 20, fontWeight: 700, color: C.white, marginBottom: 4, fontFamily: "'Playfair Display', serif" }}>Ready to apply for your PAN Card?</div>
            <div style={{ fontSize: 13, color: "rgba(255,255,255,0.8)" }}>It takes less than 5 minutes. Upload your Aadhaar and let our AI do the rest.</div>
          </div>
          <button style={{ padding: "13px 28px", borderRadius: 10, background: C.white, color: C.orange, border: "none", fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: "'DM Sans', sans-serif", whiteSpace: "nowrap", flexShrink: 0, marginLeft: 24, boxShadow: "0 4px 16px rgba(0,0,0,0.15)" }} onClick={() => setPage("upload")}>
            Start Application →
          </button>
        </div>
      </div>
    </div>
  );

  /* ════════════════════════════════════════════════════════════
     UPLOAD PAGE
  ════════════════════════════════════════════════════════════ */
  if (page === "upload") return (
    <div style={T.wrap}>
      <Navbar userName={userName} onLogout={resetAll} />
      <Stepper active={2} />
      <div style={{ ...T.hero, padding: "36px 32px 30px" }}>
        <div style={T.heroInner}>
          <h2 style={{ ...T.heroTitle, fontSize: 28 }}>Upload Your Documents</h2>
          <div style={T.heroSub2}>We'll extract your details automatically using OCR technology</div>
        </div>
      </div>
      <div style={T.content}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 24, alignItems: "start" }}>
          <div style={T.card}>
            <UCard icon="🪪" title="Aadhaar Card" required hint="JPG, PNG or PDF · Max 2MB · Front side" accept=".pdf,.jpg,.jpeg,.png"
              onChange={e => { const f = e.target.files[0]; if (!f) return; if (!["application/pdf", "image/jpeg", "image/png"].includes(f.type)) { setError("Use PDF, JPG or PNG"); return; } if (f.size > 2 * 1024 * 1024) { setError("Max 2MB"); return; } setError(""); setAadhaar(f); }}
              status={aadhaar ? { ok: true, msg: `✅ ${aadhaar.name} — ${(aadhaar.size / 1024).toFixed(1)}KB` } : null} />
            <UCard icon="📸" title="Passport Size Photo" required hint="JPG / PNG · Max 50KB · 213×213px · Colour" subhint="Over 50KB will be automatically resized" accept=".jpg,.jpeg,.png"
              onChange={e => { const f = e.target.files[0]; if (f) handlePhotoUpload(f); }}
              status={photoStatus} preview={photoPreview} previewStyle={{ width: 72, height: 72, objectFit: "cover", borderRadius: 8, border: `2px solid #fed7aa`, marginTop: 10 }} />
            <UCard icon="✍️" title="Signature" required hint="JPG / PNG · Max 50KB · Black & white" subhint="Over 60KB will be automatically resized" accept=".jpg,.jpeg,.png"
              onChange={e => { const f = e.target.files[0]; if (f) handleSignatureUpload(f); }}
              status={sigStatus} preview={sigPreview} previewStyle={{ maxWidth: 210, maxHeight: 65, borderRadius: 4, border: `2px solid #fed7aa`, marginTop: 10 }} />
            {error && <div style={T.errBox}>{error}</div>}
            {loading && (
              <div style={{ background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 10, padding: "14px 16px", marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: C.orangeD }}>{loadingStep}</span>
                  <span style={{ fontSize: 12, color: C.gray400 }}>{loadingPct}%</span>
                </div>
                <div style={{ background: C.gray100, borderRadius: 8, height: 8, overflow: "hidden" }}>
                  <div style={{ height: "100%", borderRadius: 8, background: `linear-gradient(90deg,${C.orange},${C.orangeL})`, width: `${loadingPct}%`, transition: "width 0.5s ease" }} />
                </div>
                <div style={{ fontSize: 11, color: C.gray400, marginTop: 6 }}>AI extraction in progress…</div>
              </div>
            )}
            <button style={{ ...T.primaryBtn, opacity: loading ? .6 : 1, marginTop: 4 }} onClick={handleSubmit} disabled={loading}>
              {loading ? loadingStep || "Processing..." : "⚡ Extract Details with OCR"}
            </button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={T.sideCard}>
              <div style={{ fontSize: 14, fontWeight: 700, color: C.gray800, marginBottom: 14 }}>Upload Progress</div>
              {[{ label: "Aadhaar Card", done: !!aadhaar }, { label: "Passport Photo", done: !!(photo && photoStatus?.ok) }, { label: "Signature", done: !!(signature && sigStatus?.ok) }].map((item, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 0", borderBottom: i < 2 ? `1px solid ${C.gray100}` : "none" }}>
                  <div style={{ width: 22, height: 22, borderRadius: "50%", background: item.done ? C.orange : C.gray100, border: `1px solid ${item.done ? C.orange : C.gray200}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: item.done ? C.white : C.gray400, fontWeight: 700, flexShrink: 0 }}>
                    {item.done ? "✓" : i + 1}
                  </div>
                  <span style={{ fontSize: 13, color: item.done ? C.gray800 : C.gray400, fontWeight: item.done ? 600 : 400 }}>{item.label}</span>
                </div>
              ))}
            </div>
            <div style={T.sideCard}>
              <div style={{ fontSize: 14, fontWeight: 700, color: C.gray800, marginBottom: 12 }}>⭐ Tips for best results</div>
              {["Ensure Aadhaar text is clearly visible — no blur or glare", "Photo must be recent, white background, 213×213 px", "Signature on white paper, scanned at 600 dpi", "Files are processed only for this session and never stored permanently"].map((tip, i) => (
                <div key={i} style={{ display: "flex", gap: 8, marginBottom: 10, fontSize: 12, color: C.gray600, lineHeight: 1.5 }}>
                  <span style={{ color: C.orange, flexShrink: 0 }}>›</span>{tip}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  /* ════════════════════════════════════════════════════════════
     REVIEW PAGE
  ════════════════════════════════════════════════════════════ */
  if (page === "review") return (
    <div style={T.wrap}>
      <Navbar userName={userName} onLogout={resetAll} />
      <Stepper active={3} />
      <div style={{ ...T.hero, padding: "30px 32px 26px" }}>
        <div style={{ ...T.heroInner, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ ...T.heroTitle, fontSize: 26, marginBottom: 2 }}>Review & Confirm Details</h2>
            <div style={T.heroSub2}>Verify all fields. Click Edit to correct any OCR mistakes.</div>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {form.confidence !== undefined && <ConfChip v={form.confidence} />}
            <button style={{ padding: "9px 18px", borderRadius: 8, background: editable ? `rgba(243,115,33,0.15)` : "rgba(255,255,255,0.1)", color: editable ? C.orange : "rgba(255,255,255,0.7)", border: editable ? `1px solid ${C.orange}` : "1px solid rgba(255,255,255,0.2)", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}
              onClick={() => { setEditable(!editable); setFieldErrors({}); }}>
              {editable ? "🔒 Lock" : "✏️ Edit"}
            </button>
          </div>
        </div>
      </div>
      <div style={T.content}>
        {(userEmail || userPhone) && (
          <div style={{ ...T.infoBox, display: "flex", alignItems: "center", gap: 8 }}>
            🔐 Logged in as <strong style={{ marginLeft: 2 }}>{userName}</strong>
            {userEmail && <span style={{ color: C.gray400 }}>{userEmail}</span>}
            {isReturning && <span style={T.badge(C.green, C.greenL, "#86efac")}>Returning User</span>}
          </div>
        )}
        {editable && <div style={{ ...T.infoBox, background: "#fff7ed", borderColor: "#fed7aa", color: C.orangeD }}>✏️ Edit mode — correct OCR errors. <span style={{ color: C.red }}>*</span> = required</div>}

        {/* Show a notice if gender/dob were auto-filled */}
        {(form.gender || form.dob) && (
          <div style={{ ...T.infoBox, background: "#f0fdf4", borderColor: "#86efac", color: C.green, marginBottom: 12 }}>
            ✅ Gender and Date of Birth were extracted from your Aadhaar.
            {!form.gender && " ⚠️ Gender not detected — please select manually."}
            {!form.dob && " ⚠️ Date of birth not detected — please enter manually."}
          </div>
        )}

        <RSection title="Personal Details" icon="👤">
          <RF label="Aadhaar Number" v={form.aadhaar_number} editable={false} err={fieldErrors.aadhaar_number} />
          <RF label="First Name *" v={form.first_name} ch={v => update("first_name", v)} editable={editable} err={fieldErrors.first_name} />
          <RF label="Middle Name" v={form.middle_name} ch={v => update("middle_name", v)} editable={editable} />
          <RF label="Last Name *" v={form.last_name} ch={v => update("last_name", v)} editable={editable} err={fieldErrors.last_name} />
          {/* Gender — always editable so user can fix if OCR got it wrong */}
          <RS label="Gender *" v={form.gender} opts={["Male", "Female", "Other"]} ch={v => update("gender", v)} editable={true} err={fieldErrors.gender} />
          {/* DOB — always editable */}
          <RF label="Date of Birth *" type="date" v={form.dob} ch={v => update("dob", v)} editable={true} err={fieldErrors.dob} />
        </RSection>

        <RSection title="Contact Details" icon="📞">
          <RF label="Mobile *" v={form.mobile} ch={v => update("mobile", v)} editable={editable} err={fieldErrors.mobile} />
          <RF label="Email *" v={form.email} ch={v => update("email", v)} editable={editable} err={fieldErrors.email} />
          {(userEmail || userPhone) && <div style={{ fontSize: 11, color: C.blue, gridColumn: "1/-1" }}>ℹ️ Pre-filled from your login</div>}
          <RF label="Father's First Name" v={form.fatherFirstName} ch={v => update("fatherFirstName", v)} editable={editable} />
          <RF label="Father's Last Name" v={form.fatherLastName} ch={v => update("fatherLastName", v)} editable={editable} />
        </RSection>

        <RSection title="Address" icon="🏠">
          <RF label="Flat / Door No" v={form.flat_no} ch={v => update("flat_no", v)} editable={editable} />
          <RF label="Building / Village" v={form.building} ch={v => update("building", v)} editable={editable} />
          <RF label="Road / Street" v={form.road} ch={v => update("road", v)} editable={editable} />
          <RF label="Area / Locality *" v={form.locality} ch={v => update("locality", v)} editable={editable} err={fieldErrors.locality} />
          <RF label="Town / City *" v={form.city} ch={v => update("city", v)} editable={editable} err={fieldErrors.city} />
          <RF label="District" v={form.district} ch={v => update("district", v)} editable={editable} />
          <RS label="Resident State *" v={form.residentState} ch={v => update("residentState", v)} editable={editable} err={fieldErrors.residentState}
            opts={["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"]} />
          <RF label="Pincode *" v={form.pincode} ch={v => update("pincode", v)} editable={editable} err={fieldErrors.pincode} />
        </RSection>

        <RSection title="Source of Income *" icon="💰">
          {fieldErrors.income && <div style={{ color: C.red, fontSize: 12, marginBottom: 8, gridColumn: "1/-1" }}>{fieldErrors.income}</div>}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, gridColumn: "1/-1" }}>
            {[{ key: "income_salary", label: "Salary" }, { key: "income_capital_gains", label: "Capital Gains" }, { key: "income_house_property", label: "House Property" }, { key: "income_other_sources", label: "Other Sources" }, { key: "income_business", label: "Business / Profession" }, { key: "income_no_income", label: "No Income" }].map(({ key, label }) => (
              <label key={key} style={{ display: "flex", alignItems: "center", cursor: "pointer", padding: "7px 10px", borderRadius: 8, background: C.white, border: `1px solid ${C.gray100}` }}>
                <input type="checkbox" checked={form[key] || false} onChange={() => updateCheck(key)} style={{ marginRight: 8, accentColor: C.orange, width: 15, height: 15 }} />
                <span style={{ fontSize: 13, color: C.gray800 }}>{label}</span>
              </label>
            ))}
          </div>
        </RSection>

        <RSection title="AO Code" icon="🏛️">
          <div style={{ fontSize: 11, color: C.blue, gridColumn: "1/-1", marginBottom: 4 }}>ℹ️ Auto-filled based on your district / locality. Edit if incorrect.</div>
          <RF label="Area Code" v={form.areaCode} ch={v => update("areaCode", v)} editable={editable} />
          <RF label="AO Type" v={form.aoType} ch={v => update("aoType", v)} editable={editable} />
          <RF label="Range Code" v={form.rangeCode} ch={v => update("rangeCode", v)} editable={editable} />
          <RF label="AO Number" v={form.aoNumber} ch={v => update("aoNumber", v)} editable={editable} />
        </RSection>

        <RSection title="Representative Assessee" icon="📋">
          <div style={{ fontSize: 11, color: C.blue, gridColumn: "1/-1", marginBottom: 4 }}>ℹ️ Auto-filled from father's name. Edit if needed.</div>
          <RF label="First Name" v={form.rep_first} ch={v => update("rep_first", v)} editable={editable} />
          <RF label="Last Name" v={form.rep_last} ch={v => update("rep_last", v)} editable={editable} />
          <RF label="Area / Locality" v={form.rep_area} ch={v => update("rep_area", v)} editable={editable} />
          <RF label="District" v={form.rep_district} ch={v => update("rep_district", v)} editable={editable} />
          <RS label="State" v={form.rep_state} ch={v => update("rep_state", v)} editable={editable}
            opts={["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"]} />
          <RS label="Declaration" v={form.declaration} ch={v => update("declaration", v)} editable={editable} opts={["Himself / Herself", "Authorized Signatory"]} />
          <RF label="Verifier Name" v={form.verifier} ch={v => update("verifier", v)} editable={editable} />
          <RF label="Verification Place" v={form.verification_place} ch={v => update("verification_place", v)} editable={editable} />
        </RSection>

        {error && <div style={T.errBox}>{error}</div>}
        {showRedirectMsg && <div style={T.okBox}>✅ Browser is auto-filling your PAN application. You'll be redirected when payment completes.</div>}

        <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
          <button style={T.secondaryBtn} onClick={() => { setPage("upload"); setShowRedirectMsg(false); setFieldErrors({}); }}>← Back</button>
          <button style={{ ...T.primaryBtn, flex: 1, opacity: loading || showRedirectMsg ? .6 : 1 }} onClick={handleConfirm} disabled={loading || showRedirectMsg}>
            {loading ? "Processing..." : "Confirm & Proceed →"}
          </button>
        </div>
      </div>
    </div>
  );

  /* ════════════════════════════════════════════════════════════
     SUCCESS PAGE
  ════════════════════════════════════════════════════════════ */
  if (page === "success") return (
    <div style={T.wrap}>
      <Navbar userName={userName} onLogout={resetAll} />
      <div style={{ ...T.content, maxWidth: 600, textAlign: "center", paddingTop: 60 }}>
        <div style={{ width: 90, height: 90, borderRadius: "50%", background: "linear-gradient(135deg,#d4f0dc,#a8e6b5)", margin: "0 auto 20px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 42, boxShadow: "0 8px 30px rgba(34,197,94,0.25)" }}>🎉</div>
        <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 32, color: C.navyDark, marginBottom: 8 }}>Payment Successful!</h1>
        <p style={{ fontSize: 15, color: C.gray600, marginBottom: 6 }}>Your PAN application has been submitted successfully.</p>
        <p style={{ fontSize: 15, color: C.gray800, marginBottom: 32 }}>Thank you, <strong style={{ color: C.orange }}>{applicantName}</strong>!</p>
        {paymentRef && (
          <div style={{ background: "linear-gradient(135deg,#fff7ed,#ffedd5)", border: "2px solid #fed7aa", borderRadius: 16, padding: "22px 44px", display: "inline-block", marginBottom: 28, boxShadow: "0 4px 20px rgba(243,115,33,0.15)" }}>
            <div style={{ fontSize: 11, color: "#9a6300", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>Reference Number</div>
            <div style={{ fontSize: 26, fontWeight: 800, color: "#c2410c", letterSpacing: 2, fontFamily: "'Playfair Display', serif" }}>{paymentRef}</div>
          </div>
        )}
        <p style={{ fontSize: 13, color: C.gray400, marginBottom: 32 }}>Your PAN card will be dispatched within 15–20 working days.</p>
        <button style={{ ...T.primaryBtn, width: "auto", padding: "13px 40px", display: "inline-flex" }} onClick={resetAll}>Apply for Another PAN</button>
      </div>
    </div>
  );

  return null;
}