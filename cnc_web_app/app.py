import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CNC Condition Monitoring",
    page_icon="⚙️",
    layout="wide",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (tất cả class name dùng tĩnh)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & Base ── */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif !important;
}

/* ── Sidebar polish ── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}
section[data-testid="stSidebar"] input {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    text-transform: none !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
    margin: 16px 0 !important;
}

/* ── Main area ── */
.block-container {
    padding-top: 1.5rem !important;
    max-width: 900px !important;
}

/* ── Shared card ── */
.cnc-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.cnc-card-dark {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 9999px;
    letter-spacing: .03em;
}
.badge-green  { background:#dcfce7; color:#15803d; }
.badge-yellow { background:#fef3c7; color:#b45309; }
.badge-red    { background:#fee2e2; color:#dc2626; }

/* ── Progress bar (static inline) ── */
.pbar-wrap {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0 4px;
}
.pbar-green  { background:#10b981; height:100%; border-radius:9999px; }
.pbar-yellow { background:#f59e0b; height:100%; border-radius:9999px; }
.pbar-red    { background:#ef4444; height:100%; border-radius:9999px; }

/* ── Parameter rows ── */
.param-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 2px;
}
.param-name  { font-size:14px; font-weight:600; color:#1e293b; }
.param-value { font-size:14px; font-weight:700; color:#0f172a; }
.param-desc  { font-size:12px; color:#64748b; margin-bottom:14px; }

/* ── Risk score big number ── */
.risk-number { font-size:52px; font-weight:800; line-height:1; }
.risk-green  { color:#15803d; }
.risk-yellow { color:#b45309; }
.risk-red    { color:#dc2626; }

/* ── Metric KPI boxes ── */
.kpi-row { display:flex; gap:12px; margin-bottom:20px; }
.kpi-box {
    flex:1;
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:14px 16px;
    text-align:center;
}
.kpi-label { font-size:11px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.kpi-value { font-size:20px; font-weight:800; color:#0f172a; margin-top:4px; }

/* ── Feature rows (Explainability) ── */
.feat-row {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 0;
    border-bottom:1px solid #f1f5f9;
}
.feat-name  { font-size:13px; color:#334155; }
.feat-val   { font-size:13px; font-weight:700; color:#0f172a; }
.feat-tag-red    { font-size:11px; color:#dc2626; font-weight:600; }
.feat-tag-yellow { font-size:11px; color:#b45309; font-weight:600; }
.feat-tag-gray   { font-size:11px; color:#94a3b8; }

/* ── Recommendation box ── */
.rec-box {
    background:#f0fdf4;
    border:1px solid #bbf7d0;
    border-radius:10px;
    padding:14px 18px;
    margin-top:20px;
}
.rec-box-warn {
    background:#fffbeb;
    border:1px solid #fde68a;
    border-radius:10px;
    padding:14px 18px;
    margin-top:20px;
}
.rec-box-danger {
    background:#fff1f2;
    border:1px solid #fecdd3;
    border-radius:10px;
    padding:14px 18px;
    margin-top:20px;
}
.rec-title  { font-size:13px; font-weight:700; color:#0f172a; margin-bottom:6px; }
.rec-body   { font-size:13px; color:#334155; line-height:1.6; }
.rec-note   { font-size:11px; color:#94a3b8; margin-top:6px; }

/* ── Section headers ── */
.section-header {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 24px 0 12px;
}

/* ── Divider ── */
.divider { border:none; border-top:1px solid #e2e8f0; margin:24px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
MODEL_FILE = "best_model.joblib"
BEST_THRESHOLD = 0.20

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

model_package = load_model()


def predict_risk(feat: dict):
    """Return (probability 0-1, threshold 0-1)"""
    df = pd.DataFrame([feat])
    if model_package and "pipeline" in model_package:
        p = model_package["pipeline"].predict_proba(df)[0, 1]
        t = model_package.get("threshold", BEST_THRESHOLD)
        return float(p), float(t)

    # Physics-based fallback (calibrated to notebook results)
    torque = feat["Torque [Nm]"]
    wear   = feat["Tool wear [min]"]
    tdiff  = feat["temp_diff"]
    rpm    = feat["Rotational speed [rpm]"]

    torque_r    = max(0.0, (torque - 50) / 25) if torque > 50 else 0.0
    wear_r      = max(0.0, (wear - 180) / 60)  if wear > 180 else 0.0
    heat_r      = 0.4 if (tdiff < 8.6 and rpm <= 1380) else 0.0
    overstrain_r = max(0.0, (wear * torque - 11000) / 4000) if wear * torque > 11000 else 0.0

    p = np.clip(0.03 + 0.35 * torque_r + 0.35 * wear_r + 0.2 * overstrain_r + 0.1 * heat_r, 0.02, 0.97)
    return float(p), BEST_THRESHOLD


# ─────────────────────────────────────────────
# SIDEBAR  ─  Input Panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CNC Monitor")
    st.markdown("<div style='color:#475569;font-size:12px;margin-bottom:16px;'>Nhập thông số cảm biến vận hành</div>", unsafe_allow_html=True)

    preset = st.radio(
        "Kịch bản nhanh",
        ["🔴  Báo động (CNC-002)", "🟢  Bình thường (CNC-001)", "✏️  Tùy chỉnh"],
        index=0,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    if "Báo động" in preset:
        v = dict(id="CNC-002", tp="L", air=300.0, proc=315.0, rpm=1500, torque=65.0, wear=210)
    elif "Bình thường" in preset:
        v = dict(id="CNC-001", tp="M", air=298.0, proc=308.0, rpm=1550, torque=38.0, wear=45)
    else:
        v = dict(id="CNC-003", tp="L", air=300.0, proc=310.0, rpm=1500, torque=40.0, wear=100)

    machine_id   = st.text_input("Mã máy",                     value=v["id"])
    machine_type = st.selectbox("Loại phôi",                   ["L", "M", "H"],
                                index=["L","M","H"].index(v["tp"]))
    air_temp     = st.slider("Nhiệt độ môi trường [K]",        295.0, 305.0, v["air"],  0.5)
    proc_temp    = st.slider("Nhiệt độ quá trình [K]",         305.0, 325.0, v["proc"], 0.5)
    speed_rpm    = st.slider("Tốc độ trục chính [rpm]",        1100,  2900,  v["rpm"],  10)
    torque_nm    = st.slider("Mô-men xoắn [Nm]",               3.0,   77.0,  v["torque"], 0.5)
    tool_wear    = st.slider("Thời gian mòn dao [phút]",       0,     253,   v["wear"], 1)

    st.markdown("<hr/>", unsafe_allow_html=True)
    model_status = "✅ Model thật đã nạp" if model_package else "⚠️ Chế độ mô phỏng"
    st.caption(model_status)
    st.caption("Dataset: AI4I 2020 Predictive Maintenance")


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
temp_diff = proc_temp - air_temp
power_raw = speed_rpm * torque_nm
power_kw  = (power_raw * 2 * np.pi / 60) / 1000.0
overstrain = tool_wear * torque_nm

feat = {
    "Type":                      machine_type,
    "Air temperature [K]":       air_temp,
    "Process temperature [K]":   proc_temp,
    "Rotational speed [rpm]":    speed_rpm,
    "Torque [Nm]":               torque_nm,
    "Tool wear [min]":           tool_wear,
    "temp_diff":                 temp_diff,
    "power":                     power_raw,
}

prob, threshold = predict_risk(feat)
pct = int(round(prob * 100))


# ─────────────────────────────────────────────
# RISK LEVEL (GĐ8)
# ─────────────────────────────────────────────
if prob >= 0.70:
    lvl = "danger";  badge_text = "Báo động";      lvl_text = "Dừng máy, kiểm tra ngay"
    badge_cls = "badge-red";  bar_cls = "pbar-red";    num_cls = "risk-red"
elif prob >= threshold:
    lvl = "warning"; badge_text = "Cần kiểm tra";  lvl_text = "Theo dõi chặt ca này"
    badge_cls = "badge-yellow"; bar_cls = "pbar-yellow"; num_cls = "risk-yellow"
else:
    lvl = "normal";  badge_text = "Bình thường";   lvl_text = "Vận hành ổn định"
    badge_cls = "badge-green";  bar_cls = "pbar-green";  num_cls = "risk-green"

# ─────────────────────────────────────────────
# RECOMMENDATION (GĐ10)
# ─────────────────────────────────────────────
if lvl == "danger":
    rec_cls = "rec-box-danger"
    rec_icon = "🚨"
    rec_title = "Hành động khẩn: Dừng máy và kiểm tra"
    rec_items = []
    if torque_nm > 55:
        rec_items.append("• Giảm bước tiến dao hoặc chiều sâu cắt để hạ Torque xuống dưới 55 Nm.")
    if tool_wear > 180:
        rec_items.append("• Thay dao phay mới ngay — thời gian mòn tích lũy vượt ngưỡng an toàn.")
    if temp_diff < 8.6:
        rec_items.append("• Kiểm tra bơm và áp suất dung dịch tưới nguội (coolant).")
    if not rec_items:
        rec_items.append("• Kiểm tra toàn diện các cụm cơ khí trước khi khởi động lại.")
    rec_body = "\n".join(rec_items)
elif lvl == "warning":
    rec_cls = "rec-box-warn"
    rec_icon = "⚠️"
    rec_title = "Khuyến nghị: Theo dõi và chuẩn bị bảo dưỡng"
    rec_body = (
        "• Lên kế hoạch thay dao trong ca tới nếu Tool wear tiếp tục tăng.\n"
        "• Ghi nhận biến thiên Torque theo thời gian.\n"
        "• Đảm bảo nhiệt độ coolant và lưu lượng tưới nguội đủ tiêu chuẩn."
    )
else:
    rec_cls = "rec-box"
    rec_icon = "✅"
    rec_title = "Máy đang vận hành ổn định"
    rec_body = (
        "• Tiếp tục kế hoạch bảo dưỡng định kỳ theo lịch.\n"
        "• Ghi nhận chỉ số Tool wear hàng ca để phát hiện xu hướng sớm."
    )

# ─────────────────────────────────────────────
# RENDER MAIN CONTENT
# ─────────────────────────────────────────────

# ── Header ──────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"### Kết quả kiểm tra  `{machine_id}`")
    st.caption("Hệ thống dự báo tình trạng hư hỏng máy phay CNC  ·  AI4I 2020 Dataset")
with col_badge:
    st.markdown(f"""
    <div style="text-align:right;padding-top:12px;">
        <span class="badge {badge_cls}">{badge_text}</span>
    </div>""", unsafe_allow_html=True)


# ── Risk Score + Progress (GĐ8) ─────────────
st.markdown(f"""
<div class="cnc-card">
    <div style="display:flex;align-items:center;gap:24px;">
        <div>
            <div class="section-header">Điểm nguy cơ hư hỏng</div>
            <div class="risk-number {num_cls}">{pct}<span style="font-size:24px;font-weight:600;">%</span></div>
            <div style="font-size:13px;color:#64748b;margin-top:4px;">Ngưỡng cảnh báo: <b>{int(threshold*100)}%</b> &nbsp;|&nbsp; Mức hiện tại: <b>{lvl_text}</b></div>
        </div>
        <div style="flex:1;">
            <div class="pbar-wrap" style="height:14px;">
                <div class="{bar_cls}" style="width:{pct}%;height:100%;border-radius:9999px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-top:4px;">
                <span>0%</span>
                <span>Ngưỡng {int(threshold*100)}%</span>
                <span>Báo động 70%</span>
                <span>100%</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI Row ─────────────────────────────────
st.markdown('<div class="section-header">Thông số vận hành tổng quan</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value in [
    (k1, "Nhiệt độ QT", f"{proc_temp:.0f} K"),
    (k2, "Công suất",   f"{power_kw:.1f} kW"),
    (k3, "Mòn dao",     f"{tool_wear} min"),
    (k4, "Tốc độ",      f"{speed_rpm:,} rpm"),
    (k5, "Torque",      f"{torque_nm:.0f} Nm"),
]:
    col.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")   # spacing


# ── Parameter Analysis (GĐ9) ────────────────
st.markdown('<div class="section-header">Phân tích chi tiết thông số vận hành (GĐ9 · Parameter Analysis)</div>', unsafe_allow_html=True)

# helper: build one param block purely with inline style (no dynamic class names)
def param_block(label: str, value_str: str, pct_val: int,
                bar_color: str, status: str, detail: str = ""):
    """bar_color: hex string like #ef4444"""
    return f"""
<div style="margin-bottom:18px;">
    <div class="param-row">
        <span class="param-name">{label}</span>
        <span class="param-value">{value_str}</span>
    </div>
    <div class="pbar-wrap">
        <div style="width:{pct_val}%;height:100%;background:{bar_color};border-radius:9999px;"></div>
    </div>
    <div class="param-desc">{status}{(' — ' + detail) if detail else ''}</div>
</div>"""

proc_pct  = int(min(100, max(0, (proc_temp-295)/(325-295)*100)))
power_pct = int(min(100, max(0, power_kw/12*100)))
wear_pct  = int(min(100, max(0, tool_wear/253*100)))
speed_pct = int(min(100, max(0, (speed_rpm-1100)/(2900-1100)*100)))
torq_pct  = int(min(100, max(0, torque_nm/77*100)))

proc_color  = "#ef4444" if proc_temp > 313 else "#f59e0b" if proc_temp > 310 else "#10b981"
power_color = "#ef4444" if power_kw > 9 else "#f59e0b" if power_kw > 7.5 else "#10b981"
wear_color  = "#ef4444" if tool_wear > 200 else "#f59e0b" if tool_wear > 150 else "#10b981"
speed_color = "#10b981"
torq_color  = "#ef4444" if torque_nm > 65 else "#f59e0b" if torque_nm > 50 else "#10b981"

proc_status  = "Cần kiểm tra" if proc_temp > 313 else "Theo dõi" if proc_temp > 310 else "Bình thường"
power_status = "Vượt ngưỡng" if power_kw > 9 else "Cần theo dõi" if power_kw > 7.5 else "Trong vùng tham chiếu"
wear_status  = "Mức cao — cảnh báo" if tool_wear > 200 else "Cần chú ý" if tool_wear > 150 else "Mức an toàn"
speed_status = "Trong vùng tham chiếu"
torq_status  = "Vượt ngưỡng an toàn" if torque_nm > 65 else "Cần theo dõi" if torque_nm > 50 else "Bình thường"

left_col, right_col = st.columns(2)
with left_col:
    st.markdown(param_block("🌡️  Nhiệt độ quá trình", f"{proc_temp:.1f} K", proc_pct,
                             proc_color, proc_status, f"ΔT = {temp_diff:.1f} K"), unsafe_allow_html=True)
    st.markdown(param_block("⚡  Công suất cơ học",   f"{power_kw:.2f} kW", power_pct,
                             power_color, power_status), unsafe_allow_html=True)
    st.markdown(param_block("🔧  Mô-men xoắn",        f"{torque_nm:.1f} Nm", torq_pct,
                             torq_color, torq_status), unsafe_allow_html=True)

with right_col:
    st.markdown(param_block("🪛  Độ hao mòn dao",      f"{tool_wear} phút", wear_pct,
                             wear_color, wear_status, "Ngưỡng thay dao: 200–240 phút"), unsafe_allow_html=True)
    st.markdown(param_block("🔄  Tốc độ trục chính",   f"{speed_rpm:,} rpm", speed_pct,
                             speed_color, speed_status, "Định mức: 1 400–1 800 rpm"), unsafe_allow_html=True)


# ── Feature Importance / Explainability (GĐ7) ──
st.markdown('<div class="section-header">Phân tích đóng góp thông số vào dự báo (GĐ7 · Explainability)</div>', unsafe_allow_html=True)

factors = [
    ("Tool wear [min]",    tool_wear,    240,  "phút",  "Mòn dao gây ứng suất tích lũy"),
    ("Torque [Nm]",        torque_nm,    77,   "Nm",    "Lực cắt cao → tăng nguy cơ gãy dao"),
    ("Nhiệt độ quá trình", proc_temp-273.15, 52, "°C",  "Tản nhiệt kém → hư ổ trục, lớp phủ dao"),
    ("Công suất",          power_kw,     12,   "kW",   "Vượt dải công suất định mức"),
    ("ΔT nhiệt độ",        temp_diff,    30,   "K",    "Chênh lệch nhiệt giữa KT và QT"),
]

rows_html = ""
for name, val, maxv, unit, desc in factors:
    bar_w = int(min(100, max(0, val / maxv * 100)))
    # risk contribution: simple normalized importance
    contrib = bar_w
    if contrib >= 70:
        tag_cls = "feat-tag-red"; tag = "▲ Tăng nguy cơ"
    elif contrib >= 40:
        tag_cls = "feat-tag-yellow"; tag = "~ Cần theo dõi"
    else:
        tag_cls = "feat-tag-gray"; tag = "✓ Ổn định"

    bar_color = "#ef4444" if contrib >= 70 else "#f59e0b" if contrib >= 40 else "#10b981"
    rows_html += f"""
<div class="feat-row">
    <div style="flex:2;">
        <div class="feat-val">{name}</div>
        <div style="font-size:11px;color:#94a3b8;">{desc}</div>
        <div class="pbar-wrap" style="margin:4px 0 0;">
            <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:9999px;"></div>
        </div>
    </div>
    <div style="flex:0.8;text-align:right;padding-left:12px;">
        <div class="feat-val">{val:.1f} {unit}</div>
        <div class="{tag_cls}" style="margin-top:2px;">{tag}</div>
    </div>
</div>"""

st.markdown(f'<div class="cnc-card">{rows_html}</div>', unsafe_allow_html=True)


# ── Recommendation Engine (GĐ10) ────────────
st.markdown('<div class="section-header">Khuyến nghị hành động (GĐ10 · Recommendation Engine)</div>', unsafe_allow_html=True)
rec_body_html = rec_body.replace("\n", "<br>")
st.markdown(f"""
<div class="{rec_cls}">
    <div class="rec-title">{rec_icon} &nbsp; {rec_title}</div>
    <div class="rec-body">{rec_body_html}</div>
    <div class="rec-note">Dựa trên mô hình Random Forest (PR-AUC = 0.887, ngưỡng cắt tối ưu: 0.20).</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.caption("🤖 Mô hình: Random Forest Classifier")
c2.caption("📊 Dataset: AI4I 2020 Predictive Maintenance (10,000 records)")
c3.caption(f"⚙️ {'Model thật đã nạp' if model_package else 'Chế độ mô phỏng vật lý'}")
