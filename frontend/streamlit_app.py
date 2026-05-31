"""
FPT Smart Parking CV — Streamlit Dashboard
===========================================
Simple demo dashboard for the motorcycle parking monitoring system.

Usage:
    streamlit run frontend/streamlit_app.py
"""

import os
import sys
import subprocess

import streamlit as st
import pandas as pd
import queue
import threading
import cv2

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FRONTEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.core.utils import load_config
from backend.core.video_processor import VideoProcessor


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FPT Smart Parking CV",
    page_icon="🏍️",
    layout="wide",
)

st.title("🏍️ FPT Smart Parking CV — Dashboard")
st.markdown("Hệ thống giám sát trạng thái đỗ xe máy bằng Deep Learning")

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Cài đặt")

video_path = st.sidebar.text_input(
    "Video path",
    value="data/raw_videos/input_video.mp4",
)
model_path = st.sidebar.text_input(
    "Model path",
    value="models/best.pt",
)
config_path = st.sidebar.text_input(
    "Config path",
    value="backend/config.yaml",
)

# Show current config
config_abs = os.path.join(_PROJECT_ROOT, config_path)
if os.path.isfile(config_abs):
    with st.sidebar.expander("📄 Config hiện tại", expanded=False):
        with open(config_abs, "r", encoding="utf-8") as f:
            st.code(f.read(), language="yaml")


# ── Run Inference ────────────────────────────────────────────────────────────

st.header("🚀 Chạy Inference")

col1, col2 = st.columns([1, 3])
with col1:
    run_btn = st.button("▶️ Run Inference", type="primary")

if run_btn:
    st.info("Đang xử lý video qua Multithreading (Producer-Consumer)...")
    
    # Setup queue
    frame_queue = queue.Queue(maxsize=5)
    config = load_config(config_abs)
    
    # Force don't show_cv2 for Streamlit UI
    config["show_cv2"] = False
    
    def producer(cfg, q):
        try:
            vp = VideoProcessor(cfg, frame_queue=q)
            vp.process()
        except Exception as e:
            print(f"[ERROR] Producer thread failed: {e}")
            q.put(None)
            
    # Start producer thread
    t = threading.Thread(target=producer, args=(config, frame_queue), daemon=True)
    t.start()
    
    # Streamlit image placeholder
    st_frame = st.empty()
    
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
    st.success("✅ Inference hoàn tất!")

st.divider()

# ── Output Video ─────────────────────────────────────────────────────────────

st.header("🎬 Video đã annotate")

output_video = os.path.join(_PROJECT_ROOT, "data", "outputs", "annotated_video.mp4")

if os.path.isfile(output_video):
    with open(output_video, 'rb') as f:
        st.video(f.read())
else:
    st.info("Chưa có video output. Nhấn **Run Inference** để tạo.")

st.divider()

# ── Metrics ──────────────────────────────────────────────────────────────────

st.header("📊 Kết quả phân tích")

metrics_path = os.path.join(_PROJECT_ROOT, "data", "outputs", "metrics.csv")
gaps_path = os.path.join(_PROJECT_ROOT, "data", "outputs", "gap_log.csv")
summary_path = os.path.join(_PROJECT_ROOT, "data", "outputs", "detection_summary.csv")

if os.path.isfile(metrics_path):
    df_metrics = pd.read_csv(metrics_path)

    # Summary KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tổng frames", len(df_metrics))
    col2.metric("FPS trung bình", f"{df_metrics['fps'].mean():.1f}")
    col3.metric("Xe máy TB", f"{df_metrics['motorcycle_count'].mean():.1f}")
    col4.metric("Max gaps", int(df_metrics['available_gap_count'].max()))

    # Gap status changes
    if os.path.isfile(gaps_path):
        df_gaps = pd.read_csv(gaps_path)
        changes = 0
        for _, grp in df_gaps.groupby("gap_id"):
            statuses = grp["status"].tolist()
            for i in range(1, len(statuses)):
                if statuses[i] != statuses[i - 1]:
                    changes += 1
        col5.metric("Gap changes", changes)
    else:
        col5.metric("Gap changes", "N/A")

    st.subheader("Metrics theo thời gian")

    tab1, tab2 = st.tabs(["📈 Biểu đồ", "📋 Bảng dữ liệu"])

    with tab1:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.line_chart(
                df_metrics.set_index("timestamp")[["motorcycle_count", "person_count"]],
                width="stretch",
            )
        with chart_col2:
            st.line_chart(
                df_metrics.set_index("timestamp")[["available_gap_count"]],
                color="#FFD700",
                width="stretch",
            )

    with tab2:
        st.dataframe(df_metrics.tail(100), width="stretch", height=400)

    # Gap log
    if os.path.isfile(gaps_path):
        with st.expander("🔍 Gap Log chi tiết (100 dòng cuối)"):
            st.dataframe(pd.read_csv(gaps_path).tail(100), width="stretch", height=300)

else:
    st.info("Chưa có metrics. Nhấn **Run Inference** để tạo.")

# ── Debug Screenshot ─────────────────────────────────────────────────────────

st.divider()
st.header("🖼️ Debug Screenshot")

ss_path = os.path.join(_PROJECT_ROOT, "data", "outputs", "screenshots", "pipeline_debug.jpg")
if os.path.isfile(ss_path):
    st.image(ss_path, caption="Pipeline debug frame", width="stretch")
else:
    st.info("Chưa có screenshot debug.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("FPT Smart Parking CV — Đồ án sinh viên | YOLO + BEV + Tracking")
