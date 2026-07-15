import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU_METRICS = True
except Exception as e:
    print(f"[WARN] pynvml initialization failed: {e}. GPU metrics will be missing.")
    HAS_GPU_METRICS = False



def main():
    print("="*60)
    print("COMPREHENSIVE BENCHMARK - BBox V2")
    print("="*60)

    # 1. Configs
    video_path = "data/raw_videos/input_video.mp4"
    model_path = "models/yolov8m.onnx"
    out_dir = "data/outputs/benchmark"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    # 2. Initialization
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path, task="detect")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # 3. Data collectors
    performance_log = []
    resource_log = []
    detection_log = []
    
    # Temporal tracking collectors
    track_history = defaultdict(list) # track_id -> list of dicts (frame, conf, class_id, center)

    # Qualitative evaluation
    best_conf = -1.0
    worst_conf = 2.0
    best_frame_img = None
    worst_frame_img = None

    frame_idx = 0
    print("[INFO] Starting benchmark...")
    
    # Pre-warm PSUTIL
    psutil.cpu_percent()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t_start_latency = time.perf_counter()

        # Measure inference & tracking
        t0 = time.perf_counter()
        # Use YOLO's built-in ByteTrack integration
        results = model.track(frame, imgsz=640, conf=0.25, iou=0.7, persist=True, verbose=False)[0]
        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0
        # For simplicity in benchmarking, we attribute both inference and tracking to inference_time_ms
        tracking_time_ms = 0.0

        latency_ms = (time.perf_counter() - t_start_latency) * 1000.0

        # Extract tracked objects
        tracked_objects = []
        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                tid = int(track_id.cpu().numpy())
                tracked_objects.append([x1, y1, x2, y2, tid, conf, cls, 0])
        else:
            # Fallback if tracking fails for a frame (no objects tracked)
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                tracked_objects.append([x1, y1, x2, y2, -1, conf, cls, 0])

        # Performance Logging
        performance_log.append({
            "frame_idx": frame_idx,
            "inference_time_ms": inference_time_ms,
            "tracking_time_ms": tracking_time_ms,
            "latency_ms": latency_ms,
        })

        # Resource Logging
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        gpu_util = 0.0
        vram_mb = 0.0
        if HAS_GPU_METRICS:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_util = float(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                vram_mb = mem.used / (1024**2)
            except:
                pass
        
        resource_log.append({
            "frame_idx": frame_idx,
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "gpu_util": gpu_util,
            "vram_used_mb": vram_mb
        })

        # Detection & Temporal Stats
        frame_confs = []
        class_counts = defaultdict(int)
        
        for trk in tracked_objects:
            x1, y1, x2, y2, track_id, conf, cls, _ = trk
            
            center = ((x1 + x2)/2, (y1 + y2)/2)
            track_history[int(track_id)].append({
                "frame": frame_idx,
                "conf": conf,
                "class_id": int(cls),
                "center": center
            })
            
            frame_confs.append(conf)
            class_counts[int(cls)] += 1

        avg_conf = np.mean(frame_confs) if len(frame_confs) > 0 else 0.0
        detection_log.append({
            "frame_idx": frame_idx,
            "total_detections": len(tracked_objects),
            "avg_confidence": avg_conf,
            "motorcycles": class_counts.get(3, 0),
            "persons": class_counts.get(0, 0),
        })

        # Qualitative assessment
        if len(tracked_objects) > 3:
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_frame_img = frame.copy()
            
            if avg_conf < worst_conf and avg_conf > 0:
                worst_conf = avg_conf
                worst_frame_img = frame.copy()

        if frame_idx % 200 == 0:
            print(f"  [Progress] Processed {frame_idx}/{total_frames} frames")
            
        frame_idx += 1

    cap.release()

    # ==========================================
    # DATA AGGREGATION & TEMPORAL STABILITY
    # ==========================================
    print("[INFO] Aggregating results...")
    df_perf = pd.DataFrame(performance_log)
    df_res = pd.DataFrame(resource_log)
    df_det = pd.DataFrame(detection_log)

    # 1. Performance
    avg_inf = df_perf["inference_time_ms"].mean()
    avg_lat = df_perf["latency_ms"].mean()
    avg_fps = 1000.0 / avg_lat

    # 2. Temporal Stability Metrics
    track_lifespans = []
    conf_variances = []
    class_switches = 0
    total_tracks_evaluated = 0
    jitter_values = []

    for tid, history in track_history.items():
        if len(history) < 5:
            continue
        
        total_tracks_evaluated += 1
        track_lifespans.append(len(history))
        
        confs = [h["conf"] for h in history]
        conf_variances.append(np.var(confs))
        
        classes = [h["class_id"] for h in history]
        if len(set(classes)) > 1:
            class_switches += 1
            
        # Jitter: Euclidean distance of center frame-to-frame
        centers = [h["center"] for h in history]
        diffs = [np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1])) for i in range(1, len(centers))]
        jitter_values.extend(diffs)

    # ==========================================
    # REPORT GENERATION & PLOTTING
    # ==========================================
    
    # Save CSVs
    df_perf.to_csv(f"{out_dir}/performance.csv", index=False)
    df_res.to_csv(f"{out_dir}/resource_usage.csv", index=False)
    df_det.to_csv(f"{out_dir}/detection_stats.csv", index=False)
    
    # Save Qualitative Images
    if best_frame_img is not None:
        cv2.imwrite(f"{out_dir}/best_qualitative_frame.jpg", best_frame_img)
    if worst_frame_img is not None:
        cv2.imwrite(f"{out_dir}/worst_qualitative_frame.jpg", worst_frame_img)
        
    report = f"""
BENCHMARK REPORT: BBox V2 ONNX Model
====================================
1. Performance
------------------------------------
Avg FPS             : {avg_fps:.2f}
Avg Inference Time  : {avg_inf:.2f} ms/frame
Avg Latency         : {avg_lat:.2f} ms/frame

2. Resource Usage
------------------------------------
Avg CPU Usage       : {df_res['cpu_usage'].mean():.2f} %
Avg RAM Usage       : {df_res['ram_usage'].mean():.2f} %
Avg GPU Utilization : {df_res['gpu_util'].mean():.2f} %
Avg VRAM Used       : {df_res['vram_used_mb'].mean():.2f} MB

3. Detection Statistics
------------------------------------
Avg Detections/Frame: {df_det['total_detections'].mean():.2f}
Max Detections/Frame: {df_det['total_detections'].max()}
Avg Confidence      : {df_det['avg_confidence'][df_det['avg_confidence']>0].mean():.3f}
Avg Motorcycles     : {df_det['motorcycles'].mean():.2f}
Avg Persons         : {df_det['persons'].mean():.2f}

4. Temporal Stability
------------------------------------
Total Stable Tracks (>5 frames): {total_tracks_evaluated}
Avg Detection Persistence      : {np.mean(track_lifespans):.1f} frames
Avg Confidence Variance        : {np.mean(conf_variances):.4f}
Class Switching Rate           : {(class_switches / max(1, total_tracks_evaluated)) * 100:.2f} %
Avg BBox Jitter (pixels/frame) : {np.mean(jitter_values):.2f} px
====================================
"""
    with open(f"{out_dir}/benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    # Plotting
    sns.set_style("darkgrid")
    
    # Plot 1: Performance
    plt.figure(figsize=(10, 5))
    plt.plot(df_perf["frame_idx"], df_perf["latency_ms"], label="Latency (ms)", alpha=0.7)
    plt.plot(df_perf["frame_idx"], df_perf["inference_time_ms"], label="Inference Time (ms)", alpha=0.7)
    plt.title("Performance over time")
    plt.xlabel("Frame")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.savefig(f"{out_dir}/plot_performance.png")
    plt.close()

    # Plot 2: Resource Usage
    plt.figure(figsize=(10, 5))
    if HAS_GPU_METRICS:
        plt.plot(df_res["frame_idx"], df_res["gpu_util"], label="GPU Util (%)", color='green', alpha=0.7)
    plt.plot(df_res["frame_idx"], df_res["cpu_usage"], label="CPU Usage (%)", color='orange', alpha=0.7)
    plt.plot(df_res["frame_idx"], df_res["ram_usage"], label="RAM Usage (%)", color='blue', alpha=0.7)
    plt.title("Resource Usage over time")
    plt.xlabel("Frame")
    plt.ylabel("Usage (%)")
    plt.legend()
    plt.savefig(f"{out_dir}/plot_resources.png")
    plt.close()
    
    # Plot 3: Detection Stats
    plt.figure(figsize=(10, 5))
    plt.plot(df_det["frame_idx"], df_det["motorcycles"], label="Motorcycles", color='blue')
    plt.plot(df_det["frame_idx"], df_det["avg_confidence"] * 10, label="Avg Conf x10", color='red', alpha=0.5)
    plt.title("Detections & Confidence over time")
    plt.xlabel("Frame")
    plt.ylabel("Count / Confidence")
    plt.legend()
    plt.savefig(f"{out_dir}/plot_detections.png")
    plt.close()

    print(f"[DONE] All benchmark outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
