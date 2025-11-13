import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

# -------------------------
# Video analysis functions
# -------------------------
def analyze_video(video_path, freeze_threshold=0.5, downscale_width=640):
    """
    Analyze a single video and return (results_dict, motion_values, fps).
    results_dict contains keys: Speed (px/sec), Distance (px), Freezing (%),
    Movement Index, Duration (s), Frames, FPS
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open the video: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame from: " + video_path)

    # Resize to speed up processing while preserving aspect ratio
    if downscale_width is not None:
        h, w = prev_frame.shape[:2]
        scale = downscale_width / float(w)
        prev_frame = cv2.resize(prev_frame, (downscale_width, int(h * scale)))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    total_motion = 0.0
    motion_values = []
    freeze_frames = 0
    frame_count = 0
    total_distance = 0.0
    prev_centroid = None

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if downscale_width is not None:
            h, w = frame.shape[:2]
            scale = downscale_width / float(w)
            frame = cv2.resize(frame, (downscale_width, int(h * scale)))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow (Farneback) for motion intensity
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, frame_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = float(np.mean(mag))
        motion_values.append(motion)
        total_motion += motion

        if motion < freeze_threshold:
            freeze_frames += 1

        # Centroid detection using Otsu threshold on the grayscale frame
        _, thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"])
            cy = (M["m01"] / M["m00"])
            centroid = (cx, cy)
            if prev_centroid is not None:
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                total_distance += dist
            prev_centroid = centroid

        prev_gray = frame_gray
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise RuntimeError("No frames processed for video: " + video_path)

    duration_sec = frame_count / fps if fps > 0 else frame_count / 30.0
    freezing_pct = (freeze_frames / frame_count) * 100.0
    avg_motion = total_motion / frame_count
    speed_px_sec = total_distance / duration_sec if duration_sec > 0 else 0.0

    results = {
        "Speed (px/sec)": round(speed_px_sec, 2),
        "Distance (px)": round(total_distance, 2),
        "Freezing (%)": round(freezing_pct, 2),
        "Movement Index": round(avg_motion, 5),
        "Duration (s)": round(duration_sec, 2),
        "Frames": int(frame_count),
        "FPS": round(fps, 2)
    }

    return results, motion_values, fps

# -------------------------
# GUI / App logic
# -------------------------
class ZFishApp:
    def __init__(self, root):
        self.root = root
        root.title("Zebrafish Behavior Analyzer (OpenCV)")
        root.geometry("640x420")

        self.video1 = None
        self.video2 = None
        self.latest_results = None  # will store combined dict for export

        # ---- Controls ----
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        tk.Label(top, text="Study duration (sec):").pack(side="left")
        self.duration_entry = tk.Entry(top, width=6)
        self.duration_entry.insert(0, "20")
        self.duration_entry.pack(side="left", padx=(6,12))

        btn_frame = tk.Frame(top)
        btn_frame.pack(side="left")

        tk.Button(btn_frame, text="Select Video 1 (Pre)", command=self.select_video1).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text="Select Video 2 (Post, optional)", command=self.select_video2).grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text="Run Analysis (Single)", command=self.run_single_thread).grid(row=1, column=0, pady=8)
        tk.Button(btn_frame, text="Run Comparison (2 videos)", command=self.run_compare_thread).grid(row=1, column=1, pady=8)

        # ---- Selected files labels ----
        self.label_v1 = tk.Label(root, text="Video 1: None")
        self.label_v1.pack(anchor="w", padx=10)
        self.label_v2 = tk.Label(root, text="Video 2: None")
        self.label_v2.pack(anchor="w", padx=10)

        # ---- Results treeview (Metric | PRE | POST)----
        self.tree = ttk.Treeview(root, columns=("metric", "pre", "post"), show="headings", height=9)
        self.tree.heading("metric", text="Metric")
        self.tree.heading("pre", text="PRE")
        self.tree.heading("post", text="POST")
        self.tree.column("metric", width=220)
        self.tree.column("pre", width=140, anchor="center")
        self.tree.column("post", width=140, anchor="center")
        self.tree.pack(padx=10, pady=10, fill="x")

        # ---- Export button ----
        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=8, pady=6)
        tk.Button(bottom, text="Export Results CSV", command=self.export_csv).pack(side="left")
        tk.Button(bottom, text="Plot last motion timecourse", command=self.plot_last_motion).pack(side="left", padx=8)

    # ---- File selectors ----
    def select_video1(self):
        path = filedialog.askopenfilename(title="Select Video 1", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")])
        if path:
            self.video1 = path
            self.label_v1.config(text=f"Video 1: {os.path.basename(path)}")

    def select_video2(self):
        path = filedialog.askopenfilename(title="Select Video 2", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")])
        if path:
            self.video2 = path
            self.label_v2.config(text=f"Video 2: {os.path.basename(path)}")

    # ---- Thread wrappers ----
    def run_single_thread(self):
        if not self.video1:
            messagebox.showerror("Error", "Select Video 1 first.")
            return
        t = threading.Thread(target=self.run_single)
        t.daemon = True
        t.start()

    def run_compare_thread(self):
        if not self.video1 or not self.video2:
            messagebox.showerror("Error", "Select both Video 1 and Video 2 for comparison.")
            return
        t = threading.Thread(target=self.run_compare)
        t.daemon = True
        t.start()

    # ---- Core run functions ----
    def run_single(self):
        self._set_busy(True)
        try:
            start = time.time()
            r1, mv1, fps1 = analyze_video(self.video1, freeze_threshold=0.5, downscale_width=640)
            elapsed = time.time() - start

            # Format as Pre-only combined dict for display/export
            combined = {f"Pre-{k}": v for k, v in r1.items()}
            self.latest_results = (combined, (mv1,), (fps1,), (self.video1,))

            self.update_tree(combined)
            messagebox.showinfo("Done", f"Single analysis finished in {elapsed:.1f}s")
            self.plot_metrics(r1, title=os.path.basename(self.video1))
            self.plot_motion_timecourse(mv1, fps1, title=os.path.basename(self.video1))

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self._set_busy(False)

    def run_compare(self):
        self._set_busy(True)
        try:
            start = time.time()
            r1, mv1, fps1 = analyze_video(self.video1, freeze_threshold=0.5, downscale_width=640)
            r2, mv2, fps2 = analyze_video(self.video2, freeze_threshold=0.5, downscale_width=640)
            elapsed = time.time() - start

            combined = {}
            for k, v in r1.items():
                combined[f"Pre-{k}"] = v
            for k, v in r2.items():
                combined[f"Post-{k}"] = v

            self.latest_results = (combined, (mv1, mv2), (fps1, fps2), (self.video1, self.video2))
            self.update_tree(combined)
            messagebox.showinfo("Done", f"Comparison finished in {elapsed:.1f}s")

            # plots
            self.plot_compare(r1, r2, labels=(os.path.basename(self.video1), os.path.basename(self.video2)))
            plt.figure(figsize=(9,4))
            plt.plot(np.arange(len(mv1)) / fps1, mv1, label="Pre")
            plt.plot(np.arange(len(mv2)) / fps2, mv2, label="Post")
            plt.xlabel("Time (s)")
            plt.ylabel("Motion magnitude")
            plt.legend()
            plt.title("Motion timecourse (Pre vs Post)")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self._set_busy(False)

    # ---- UI helpers ----
    def update_tree(self, combined_results):
        """
        combined_results: dict with keys like "Pre-Speed (px/sec)" and "Post-Speed (px/sec)"
        This function populates the 3-column tree with Metric | PRE | POST
        """
        # clear
        for i in self.tree.get_children():
            self.tree.delete(i)

        metrics = {}
        for key, val in combined_results.items():
            if key.startswith("Pre-"):
                metric = key.replace("Pre-", "")
                metrics.setdefault(metric, {})["PRE"] = val
            elif key.startswith("Post-"):
                metric = key.replace("Post-", "")
                metrics.setdefault(metric, {})["POST"] = val
            else:
                # fallback: single mode keys
                metrics.setdefault(key, {})["PRE"] = val

        # stable ordering for presentation
        order = ["Speed (px/sec)", "Distance (px)", "Freezing (%)", "Movement Index", "Duration (s)", "Frames", "FPS"]
        for metric in order:
            if metric in metrics:
                vals = metrics[metric]
                self.tree.insert("", "end", values=(metric, vals.get("PRE", ""), vals.get("POST", "")))

        # insert any remaining metrics not in order
        for metric, vals in metrics.items():
            if metric not in order:
                self.tree.insert("", "end", values=(metric, vals.get("PRE", ""), vals.get("POST", "")))

    def export_csv(self):
        if not self.latest_results:
            messagebox.showerror("Error", "No results to export")
            return

        combined = self.latest_results[0]  # dict with Pre-/Post- keys

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path:
            return

        # build metric table
        metrics = {}
        for key, val in combined.items():
            if key.startswith("Pre-"):
                m = key.replace("Pre-", "")
                metrics.setdefault(m, {})["PRE"] = val
            elif key.startswith("Post-"):
                m = key.replace("Post-", "")
                metrics.setdefault(m, {})["POST"] = val
            else:
                metrics.setdefault(key, {})["PRE"] = val

        # write csv with header Metric, PRE, POST
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "PRE", "POST"])
            # same ordering as update_tree
            order = ["Speed (px/sec)", "Distance (px)", "Freezing (%)", "Movement Index", "Duration (s)", "Frames", "FPS"]
            for metric in order:
                if metric in metrics:
                    writer.writerow([metric, metrics[metric].get("PRE", ""), metrics[metric].get("POST", "")])
            # remaining metrics
            for metric, vals in metrics.items():
                if metric not in order:
                    writer.writerow([metric, vals.get("PRE", ""), vals.get("POST", "")])

        messagebox.showinfo("Exported", f"Results saved to {path}")

    # ---- plotting helpers ----
    def plot_metrics(self, results, title="Behavioral Metrics"):
        keys = ["Speed (px/sec)", "Distance (px)", "Freezing (%)", "Movement Index"]
        found = [results.get(k, 0) for k in keys]
        plt.figure(figsize=(6,4))
        plt.bar(keys, found)
        plt.ylabel("Value")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_motion_timecourse(self, motion_values, fps, title="Motion"):
        t = np.arange(len(motion_values)) / fps
        plt.figure(figsize=(8,3))
        plt.plot(t, motion_values)
        plt.xlabel("Time (s)")
        plt.ylabel("Motion magnitude")
        plt.title(f"Motion timecourse - {title}")
        plt.tight_layout()
        plt.show()

    def plot_compare(self, r1, r2, labels=("A","B")):
        keys = ["Speed (px/sec)", "Distance (px)", "Freezing (%)", "Movement Index"]
        a = [r1.get(k, 0) for k in keys]
        b = [r2.get(k, 0) for k in keys]
        x = np.arange(len(keys))
        width = 0.35
        plt.figure(figsize=(7,4))
        plt.bar(x - width/2, a, width, label=labels[0])
        plt.bar(x + width/2, b, width, label=labels[1])
        plt.xticks(x, keys, rotation=15)
        plt.ylabel("Value")
        plt.title("Pre vs Post behavioral metrics")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_last_motion(self):
        if not self.latest_results:
            messagebox.showerror("Error", "No motion data available")
            return
        # latest_results stores motion arrays at index 1
        motions = self.latest_results[1]
        fps = self.latest_results[2]
        if isinstance(motions, tuple):
            # plot first only
            mv = motions[0]
            ffps = fps[0]
            self.plot_motion_timecourse(mv, ffps, title="Last motion (first video)")
        else:
            mv = motions
            ffps = fps
            self.plot_motion_timecourse(mv, ffps, title="Last motion")

    def _set_busy(self, busy=True):
        if busy:
            self.root.config(cursor="watch")
        else:
            self.root.config(cursor="")

# -------------------------
# Run app
# -------------------------
def main():
    root = tk.Tk()
    app = ZFishApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
