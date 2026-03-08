#!/usr/bin/env python3
"""
Hand Gesture Mouse Control - v1.0.0
- Hand detection runs entirely in Python (OpenCV + MediaPipe)
- Always-on-top floating camera window (tkinter) stays visible even when browser is minimized
- WebSocket server sends mouse commands to pyautogui
- Browser dashboard for settings/status (optional)

Usage: python app.py
"""

import argparse
import asyncio
import json
import threading
import webbrowser
import time
import queue
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode
)
import pyautogui
import numpy as np
from aiohttp import web

# ── PyAutoGUI config ──────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# ── Model path ────────────────────────────────────────────────────────────────
import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

# ── One Euro Filter (jitter removal) ─────────────────────────────────────────
class LowPassFilter:
    """Simple exponential low-pass filter."""
    def __init__(self, alpha=1.0):
        self.y = None
        self.alpha = alpha

    def __call__(self, value, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.y = value
        else:
            self.y = self.alpha * value + (1 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None


class OneEuroFilter:
    """One Euro Filter — speed-adaptive jitter reduction.
    Low min_cutoff = smoother when still.  High beta = more responsive to fast moves."""
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if self.last_time is not None and t is not None:
            self.freq = 1.0 / max(t - self.last_time, 1e-9)
        self.last_time = t

        prev = self.x_filter.y
        dx = 0.0 if prev is None else (x - prev) * self.freq
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter(x, self._alpha(cutoff))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


# ── Shared state (thread-safe via simple primitives) ──────────────────────────
class AppState:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()

        # Settings (can be updated from browser)
        self.pinch_threshold = 0.05
        self.smoothing = 0.3          # controls One Euro Filter min_cutoff
        self.sensitivity = 1.0        # 0.5 = big hand moves, 2.0 = tiny hand moves

        # Derived hand range (recomputed when sensitivity changes)
        self._base_center = 0.5
        self._base_half = 0.2         # default range is 0.3–0.7
        self._update_hand_range()

        # Target position from detection thread (raw, before filtering)
        self.target_x = screen_width / 2.0
        self.target_y = screen_height / 2.0

        # Actual cursor position (updated by cursor thread at 60fps)
        self.curr_x = screen_width / 2.0
        self.curr_y = screen_height / 2.0

        self.last_left = False
        self.last_right = False
        self.last_scroll = False
        self.smooth_scroll_y = None

        # Scroll constants
        self.SCROLL_SMOOTHING = 0.3
        self.SCROLL_SENSITIVITY = 6000
        self.SCROLL_DEAD_ZONE = 5

        # Status for dashboard
        self.hand_detected = False
        self.gesture_left = False
        self.gesture_right = False
        self.gesture_scroll = False
        self.fps = 0
        self.gesture_count = 0

        # Frame queue for tkinter window (latest frame only)
        self.frame_queue = queue.Queue(maxsize=2)

        # WebSocket clients for broadcasting status
        self.ws_clients = set()

    def _update_hand_range(self):
        """Recompute hand_range_min/max from sensitivity."""
        half = self._base_half / max(self.sensitivity, 0.1)
        self.hand_range_min = max(0.0, self._base_center - half)
        self.hand_range_max = min(1.0, self._base_center + half)

state = AppState()

# ── MediaPipe hand connections (for drawing) ──────────────────────────────────
# New Tasks API: connections are frozensets of (start_idx, end_idx) tuples
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]

# ── Colour palette ────────────────────────────────────────────────────────────
ACCENT       = (231, 92, 108)   # BGR: #6c5ce7 → purple
ACCENT_LIGHT = (254, 155, 162)  # BGR: #a29bfe
LEFT_COLOR   = (110, 203, 253)  # BGR: #fdcb6e → yellow
RIGHT_COLOR  = (255, 185, 116)  # BGR: #74b9ff → blue
SCROLL_COLOR = (254, 155, 162)  # BGR: #a29bfe → purple
GREEN        = (0, 255, 0)
RED          = (0, 0, 255)
WHITE        = (255, 255, 255)
DARK_BG      = (19, 15, 15)     # BGR: #0f0f13


def dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def remap(val, in_min, in_max):
    val = max(in_min, min(val, in_max))
    return (val - in_min) / (in_max - in_min)


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=0.6):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_overlay(frame, hand_detected, is_left, is_right, is_scroll, fps):
    """Draw HUD overlay on the camera frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "GestureMouse", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (162, 155, 254), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps}"
    cv2.putText(frame, fps_text, (w - 80, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136, 136, 168), 1, cv2.LINE_AA)

    # Hand status badge
    if hand_detected:
        badge_color = (201, 206, 0)   # BGR for teal
        badge_text = "HAND DETECTED"
    else:
        badge_color = (100, 100, 120)
        badge_text = "NO HAND"

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (8, h - 30), (8 + len(badge_text) * 9 + 8, h - 8),
                  badge_color, -1)
    cv2.addWeighted(overlay2, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, badge_text, (12, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 230, 220) if hand_detected else (100, 100, 120),
                1, cv2.LINE_AA)

    # Gesture indicators (bottom right)
    gestures = [
        ("L-CLICK", is_left,   LEFT_COLOR),
        ("R-CLICK", is_right,  RIGHT_COLOR),
        ("SCROLL",  is_scroll, SCROLL_COLOR),
    ]
    bx = w - 90
    by = h - 30
    for i, (label, active, color) in enumerate(gestures):
        x = bx + i * 0  # stack vertically
        y = by - i * 22
        dot_color = color if active else (60, 60, 80)
        cv2.circle(frame, (w - 85, y + 5), 5, dot_color, -1)
        cv2.putText(frame, label, (w - 76, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color if active else (80, 80, 100),
                    1, cv2.LINE_AA)


def draw_hand_skeleton(frame, landmarks):
    """Draw hand skeleton using manual connections (Tasks API compatible)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], ACCENT, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, ACCENT_LIGHT, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, ACCENT, 1, cv2.LINE_AA)


# ── Hand detection thread ─────────────────────────────────────────────────────
def hand_detection_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize capture buffer lag

    frame_count = 0
    fail_count = 0
    fps_timer = time.time()
    fps = 0

    opts = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,   # VIDEO mode uses tracking → much faster
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(opts) as detector:
        while state.running:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 60:
                    print("ERROR: Camera disconnected or unavailable!")
                    state.running = False
                    break
                continue
            fail_count = 0

            frame_count += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = int(frame_count / elapsed)
                frame_count = 0
                fps_timer = now
                with state.lock:
                    state.fps = fps

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # VIDEO mode requires a monotonically increasing timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, timestamp_ms)

            hand_detected = False
            is_left = is_right = is_scroll = False

            if result.hand_landmarks:
                hand_detected = True
                landmarks = result.hand_landmarks[0]  # list of NormalizedLandmark

                wrist     = landmarks[0]
                thumb_tip = landmarks[4]
                idx_tip   = landmarks[8]
                mid_tip   = landmarks[12]
                pnk_tip   = landmarks[20]

                with state.lock:
                    thresh    = state.pinch_threshold
                    smoothing = state.smoothing
                    rmin      = state.hand_range_min
                    rmax      = state.hand_range_max

                d_idx = dist(thumb_tip, idx_tip)
                d_mid = dist(thumb_tip, mid_tip)
                d_pnk = dist(thumb_tip, pnk_tip)

                is_left   = d_idx < thresh
                is_right  = d_mid < thresh
                is_scroll = d_pnk < thresh

                # Draw skeleton
                draw_hand_skeleton(frame, landmarks)

                # Draw pinch lines
                h_px, w_px = frame.shape[:2]
                if is_left:
                    cv2.line(frame,
                             (int(thumb_tip.x * w_px), int(thumb_tip.y * h_px)),
                             (int(idx_tip.x * w_px),   int(idx_tip.y * h_px)),
                             LEFT_COLOR, 3, cv2.LINE_AA)
                if is_right:
                    cv2.line(frame,
                             (int(thumb_tip.x * w_px), int(thumb_tip.y * h_px)),
                             (int(mid_tip.x * w_px),   int(mid_tip.y * h_px)),
                             RIGHT_COLOR, 3, cv2.LINE_AA)
                if is_scroll:
                    cv2.line(frame,
                             (int(thumb_tip.x * w_px), int(thumb_tip.y * h_px)),
                             (int(pnk_tip.x * w_px),   int(pnk_tip.y * h_px)),
                             SCROLL_COLOR, 3, cv2.LINE_AA)

                # ── Set target for cursor thread ─────────────────────────────
                x = remap(wrist.x, rmin, rmax)
                y = remap(wrist.y, rmin, rmax)

                with state.lock:
                    state.target_x = x * screen_width
                    state.target_y = y * screen_height
                    last_scroll = state.last_scroll

                # Scroll is still handled here (doesn't need 60fps)
                if is_scroll:
                    with state.lock:
                        smooth_scroll_y = state.smooth_scroll_y
                    if smooth_scroll_y is None or not last_scroll:
                        smooth_scroll_y = y
                    else:
                        prev = smooth_scroll_y
                        smooth_scroll_y += (y - smooth_scroll_y) * state.SCROLL_SMOOTHING
                        delta = (smooth_scroll_y - prev) * -state.SCROLL_SENSITIVITY
                        if abs(delta) > state.SCROLL_DEAD_ZONE:
                            pyautogui.scroll(int(delta))
                    with state.lock:
                        state.smooth_scroll_y = smooth_scroll_y

                with state.lock:
                    if is_left and not state.last_left:    state.gesture_count += 1
                    if is_right and not state.last_right:  state.gesture_count += 1
                    if is_scroll and not state.last_scroll: state.gesture_count += 1
                    state.last_left   = is_left
                    state.last_right  = is_right
                    state.last_scroll = is_scroll

            else:
                # Release mouse button if it was held (hand left frame while clicking)
                with state.lock:
                    was_left = state.last_left
                    state.last_left = state.last_right = state.last_scroll = False
                    state.smooth_scroll_y = None
                if was_left:
                    pyautogui.mouseUp()

            with state.lock:
                state.hand_detected  = hand_detected
                state.gesture_left   = is_left
                state.gesture_right  = is_right
                state.gesture_scroll = is_scroll

            # Draw HUD
            draw_overlay(frame, hand_detected, is_left, is_right, is_scroll, fps)

            # Push frame to queue (drop old frame if full)
            if state.frame_queue.full():
                try:
                    state.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                state.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    cap.release()


# ── 60fps cursor update thread ────────────────────────────────────────────────
def cursor_update_thread():
    """Runs at ~60 Hz, applies One Euro Filter, and moves the cursor.
    Decoupled from camera frame rate for fluid motion."""
    filter_x = OneEuroFilter(freq=60.0, min_cutoff=1.5, beta=0.01)
    filter_y = OneEuroFilter(freq=60.0, min_cutoff=1.5, beta=0.01)
    interval = 1.0 / 60.0  # ~16.67 ms

    # Track previous click state LOCALLY to detect edges correctly
    prev_left = False
    prev_right = False

    while state.running:
        t = time.time()

        with state.lock:
            hand = state.hand_detected
            tx = state.target_x
            ty = state.target_y
            is_left = state.gesture_left
            is_right = state.gesture_right
            is_scroll = state.gesture_scroll
            smoothing = state.smoothing

        if hand and not is_scroll:
            # Update filter min_cutoff from smoothing slider
            # smoothing 0.05→0.5 maps to min_cutoff 3.0→0.3 (lower = smoother)
            mc = max(0.3, 3.0 - smoothing * 6.0)
            filter_x.min_cutoff = mc
            filter_y.min_cutoff = mc

            fx = filter_x(tx, t)
            fy = filter_y(ty, t)
            fx = max(0, min(fx, screen_width - 1))
            fy = max(0, min(fy, screen_height - 1))

            with state.lock:
                state.curr_x = fx
                state.curr_y = fy

            pyautogui.moveTo(int(fx), int(fy))

            # Click handling (edge detection using local prev state)
            if is_left and not prev_left:
                pyautogui.mouseDown()
            elif not is_left and prev_left:
                pyautogui.mouseUp()
            if is_right and not prev_right:
                pyautogui.click(button='right')

            prev_left = is_left
            prev_right = is_right
        else:
            # Release mouse if hand disappeared while clicking
            if prev_left:
                pyautogui.mouseUp()
            prev_left = False
            prev_right = False
            filter_x.reset()
            filter_y.reset()

        elapsed = time.time() - t
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── Always-on-top tkinter camera window ──────────────────────────────────────
def run_camera_window():
    """Runs the always-on-top floating camera preview using tkinter.
    MUST be called from the main thread."""
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("GestureMouse — Camera")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.95)
    root.resizable(True, True)
    root.configure(bg="#0f0f13")

    # Window geometry: small preview, bottom-right
    win_w, win_h = 320, 260
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{win_w}x{win_h}+{sw - win_w - 20}+{sh - win_h - 60}")

    # Title bar
    title_bar = tk.Frame(root, bg="#1a1a24", height=28)
    title_bar.pack(fill=tk.X, side=tk.TOP)
    title_bar.pack_propagate(False)

    tk.Label(title_bar, text="● LIVE  GestureMouse",
             bg="#1a1a24", fg="#a29bfe",
             font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=8)

    # Minimize/restore button
    minimized = [False]
    def toggle_size():
        if minimized[0]:
            root.geometry(f"{win_w}x{win_h}")
            minimized[0] = False
        else:
            root.geometry(f"{win_w}x30")
            minimized[0] = True

    tk.Button(title_bar, text="—", bg="#2a2a3d", fg="white",
              relief=tk.FLAT, font=("Segoe UI", 9),
              command=toggle_size, cursor="hand2",
              padx=6).pack(side=tk.RIGHT, padx=4, pady=2)

    # Canvas for video
    canvas = tk.Canvas(root, bg="#0f0f13", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Drag support
    drag = {"x": 0, "y": 0}
    def on_drag_start(e):
        drag["x"] = e.x_root - root.winfo_x()
        drag["y"] = e.y_root - root.winfo_y()
    def on_drag_motion(e):
        root.geometry(f"+{e.x_root - drag['x']}+{e.y_root - drag['y']}")
    title_bar.bind("<ButtonPress-1>", on_drag_start)
    title_bar.bind("<B1-Motion>", on_drag_motion)

    # Clean exit handlers (Escape key or window close button)
    def on_close():
        state.running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda e: on_close())

    photo_ref = [None]

    def update_frame():
        if not state.running:
            root.destroy()
            return
        try:
            frame = state.frame_queue.get_nowait()
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw > 1 and ch > 1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((cw, ch), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                photo_ref[0] = photo  # prevent GC
        except queue.Empty:
            pass
        root.after(16, update_frame)  # ~60 fps

    root.after(100, update_frame)
    root.mainloop()


# ── Browser dashboard HTML ────────────────────────────────────────────────────
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GestureMouse — Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #08080c; --bg2: #111118; --bg3: #16161f;
      --card: rgba(22,22,34,0.6); --card-border: rgba(108,92,231,0.15);
      --glass: rgba(22,22,34,0.45);
      --border: #1e1e30; --accent: #7c6cf0; --accent-l: #a89eff;
      --accent-glow: rgba(124,108,240,0.35);
      --success: #34d9a3; --warn: #f9c74f; --blue: #60a5fa;
      --danger: #f87171; --text: #eeeef5; --muted: #7e7e9a;
      --dim: #4a4a60; --r: 12px;
    }

    body {
      background: var(--bg); color: var(--text);
      font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
      height: 100vh; overflow: hidden; display: flex; flex-direction: column;
    }

    /* Subtle animated background glow */
    body::before {
      content: ''; position: fixed; top: -40%; left: -10%; width: 60%; height: 80%;
      background: radial-gradient(ellipse, rgba(124,108,240,0.06) 0%, transparent 70%);
      pointer-events: none; z-index: 0; animation: bgShift 20s ease-in-out infinite alternate;
    }
    body::after {
      content: ''; position: fixed; bottom: -30%; right: -10%; width: 50%; height: 70%;
      background: radial-gradient(ellipse, rgba(52,217,163,0.04) 0%, transparent 70%);
      pointer-events: none; z-index: 0; animation: bgShift 15s ease-in-out infinite alternate-reverse;
    }
    @keyframes bgShift { from { transform: translate(0,0); } to { transform: translate(40px, 20px); } }
    @keyframes pulse { 0%,100% { transform: scale(1); opacity:1; } 50% { transform: scale(1.5); opacity:0.5; } }
    @keyframes fadeIn { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: none; } }
    @keyframes heroFloat {
      0%,100% { transform: translateY(0); }
      50% { transform: translateY(-8px); }
    }

    /* Header */
    .hdr {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 28px; background: var(--bg2); position: relative; z-index: 10;
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand svg { width: 26px; height: 26px; color: var(--accent); filter: drop-shadow(0 0 6px var(--accent-glow)); }
    .brand h1 { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
    .brand span { background: linear-gradient(135deg, var(--accent), var(--accent-l));
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .pill {
      display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 500;
      padding: 6px 16px; border-radius: 20px;
      background: var(--glass); border: 1px solid var(--card-border);
      backdrop-filter: blur(10px);
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim);
           transition: all 0.3s ease; }
    .dot.ok  { background: var(--success); box-shadow: 0 0 10px rgba(52,217,163,0.5);
               animation: pulse 2s infinite; }
    .dot.err { background: var(--danger); box-shadow: 0 0 10px rgba(248,113,113,0.5); }

    /* Layout */
    .main { flex: 1; display: flex; overflow: hidden; position: relative; z-index: 1; }

    .sidebar {
      width: 280px; background: var(--bg2); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow-y: auto; flex-shrink: 0;
      backdrop-filter: blur(12px);
    }
    .sec { padding: 20px 22px; border-bottom: 1px solid var(--border); }
    .sec:last-child { border-bottom: none; }
    .sec-title {
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 1.5px; color: var(--dim); margin-bottom: 14px;
    }

    /* Gesture cards */
    .g-list { display: flex; flex-direction: column; gap: 8px; }
    .g-item {
      display: flex; align-items: center; gap: 10px; padding: 10px 14px;
      border-radius: 10px; background: var(--card); border: 1px solid var(--card-border);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      backdrop-filter: blur(8px); cursor: default;
    }
    .g-item:hover { background: rgba(30,30,50,0.7); transform: translateX(3px); }
    .g-item.on {
      border-color: var(--accent); background: rgba(124,108,240,0.1);
      box-shadow: 0 0 20px rgba(124,108,240,0.15), inset 0 0 20px rgba(124,108,240,0.05);
      transform: translateX(3px);
    }
    .g-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
             transition: all 0.3s ease; }
    .g-dot.lc { background: var(--warn); }
    .g-dot.rc { background: var(--blue); }
    .g-dot.sc { background: var(--accent-l); }
    .g-item.on .g-dot { box-shadow: 0 0 12px currentColor; animation: pulse 1.5s infinite; }
    .g-item.on .g-dot.lc { color: var(--warn); }
    .g-item.on .g-dot.rc { color: var(--blue); }
    .g-item.on .g-dot.sc { color: var(--accent-l); }
    .g-name { font-size: 13px; font-weight: 500; }
    .g-key {
      margin-left: auto; font-size: 9px; font-weight: 600; color: var(--dim);
      padding: 3px 8px; border-radius: 5px; background: rgba(8,8,14,0.6);
      letter-spacing: 0.3px; text-transform: uppercase;
    }

    /* Sliders */
    .ctrl { margin-bottom: 16px; }
    .ctrl:last-child { margin-bottom: 0; }
    .ctrl-hdr { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .ctrl-lbl { font-size: 12px; color: var(--muted); font-weight: 400; }
    .ctrl-val { font-size: 12px; font-weight: 700; color: var(--accent-l);
                font-variant-numeric: tabular-nums; }

    input[type=range] {
      -webkit-appearance: none; width: 100%; height: 4px;
      border-radius: 4px; background: var(--bg); outline: none; cursor: pointer;
      transition: background 0.2s;
    }
    input[type=range]:hover { background: #1a1a2a; }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 18px; height: 18px;
      border-radius: 50%; background: linear-gradient(135deg, var(--accent), var(--accent-l));
      border: 2px solid var(--bg2); cursor: pointer;
      box-shadow: 0 0 10px var(--accent-glow);
      transition: transform 0.15s, box-shadow 0.15s;
    }
    input[type=range]::-webkit-slider-thumb:hover {
      transform: scale(1.2); box-shadow: 0 0 16px var(--accent-glow);
    }

    /* Content area */
    .content {
      flex: 1; display: flex; align-items: center; justify-content: center;
      background: var(--bg); padding: 32px; position: relative;
    }

    .center {
      display: flex; flex-direction: column; align-items: center;
      gap: 24px; text-align: center;
      animation: fadeIn 0.8s ease-out;
    }

    .hero {
      width: 88px; height: 88px; border-radius: 50%;
      background: linear-gradient(135deg, var(--accent), #9f6cf0, var(--accent-l));
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 50px var(--accent-glow), 0 0 100px rgba(124,108,240,0.15);
      animation: heroFloat 4s ease-in-out infinite;
      position: relative;
    }
    .hero::before {
      content: ''; position: absolute; inset: -4px; border-radius: 50%;
      background: linear-gradient(135deg, var(--accent), transparent, var(--accent-l));
      opacity: 0.3; filter: blur(8px);
    }
    .hero svg { width: 40px; height: 40px; color: #fff; position: relative; z-index: 1;
                filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3)); }

    .hero-title { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
    .hero-sub { font-size: 14px; color: var(--muted); max-width: 380px; line-height: 1.7;
                font-weight: 300; }

    .stats { display: flex; gap: 16px; margin-top: 8px; }
    .stat {
      background: var(--card); border: 1px solid var(--card-border);
      border-radius: var(--r); padding: 18px 28px; text-align: center;
      min-width: 120px; backdrop-filter: blur(8px);
      transition: all 0.3s ease;
    }
    .stat:hover { border-color: var(--accent); transform: translateY(-2px);
                  box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    .stat-v { font-size: 26px; font-weight: 800; color: var(--accent-l);
              font-variant-numeric: tabular-nums; }
    .stat-l { font-size: 10px; color: var(--dim); text-transform: uppercase;
              letter-spacing: 1px; margin-top: 4px; font-weight: 600; }

    .notice {
      background: var(--card); border: 1px solid var(--card-border);
      border-radius: var(--r); padding: 14px 20px; font-size: 13px;
      color: var(--muted); max-width: 420px; line-height: 1.6;
      backdrop-filter: blur(8px); font-weight: 300;
    }

    /* Performance sidebar stat */
    .perf-card {
      width: 100%; background: var(--card); border: 1px solid var(--card-border);
      border-radius: 10px; padding: 18px; text-align: center;
      backdrop-filter: blur(8px);
    }
    .perf-card .stat-v { font-size: 28px; font-weight: 800; }

    /* Footer */
    .ftr {
      padding: 8px 28px; background: var(--bg2); border-top: 1px solid var(--border);
      display: flex; justify-content: space-between; font-size: 10px;
      color: var(--dim); font-weight: 500; letter-spacing: 0.3px; z-index: 10;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--dim); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
  </style>
</head>
<body>
  <header class="hdr">
    <div class="brand">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 11V6a2 2 0 0 0-4 0v0M14 10V4a2 2 0 0 0-4 0v2M10 10.5V6a2 2 0 0 0-4 0v8"/>
        <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>
      </svg>
      <h1>Gesture<span>Mouse</span></h1>
    </div>
    <div class="pill">
      <div class="dot" id="dot"></div>
      <span id="statusTxt">Connecting...</span>
    </div>
  </header>

  <div class="main">
    <aside class="sidebar">
      <div class="sec">
        <div class="sec-title">Active Gestures</div>
        <div class="g-list">
          <div class="g-item" id="gL">
            <div class="g-dot lc"></div>
            <span class="g-name">Left Click</span>
            <span class="g-key">Thumb + Index</span>
          </div>
          <div class="g-item" id="gR">
            <div class="g-dot rc"></div>
            <span class="g-name">Right Click</span>
            <span class="g-key">Thumb + Middle</span>
          </div>
          <div class="g-item" id="gS">
            <div class="g-dot sc"></div>
            <span class="g-name">Scroll</span>
            <span class="g-key">Thumb + Pinky</span>
          </div>
        </div>
      </div>

      <div class="sec">
        <div class="sec-title">Settings</div>
        <div class="ctrl">
          <div class="ctrl-hdr">
            <span class="ctrl-lbl">Pinch Threshold</span>
            <span class="ctrl-val" id="threshVal">0.05</span>
          </div>
          <input type="range" id="pinchThreshold" min="0.02" max="0.15" step="0.01" value="0.05">
        </div>
        <div class="ctrl">
          <div class="ctrl-hdr">
            <span class="ctrl-lbl">Smoothing</span>
            <span class="ctrl-val" id="smoothVal">0.30</span>
          </div>
          <input type="range" id="smoothing" min="0.05" max="0.5" step="0.05" value="0.3">
        </div>
        <div class="ctrl">
          <div class="ctrl-hdr">
            <span class="ctrl-lbl">Sensitivity</span>
            <span class="ctrl-val" id="sensVal">1.00</span>
          </div>
          <input type="range" id="sensitivity" min="0.5" max="2.5" step="0.1" value="1.0">
        </div>
      </div>

      <div class="sec">
        <div class="sec-title">Performance</div>
        <div class="perf-card">
          <div class="stat-v" id="fpsVal">--</div>
          <div class="stat-l">Frames / sec</div>
        </div>
      </div>
    </aside>

    <div class="content">
      <div class="center">
        <div class="hero">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 11V6a2 2 0 0 0-4 0v0M14 10V4a2 2 0 0 0-4 0v2M10 10.5V6a2 2 0 0 0-4 0v8"/>
            <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>
          </svg>
        </div>
        <h2 class="hero-title">Hand Gesture Mouse Control</h2>
        <p class="hero-sub">
          Move your hand in front of the camera to control the cursor.
          Pinch fingers together to click, right-click, or scroll.
        </p>
        <div class="notice">
          📷 Camera detection runs natively in Python — it stays active even when
          this browser window is minimized. Look for the floating camera preview.
        </div>
        <div class="stats">
          <div class="stat">
            <div class="stat-v" id="gestureCount">0</div>
            <div class="stat-l">Gestures</div>
          </div>
          <div class="stat">
            <div class="stat-v" id="handStatus">—</div>
            <div class="stat-l">Hand</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <footer class="ftr">
    <span>MediaPipe · OpenCV · Python</span>
    <span>v3.0</span>
  </footer>

  <script>
    const dot = document.getElementById('dot');
    const statusTxt = document.getElementById('statusTxt');
    const gL = document.getElementById('gL');
    const gR = document.getElementById('gR');
    const gS = document.getElementById('gS');
    const fpsVal = document.getElementById('fpsVal');
    const gestureCount = document.getElementById('gestureCount');
    const handStatus = document.getElementById('handStatus');

    const pinchSlider = document.getElementById('pinchThreshold');
    const smoothSlider = document.getElementById('smoothing');
    const sensSlider = document.getElementById('sensitivity');
    const threshVal = document.getElementById('threshVal');
    const smoothVal = document.getElementById('smoothVal');
    const sensVal = document.getElementById('sensVal');

    pinchSlider.oninput = () => {
      threshVal.textContent = parseFloat(pinchSlider.value).toFixed(2);
      sendSettings();
    };
    smoothSlider.oninput = () => {
      smoothVal.textContent = parseFloat(smoothSlider.value).toFixed(2);
      sendSettings();
    };
    sensSlider.oninput = () => {
      sensVal.textContent = parseFloat(sensSlider.value).toFixed(2);
      sendSettings();
    };

    let ws, reconnects = 0;

    function connect() {
      ws = new WebSocket('ws://localhost:8765/ws');
      ws.onopen = () => {
        reconnects = 0;
        dot.className = 'dot ok';
        statusTxt.textContent = 'Connected';
        sendSettings();
      };
      ws.onclose = () => {
        dot.className = 'dot err';
        statusTxt.textContent = 'Reconnecting...';
        if (reconnects < 10) { reconnects++; setTimeout(connect, 1000 * reconnects); }
        else statusTxt.textContent = 'Connection failed';
      };
      ws.onerror = e => console.error(e);
      ws.onmessage = msg => {
        try {
          const d = JSON.parse(msg.data);
          if (d.type === 'status') {
            gL.classList.toggle('on', d.left);
            gR.classList.toggle('on', d.right);
            gS.classList.toggle('on', d.scroll);
            fpsVal.textContent = d.fps;
            gestureCount.textContent = d.gesture_count;
            handStatus.textContent = d.hand ? 'YES' : 'NO';
          }
        } catch(e) {}
      };
    }
    connect();

    function sendSettings() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'settings',
          pinch_threshold: parseFloat(pinchSlider.value),
          smoothing: parseFloat(smoothSlider.value),
          sensitivity: parseFloat(sensSlider.value)
        }));
      }
    }
  </script>
</body>
</html>
"""


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    with state.lock:
        state.ws_clients.add(ws)
    print("Dashboard client connected.")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get('type') == 'settings':
                        with state.lock:
                            if 'pinch_threshold' in data:
                                state.pinch_threshold = float(data['pinch_threshold'])
                            if 'smoothing' in data:
                                state.smoothing = float(data['smoothing'])
                            if 'sensitivity' in data:
                                state.sensitivity = float(data['sensitivity'])
                                state._update_hand_range()
                except Exception as e:
                    print(f"WS message error: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                print(f"WS error: {ws.exception()}")
    except Exception as e:
        print(f"WS connection error: {e}")
    finally:
        with state.lock:
            state.ws_clients.discard(ws)
        print("Dashboard client disconnected.")

    return ws


async def index(request):
    return web.Response(text=HTML_CONTENT, content_type='text/html')


# ── Status broadcast task ─────────────────────────────────────────────────────
async def broadcast_status():
    """Push gesture/status updates to all connected dashboard clients."""
    while True:
        await asyncio.sleep(0.1)  # 10 Hz
        with state.lock:
            clients = set(state.ws_clients)
            payload = json.dumps({
                "type": "status",
                "hand": state.hand_detected,
                "left": state.gesture_left,
                "right": state.gesture_right,
                "scroll": state.gesture_scroll,
                "fps": state.fps,
                "gesture_count": state.gesture_count,
            })
        for ws in clients:
            try:
                await ws.send_str(payload)
            except Exception:
                pass


# ── Asyncio server (runs in a background thread) ──────────────────────────────
def open_browser():
    time.sleep(2.0)
    webbrowser.open('http://localhost:8765')


async def server_main():
    app_web = web.Application()
    app_web.router.add_get('/', index)
    app_web.router.add_get('/ws', websocket_handler)

    runner = web.AppRunner(app_web)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8765)
    await site.start()

    asyncio.create_task(broadcast_status())

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def run_server_thread():
    """Run the aiohttp server in a dedicated thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server_main())
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        loop.close()


# ── Main entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GestureMouse — Hand Gesture Mouse Control")
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open the dashboard in the browser')
    args = parser.parse_args()

    print("=" * 52)
    print("  GestureMouse v3")
    print("=" * 52)
    print(f"  Dashboard : http://localhost:8765")
    print(f"  Screen    : {screen_width}x{screen_height}")
    print("  Camera window is always-on-top (native)")
    print("  Press Ctrl+C or Esc to stop")
    print("=" * 52)

    # Start hand detection in background thread
    det_thread = threading.Thread(target=hand_detection_thread, daemon=True)
    det_thread.start()

    # Start 60fps cursor update thread
    cur_thread = threading.Thread(target=cursor_update_thread, daemon=True)
    cur_thread.start()

    # Start aiohttp server in background thread
    srv_thread = threading.Thread(target=run_server_thread, daemon=True)
    srv_thread.start()

    # Open browser in background thread (unless --no-browser)
    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    # Run tkinter camera window on the MAIN thread (required by Tcl/Tk)
    try:
        run_camera_window()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        print("\nServer stopped.")
