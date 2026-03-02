# GestureMouse v3

Control your mouse cursor with hand gestures using your webcam — no extra hardware needed.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)

---

## Features

- 🖱️ **Move cursor** — move your wrist in front of the camera
- 👆 **Left click** — pinch Thumb + Index finger
- ✌️ **Right click** — pinch Thumb + Middle finger
- 🔄 **Scroll** — pinch Thumb + Pinky finger and move hand up/down
- 📷 **Always-on-top camera window** — native floating preview stays visible even when the browser is minimized
- 🌐 **Web dashboard** — real-time gesture indicators, FPS counter, and adjustable settings
- ⚡ **60 FPS target** — uses MediaPipe Tasks VIDEO mode for fast tracking

---

## Requirements

- Python 3.9+
- Webcam
- Windows (tested on Windows 11)

> **Note:** `tkinter` is included with the standard Python installer on Windows. If it's missing, reinstall Python and make sure "tcl/tk and IDLE" is checked.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/gesture-mouse.git
cd gesture-mouse

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The MediaPipe hand landmark model (`hand_landmarker.task`, ~7 MB) is **downloaded automatically** on first run.

---

## Usage

```bash
python app.py
```

- A floating camera preview window will appear (bottom-right, always-on-top)
- Your browser will open the dashboard at `http://localhost:8765`
- Move your hand in front of the camera to start controlling the mouse

### Stopping

Press `Ctrl+C` in the terminal, or close the camera window.

---

## Gesture Reference

| Gesture | Action |
|---|---|
| Move wrist | Move cursor |
| Thumb + Index pinch | Left click (hold = drag) |
| Thumb + Middle pinch | Right click |
| Thumb + Pinky pinch + move up/down | Scroll |

---

## Settings (via dashboard)

| Setting | Description | Default |
|---|---|---|
| Pinch Threshold | How close fingers must be to trigger a pinch | 0.05 |
| Smoothing | Cursor movement smoothing (higher = smoother but slower) | 0.10 |

---

## Architecture

```
Main thread      → tkinter always-on-top camera window
Thread-1         → hand_detection_thread (OpenCV + MediaPipe Tasks API)
Thread-2         → aiohttp HTTP + WebSocket server
Thread-3         → browser auto-open
```

Detection runs entirely in Python — the browser is only used as a settings dashboard and is **not required** for the mouse control to work.

---

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection |
| `opencv-python` | Camera capture & frame processing |
| `pyautogui` | Mouse control |
| `aiohttp` | HTTP + WebSocket server |
| `Pillow` | Frame rendering in tkinter window |
| `tkinter` | Always-on-top native camera window (stdlib) |

---

## License

MIT
