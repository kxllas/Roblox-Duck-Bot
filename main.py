"""
Roblox Duck Bot — screen-based helper for a duck-and-pipes mini-game.
Python 3.10+. Uses MSS + OpenCV for vision and PyAutoGUI for Space input.

Important:
- No vision-only bot can guarantee zero losses; game speed/physics and screen effects vary.
- Run Roblox in windowed/borderless mode. Set the capture rectangle around the playfield.
- This tool sends Space to the active window. Keep Roblox focused while running.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mss
import numpy as np
import pyautogui

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"

DEFAULT = {
    "duck_hsv_min": [0, 80, 80],
    "duck_hsv_max": [25, 255, 255],
    "pipe_hsv_min": [35, 50, 40],
    "pipe_hsv_max": [95, 255, 255],
    "threshold": 150,
    "delay_ms": 0,
    "left": 0, "top": 0, "width": 800, "height": 600,
    "fps": 60,
    "adaptive": True
}

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def load_config():
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        merged = DEFAULT.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT.copy()


def save_config(data):
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def hsv_mask(hsv, lo, hi):
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def largest_center(mask, min_area=20):
    # Returns center and bounding box for the largest plausible connected component.
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = min_area
    for c in contours:
        area = cv2.contourArea(c)
        if area > best_area:
            x, y, w, h = cv2.boundingRect(c)
            best_area = area
            best = (x + w // 2, y + h // 2, x, y, w, h)
    return best


class DuckBot:
    def __init__(self, get_config, status_callback):
        self.get_config = get_config
        self.status = status_callback
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = None
        self.prev_pipe_x = None
        self.prev_t = None
        self.speed = 0.0
        self.best_threshold = None
        self.last_jump = 0.0
        self.last_seen_duck = None
        self.last_seen_pipe = None
        self.last_score = 0

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.status("Продолжение")
        else:
            self.pause_event.set()
            self.status("Пауза (F8)")

    def calibrate_color(self, screen_x, screen_y, which):
        cfg = self.get_config()
        with mss.mss() as sct:
            shot = np.array(sct.grab({"left": screen_x, "top": screen_y, "width": 1, "height": 1}))
        bgr = shot[0, 0, :3].reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        lo, hi = np.maximum(hsv - [10, 70, 70], [0, 0, 0]), np.minimum(hsv + [10, 70, 70], [179, 255, 255])
        cfg[f"{which}_hsv_min"] = lo.tolist()
        cfg[f"{which}_hsv_max"] = hi.tolist()
        self.status(f"Цвет {which}: HSV {hsv.tolist()}")

    def loop(self):
        cfg = self.get_config()
        target_dt = 1.0 / max(10, min(120, int(cfg.get("fps", 60))))
        try:
            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    tick = time.perf_counter()
                    if self.pause_event.is_set():
                        time.sleep(0.05)
                        continue

                    cfg = self.get_config()
                    region = {
                        "left": int(cfg["left"]), "top": int(cfg["top"]),
                        "width": max(1, int(cfg["width"])), "height": max(1, int(cfg["height"]))
                    }
                    frame = np.asarray(sct.grab(region))
                    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

                    duck_mask = hsv_mask(hsv, cfg["duck_hsv_min"], cfg["duck_hsv_max"])
                    pipe_mask = hsv_mask(hsv, cfg["pipe_hsv_min"], cfg["pipe_hsv_max"])
                    duck = largest_center(duck_mask, 15)
                    pipe = largest_center(pipe_mask, 40)

                    now = time.perf_counter()
                    if duck and pipe:
                        duck_x = duck[0]
                        # Pipes are expected to approach from the right; ignore objects behind duck.
                        px = pipe[2] if pipe[2] > duck_x else None
                        if px is not None:
                            if self.prev_pipe_x is not None and self.prev_t is not None:
                                dt = max(0.001, now - self.prev_t)
                                measured = (self.prev_pipe_x - px) / dt
                                if 0 < measured < 3000:
                                    self.speed = 0.75 * self.speed + 0.25 * measured if self.speed else measured
                            self.prev_pipe_x, self.prev_t = px, now
                            distance = px - duck[4] // 2 - duck_x
                            threshold = float(cfg["threshold"])
                            # Adaptive correction is intentionally conservative.
                            if cfg.get("adaptive") and self.best_threshold is not None:
                                threshold = 0.8 * threshold + 0.2 * self.best_threshold
                            delay = max(0, int(cfg["delay_ms"])) / 1000.0
                            predicted = distance - max(0.0, self.speed) * delay
                            if predicted <= threshold and now - self.last_jump > 0.28:
                                pyautogui.press("space")
                                self.last_jump = now
                                self.status(f"Прыжок | дистанция {distance:.0f}px | скорость {self.speed:.0f}px/s")
                    else:
                        # Clear stale velocity if detections disappear for a while.
                        if self.prev_t and now - self.prev_t > 1.0:
                            self.speed = 0.0
                            self.prev_pipe_x = self.prev_t = None

                    elapsed = time.perf_counter() - tick
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)
        except pyautogui.FailSafeException:
            self.status("Аварийная остановка: мышь в левом верхнем углу")
        except Exception as exc:
            self.status(f"Ошибка: {exc}")
        finally:
            self.status("Бот остановлен")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Roblox Duck Bot")
        self.root.geometry("640x650")
        self.root.minsize(600, 600)
        self.cfg = load_config()
        self.bot = DuckBot(self.read_config, self.set_status)
        self.vars = {}
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(150, self.poll_hotkeys)

    def _build(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="Roblox Duck Bot", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(main, text="Настрой область игры, цвета объектов и порог прыжка.", foreground="#555").pack(anchor="w", pady=(0, 10))

        # Screen capture region.
        region = ttk.LabelFrame(main, text="Область захвата экрана (координаты в пикселях)", padding=8)
        region.pack(fill="x", pady=5)
        self.add_fields(region, [("left","X"),("top","Y"),("width","Ширина"),("height","Высота")], columns=4)

        # HSV controls.
        colors = ttk.LabelFrame(main, text="HSV-диапазоны (H: 0–179, S/V: 0–255)", padding=8)
        colors.pack(fill="x", pady=5)
        self.add_fields(colors, [
            ("duck_hsv_min","Утка min (H,S,V)"), ("duck_hsv_max","Утка max (H,S,V)"),
            ("pipe_hsv_min","Трубы min (H,S,V)"), ("pipe_hsv_max","Трубы max (H,S,V)")
        ], columns=2, vector=True)

        # Jump tuning.
        tuning = ttk.LabelFrame(main, text="Настройки прыжка", padding=8)
        tuning.pack(fill="x", pady=5)
        self.add_fields(tuning, [("threshold","Порог расстояния (px)"),("delay_ms","Задержка (мс)"),("fps","Целевой FPS")], columns=3)
        self.vars["adaptive"] = tk.BooleanVar(value=bool(self.cfg.get("adaptive", True)))
        ttk.Checkbutton(tuning, text="Адаптивная калибровка (экспериментальная)", variable=self.vars["adaptive"]).grid(row=1, column=0, columnspan=3, sticky="w", pady=5)

        # Calibration and controls.
        actions = ttk.LabelFrame(main, text="Калибровка", padding=8)
        actions.pack(fill="x", pady=5)
        ttk.Label(actions, text="Наведи курсор на объект и нажми соответствующую кнопку:").pack(anchor="w")
        row = ttk.Frame(actions); row.pack(fill="x", pady=5)
        ttk.Button(row, text="Запомнить цвет уточки", command=lambda: self.calibrate("duck")).pack(side="left", padx=(0,6))
        ttk.Button(row, text="Запомнить цвет трубы", command=lambda: self.calibrate("pipe")).pack(side="left")
        ttk.Label(actions, text="Калибровка берёт пиксель под курсором; область захвата должна включать объект.", foreground="#555").pack(anchor="w")

        buttons = ttk.Frame(main); buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Старт", command=self.start).pack(side="left", padx=(0,8))
        ttk.Button(buttons, text="Стоп", command=self.stop).pack(side="left", padx=(0,8))
        ttk.Button(buttons, text="Сохранить настройки", command=self.save).pack(side="left")
        self.status_var = tk.StringVar(value="Готов. F8 — пауза, F9 — стоп.")
        ttk.Label(main, textvariable=self.status_var, relief="sunken", padding=7).pack(fill="x", side="bottom")

    def add_fields(self, parent, fields, columns=2, vector=False):
        for i, (key, label) in enumerate(fields):
            r, c = divmod(i, columns)
            cell = ttk.Frame(parent); cell.grid(row=r, column=c, sticky="ew", padx=5, pady=4)
            ttk.Label(cell, text=label).pack(anchor="w")
            val = self.cfg.get(key, "")
            var = tk.StringVar(value=",".join(map(str,val)) if vector else str(val))
            self.vars[key] = var
            ttk.Entry(cell, textvariable=var, width=22).pack(fill="x")
        for c in range(columns):
            parent.columnconfigure(c, weight=1)

    def read_config(self):
        out = self.cfg.copy()
        for key, var in self.vars.items():
            if key == "adaptive":
                out[key] = var.get()
            elif key.endswith("_hsv_min") or key.endswith("_hsv_max"):
                try:
                    values = [int(x.strip()) for x in var.get().split(",")]
                    if len(values) != 3: raise ValueError()
                    out[key] = values
                except Exception:
                    out[key] = self.cfg[key]
            else:
                try: out[key] = int(var.get())
                except Exception: out[key] = self.cfg.get(key, 0)
        return out

    def save(self):
        self.cfg = self.read_config()
        save_config(self.cfg)
        self.set_status("Настройки сохранены в config.json")

    def start(self):
        self.save()
        self.bot.start()
        self.set_status("Бот запущен. Roblox должен быть активным окном.")

    def stop(self):
        self.bot.stop()
        self.set_status("Остановка...")

    def calibrate(self, which):
        x, y = pyautogui.position()
        self.bot.calibrate_color(x, y, which)
        self.cfg = self.read_config()
        # Refresh vector fields with calibrated values.
        for key in (f"{which}_hsv_min", f"{which}_hsv_max"):
            self.vars[key].set(",".join(map(str, self.cfg[key])))
        self.save()

    def set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def poll_hotkeys(self):
        try:
            import keyboard
            if keyboard.is_pressed("F8"):
                if not getattr(self, "_f8_down", False):
                    self.bot.toggle_pause()
                self._f8_down = True
            else:
                self._f8_down = False
            if keyboard.is_pressed("F9"):
                self.bot.stop()
        except Exception:
            pass
        self.root.after(120, self.poll_hotkeys)

    def close(self):
        self.bot.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()
