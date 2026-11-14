"""
Roblox Duck Bot v2 — heuristic vision for the duck/pipe mini-game.
Designed around the supplied screenshot: a narrow vertical playfield, a yellow/orange
duck, and red/white pipe markers. It does NOT guarantee a win: the exact game physics
and jump behavior must be calibrated with live gameplay.
"""
from __future__ import annotations
import json, threading, time
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
    "left": 1410, "top": 290, "width": 230, "height": 380,
    "fps": 60, "jump_distance": 95, "jump_cooldown_ms": 420,
    "duck_hsv_min": [12, 105, 100], "duck_hsv_max": [42, 255, 255],
    "red1_min": [0, 100, 90], "red1_max": [12, 255, 255],
    "red2_min": [168, 100, 90], "red2_max": [179, 255, 255],
    "min_duck_area": 35, "debug": False
}
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

def load_cfg():
    try:
        d = DEFAULT.copy()
        d.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        return d
    except Exception:
        return DEFAULT.copy()

def largest_components(mask, min_area):
    mask = cv2.medianBlur(mask, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        found.append({"x": x, "y": y, "w": w, "h": h,
                      "cx": x + w / 2, "cy": y + h / 2, "area": area,
                      "contour": c})
    return sorted(found, key=lambda a: a["area"], reverse=True)

class VisionBot:
    def __init__(self, cfg_get, status):
        self.cfg_get, self.status = cfg_get, status
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = None
        self.last_jump = 0.0
        self.last_pipe_x = None
        self.pipe_velocity = 0.0
        self.last_frame_time = None
        self.last_duck = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def pause_toggle(self):
        self.pause_event.set() if not self.pause_event.is_set() else self.pause_event.clear()
        self.status("Пауза" if self.pause_event.is_set() else "Продолжение")

    def detect(self, frame, cfg):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Duck: orange/yellow body, restricted to the game viewport.
        duck_mask = cv2.inRange(hsv, np.array(cfg["duck_hsv_min"], np.uint8),
                                np.array(cfg["duck_hsv_max"], np.uint8))
        duck_candidates = largest_components(duck_mask, int(cfg["min_duck_area"]))
        duck = None
        # Prefer compact, roughly round components; outer yellow ring/body is usually compact.
        scored = []
        for d in duck_candidates:
            ratio = d["w"] / max(1, d["h"])
            if 0.55 <= ratio <= 1.8:
                score = d["area"] / (1 + abs(1 - ratio))
                scored.append((score, d))
        if scored:
            duck = max(scored, key=lambda x: x[0])[1]

        # Pipes: find red bands. Pair top/bottom bands with similar X coordinate.
        m1 = cv2.inRange(hsv, np.array(cfg["red1_min"], np.uint8), np.array(cfg["red1_max"], np.uint8))
        m2 = cv2.inRange(hsv, np.array(cfg["red2_min"], np.uint8), np.array(cfg["red2_max"], np.uint8))
        red = cv2.bitwise_or(m1, m2)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        bands = largest_components(red, 8)
        # Keep band-like red pieces, not tiny red decorations.
        bands = [b for b in bands if b["w"] >= 8 and b["h"] >= 2]
        pairs = []
        for i, a in enumerate(bands):
            for b in bands[i+1:]:
                if abs(a["cx"] - b["cx"]) <= max(12, frame.shape[1] * 0.08):
                    upper, lower = sorted([a, b], key=lambda q: q["cy"])
                    gap_top = upper["cy"] + upper["h"] / 2
                    gap_bottom = lower["cy"] - lower["h"] / 2
                    gap = gap_bottom - gap_top
                    if gap > frame.shape[0] * 0.10:
                        pairs.append({
                            "x": (upper["cx"] + lower["cx"]) / 2,
                            "gap_top": gap_top, "gap_bottom": gap_bottom,
                            "gap_center": (gap_top + gap_bottom) / 2,
                            "gap": gap
                        })
        # Nearest obstacle ahead of the duck (or right-most pair if duck is not found).
        if pairs:
            dx = duck["cx"] if duck else frame.shape[1] * 0.35
            ahead = [p for p in pairs if p["x"] >= dx - 5]
            pipe = min(ahead, key=lambda p: p["x"]) if ahead else None
        else:
            pipe = None
        return duck, pipe

    def run(self):
        try:
            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    start = time.perf_counter()
                    if self.pause_event.is_set():
                        time.sleep(.04)
                        continue
                    c = self.cfg_get()
                    region = {"left": int(c["left"]), "top": int(c["top"]),
                              "width": max(1, int(c["width"])), "height": max(1, int(c["height"]))}
                    frame = np.asarray(sct.grab(region))
                    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    duck, pipe = self.detect(bgr, c)
                    now = time.perf_counter()

                    if pipe:
                        if self.last_pipe_x is not None and self.last_frame_time:
                            dt = max(.001, now - self.last_frame_time)
                            v = (self.last_pipe_x - pipe["x"]) / dt
                            if 0 < v < 1500:
                                self.pipe_velocity = v if not self.pipe_velocity else .7*self.pipe_velocity + .3*v
                        self.last_pipe_x = pipe["x"]
                        self.last_frame_time = now

                    if duck and pipe:
                        # In this game screenshot, the duck must pass through the vertical gap.
                        # Space is assumed to move it upward; jump only if it is below gap center
                        # and the pipe is approaching the duck.
                        duck_y = duck["cy"]
                        gap_center = pipe["gap_center"]
                        dist = pipe["x"] - duck["cx"]
                        eta = dist / max(1.0, self.pipe_velocity)
                        close = dist <= float(c["jump_distance"])
                        predicted_y = duck_y  # Conservative: no unverified physics extrapolation.
                        below_center = predicted_y > gap_center + 4
                        cooldown = int(c["jump_cooldown_ms"]) / 1000.0
                        if close and below_center and now - self.last_jump > cooldown:
                            pyautogui.press("space")
                            self.last_jump = now
                            self.status(f"Прыжок | до трубы {dist:.0f}px | смещение от центра прохода {duck_y-gap_center:.0f}px")
                    elif not duck:
                        self.status("Уточка не найдена: проверь HSV/область захвата")
                    elif not pipe:
                        self.status("Трубы не найдены: проверь область захвата/цвет препятствий")

                    dt = time.perf_counter() - start
                    target = 1.0 / max(10, min(120, int(c["fps"])))
                    if dt < target:
                        time.sleep(target - dt)
        except pyautogui.FailSafeException:
            self.status("Аварийная остановка: мышь в левом верхнем углу")
        except Exception as e:
            self.status("Ошибка: " + str(e))
        finally:
            self.status("Бот остановлен")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Roblox Duck Bot v2")
        self.root.geometry("620x560")
        self.cfg = load_cfg()
        self.vars = {}
        self.bot = VisionBot(self.read_cfg, self.set_status)
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(150, self.hotkeys)

    def field(self, parent, key, label, row, col):
        f = ttk.Frame(parent); f.grid(row=row, column=col, sticky="ew", padx=5, pady=4)
        ttk.Label(f, text=label).pack(anchor="w")
        initial = ",".join(map(str, self.cfg[key])) if key.startswith("duck_hsv") else str(self.cfg[key])
        v = tk.StringVar(value=initial)
        self.vars[key] = v
        ttk.Entry(f, textvariable=v, width=18).pack(fill="x")

    def build(self):
        m = ttk.Frame(self.root, padding=12); m.pack(fill="both", expand=True)
        ttk.Label(m, text="Roblox Duck Bot v2", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(m, text="Распознавание уточки и прохода между трубами по каждому кадру.").pack(anchor="w", pady=(0,8))
        r = ttk.LabelFrame(m, text="Область игрового поля на всём экране", padding=8); r.pack(fill="x", pady=5)
        for i, (k,l) in enumerate([("left","X"),("top","Y"),("width","Ширина"),("height","Высота")]):
            self.field(r,k,l,0,i)
        t = ttk.LabelFrame(m, text="Параметры управления", padding=8); t.pack(fill="x", pady=5)
        for i,(k,l) in enumerate([("jump_distance","Дистанция до трубы (px)"),("jump_cooldown_ms","Интервал прыжков (мс)"),("fps","Целевой FPS")]):
            self.field(t,k,l,0,i)
        h = ttk.LabelFrame(m, text="Распознавание уточки (HSV)", padding=8); h.pack(fill="x", pady=5)
        for i,(k,l) in enumerate([("duck_hsv_min","Утка min H,S,V"),("duck_hsv_max","Утка max H,S,V")]):
            self.field(h,k,l,0,i)
        p = ttk.LabelFrame(m, text="Горячие клавиши", padding=8); p.pack(fill="x", pady=5)
        ttk.Label(p, text="F8 — пауза/продолжение · F9 — стоп · Esc — аварийная остановка PyAutoGUI").pack(anchor="w")
        b = ttk.Frame(m); b.pack(fill="x", pady=10)
        ttk.Button(b,text="Старт",command=self.start).pack(side="left",padx=(0,8))
        ttk.Button(b,text="Стоп",command=self.stop).pack(side="left",padx=(0,8))
        ttk.Button(b,text="Сохранить настройки",command=self.save).pack(side="left")
        self.status_var=tk.StringVar(value="Готов. Проверь координаты поля перед запуском.")
        ttk.Label(m,textvariable=self.status_var,relief="sunken",padding=7).pack(fill="x",side="bottom")

    def read_cfg(self):
        d=self.cfg.copy()
        for k,v in self.vars.items():
            try:
                if k.startswith("duck_hsv"):
                    d[k]=[int(x.strip()) for x in v.get().split(",")]
                    if len(d[k])!=3: raise ValueError()
                else: d[k]=int(v.get())
            except Exception: pass
        return d

    def save(self):
        self.cfg=self.read_cfg()
        CONFIG_PATH.write_text(json.dumps(self.cfg,ensure_ascii=False,indent=2),encoding="utf-8")
        self.status("Настройки сохранены")

    def start(self):
        self.save()
        self.bot.start()
        self.status("Запущен. Roblox должен быть видимым и активным окном.")

    def stop(self):
        self.bot.stop()
        self.status("Остановка...")

    def status(self,s):
        self.root.after(0,lambda:self.status_var.set(s))

    def hotkeys(self):
        try:
            import keyboard
            if keyboard.is_pressed("F8"):
                if not getattr(self,"f8",False): self.bot.pause_toggle()
                self.f8=True
            else: self.f8=False
            if keyboard.is_pressed("F9"): self.bot.stop()
        except Exception: pass
        self.root.after(150,self.hotkeys)

    def close(self):
        self.bot.stop()
        self.root.destroy()

if __name__=="__main__":
    root=tk.Tk()
    try: ttk.Style().theme_use("vista")
    except Exception: pass
    App(root)
    root.mainloop()
