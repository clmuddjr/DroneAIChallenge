import cv2
import time
import csv
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from djitellopy import Tello
from djitellopy import TelloException


# ---------------------------
# BASE DIR CONTROL
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR = Path("/Users/charlesleemuddjr/2021PythonProjects/Capstone I").expanduser().resolve()

MODEL_PATH = BASE_DIR / "runs" / "runs" / "detect" / "train2" / "weights" / "best.pt"

PHOTO_DIR = BASE_DIR / "tello_photos"
PHOTO_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = PHOTO_DIR / "photo_log.csv"


# ---------------------------
# BOX + DRAW HELPERS
# ---------------------------
def expand_box(x1, y1, x2, y2, w, h, scale=1.35):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = int(max(0, cx - bw / 2))
    ny1 = int(max(0, cy - bh / 2))
    nx2 = int(min(w - 1, cx + bw / 2))
    ny2 = int(min(h - 1, cy + bh / 2))
    return nx1, ny1, nx2, ny2


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, d["label"], (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# ---------------------------
# CSV + SAVE HELPERS
# ---------------------------
def ensure_csv_header(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "action",
                "raw_path", "annot_path",
                "best_crop_path", "all_crops_dir",
                "battery", "airborne",
                "num_detections",
                "detections"
            ])


def format_detections_for_csv(detections):
    parts = []
    for d in detections:
        parts.append(f"{d['name']}:{d['conf']:.3f}:{d['x1']}:{d['y1']}:{d['x2']}:{d['y2']}")
    return "|".join(parts)


def save_best_crop(raw_frame, detections, out_path: Path):
    if not detections:
        return ""
    best = max(detections, key=lambda d: d["conf"])
    crop = safe_crop(raw_frame, best["x1"], best["y1"], best["x2"], best["y2"])
    if crop is None:
        return ""
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def save_all_crops(raw_frame, detections, crops_dir: Path):
    crops_dir.mkdir(parents=True, exist_ok=True)
    if not detections:
        return str(crops_dir)

    det_sorted = sorted(detections, key=lambda d: d["conf"], reverse=True)
    for i, d in enumerate(det_sorted):
        crop = safe_crop(raw_frame, d["x1"], d["y1"], d["x2"], d["y2"])
        if crop is None:
            continue
        crop_path = crops_dir / f"{i:02d}_{d['name']}_{d['conf']:.2f}.jpg"
        cv2.imwrite(str(crop_path), crop)
    return str(crops_dir)


def save_crops_for_class(raw_frame, detections, crops_dir: Path, class_name: str):
    crops_dir.mkdir(parents=True, exist_ok=True)
    dets = [d for d in detections if d["name"].lower() == class_name.lower()]
    if not dets:
        return "", str(crops_dir)

    det_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)

    best = det_sorted[0]
    best_crop = safe_crop(raw_frame, best["x1"], best["y1"], best["x2"], best["y2"])
    best_path = ""
    if best_crop is not None:
        best_path = str(crops_dir / f"best_{class_name}_{best['conf']:.2f}.jpg")
        cv2.imwrite(best_path, best_crop)

    for i, d in enumerate(det_sorted):
        crop = safe_crop(raw_frame, d["x1"], d["y1"], d["x2"], d["y2"])
        if crop is None:
            continue
        crop_path = crops_dir / f"{i:02d}_{class_name}_{d['conf']:.2f}.jpg"
        cv2.imwrite(str(crop_path), crop)

    return best_path, str(crops_dir)


def log_action(ts, action, raw_path, annot_path, best_crop_path, crops_dir, battery, airborne, detections):
    det_str = format_detections_for_csv(detections)
    with open(LOG_PATH, "a", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            ts, action,
            raw_path, annot_path,
            best_crop_path, crops_dir,
            battery, airborne,
            len(detections),
            det_str
        ])


def ts_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def bundle_save(action: str, raw_frame, annotated_frame, detections, tello: Tello, airborne: bool):
    ts = ts_now()

    raw_path = str(PHOTO_DIR / f"{ts}_{action}_raw.jpg")
    annot_path = str(PHOTO_DIR / f"{ts}_{action}_annot.jpg")

    cv2.imwrite(raw_path, raw_frame)
    cv2.imwrite(annot_path, annotated_frame)

    best_crop_path = save_best_crop(raw_frame, detections, PHOTO_DIR / f"{ts}_{action}_best.jpg")
    all_crops_dir = save_all_crops(raw_frame, detections, PHOTO_DIR / f"{ts}_{action}_all_crops")

    try:
        battery = _call_any(tello, ["get_battery", "query_battery"])
    except Exception:
        battery = ""

    log_action(
        ts=ts,
        action=action,
        raw_path=raw_path,
        annot_path=annot_path,
        best_crop_path=best_crop_path,
        crops_dir=all_crops_dir,
        battery=battery,
        airborne=airborne,
        detections=detections
    )

    return f"SAVED: {action} | det={len(detections)}"


# ---------------------------------------------------------
# VERSION-SAFE PREFLIGHT + SAFE TAKEOFF
# ---------------------------------------------------------
def _call_any(tello: Tello, candidates):
    last_attr = None
    for name in candidates:
        last_attr = name
        fn = getattr(tello, name, None)
        if callable(fn):
            return fn()
    raise AttributeError(f"No available method among: {candidates} (last tried: {last_attr})")


def preflight_report(tello: Tello):
    def safe_call(candidates):
        try:
            return _call_any(tello, candidates)
        except Exception as e:
            return f"<err: {type(e).__name__}: {e}>"

    print("\n--- PREFLIGHT ---")
    print("Battery:", safe_call(["get_battery", "query_battery"]))
    print("Temp (°C):", safe_call(["get_temperature", "query_temperature"]))
    print("Height (cm):", safe_call(["get_height", "query_height"]))
    print("Barometer (cm):", safe_call(["get_barometer", "query_barometer"]))
    print("Pitch/Roll/Yaw:", safe_call(["get_attitude", "query_attitude"]))
    print("Speed X/Y/Z:",
          safe_call(["get_speed_x", "query_speed_x"]),
          safe_call(["get_speed_y", "query_speed_y"]),
          safe_call(["get_speed_z", "query_speed_z"]))
    print("Flight time (s):", safe_call(["get_flight_time", "query_flight_time"]))
    print("-----------------\n")


def safe_takeoff(tello: Tello, airborne_flag: bool):
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass

    time.sleep(0.2)

    try:
        tello.takeoff()
        return True, "TAKEOFF ✅"
    except Exception as e:
        return airborne_flag, f"TAKEOFF FAILED: {str(e).splitlines()[-1]}"


# ---------------------------
# INIT
# ---------------------------
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
print("Model names:", model.names)

ensure_csv_header(LOG_PATH)

tello = Tello()
tello.connect()
print("Battery:", _call_any(tello, ["get_battery", "query_battery"]), "%")

tello.streamon()
frame_read = tello.get_frame_read()

airborne = False

# RC (left/right, forward/back, up/down, yaw)
lr = fb = ud = yw = 0

MANUAL_SPEED = 35
MANUAL_BURST_SEC = 0.25
manual_until = 0.0

# Stabilize window after takeoff (prevents accidental motion immediately)
STABILIZE_SEC = 1.0
stabilize_until = 0.0

RC_SEND_HZ = 20
rc_last_send = 0.0

status_msg = ""
status_msg_until = 0.0

last_fps_t = time.time()
frames = 0
fps = 0.0


# ---------------------------
# AUTO_FIND SETTINGS
# ---------------------------
TARGETS = ["best", "charles", "katherine", "luna", "rifkind"]
target_idx = 0
target_name = TARGETS[target_idx]

AUTO_FIND = False
CONF_THRESH = 0.60
FOUND_STREAK_N = 6
found_streak = 0

# 360 scan via step-rotations
SCAN_STEP_DEG = 15          # short turns
SCAN_STEP_PAUSE_SEC = 0.35  # pause between steps (gives the video time to update)
scan_next_step_at = 0.0
scan_rotated_deg = 0        # how far we have rotated in this scan cycle
scan_dir = +1               # +1=clockwise, -1=counter-clockwise
scan_sweep_count = 0        # how many full sweeps completed


FOUND_COOLDOWN_SEC = 3.0
cooldown_until = 0.0


def set_manual(lr_v=0, fb_v=0, ud_v=0, yw_v=0, burst=MANUAL_BURST_SEC):
    global lr, fb, ud, yw, manual_until
    lr, fb, ud, yw = lr_v, fb_v, ud_v, yw_v
    manual_until = time.time() + burst


def set_status(msg: str, sec: float = 2.0):
    global status_msg, status_msg_until
    status_msg, status_msg_until = (msg, time.time() + sec)
    print(msg)


def cycle_target():
    global target_idx, target_name, found_streak
    target_idx = (target_idx + 1) % len(TARGETS)
    target_name = TARGETS[target_idx]
    found_streak = 0
    set_status(f"Target set to: {target_name}", 2.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# Multi-line HUD text (won't run off screen)
# NOTE: yaw uses U/O to avoid collisions with l=land and o=photo modes.
HUD_LINES = [
    "t=takeoff  l=land  SPACE=hover  q=quit(lands)  ESC=EMERGENCY",
    # "Manual: W/S fb  A/D lr  I/K ud  U/O yaw   +/- speed   [/] rotate 15deg",
    "Manual: W/S fb  A/D lr  I/K ud  C/V yaw   +/- speed   [/] rotate 15deg",
    "Photos: p=bundle  o=crops  1=best  2=all  3=class",
    "AUTO_FIND: f=toggle  b=cycle target  z/x=conf down/up",

]

print("\nControls:")
for line in HUD_LINES:
    print(" ", line)
print("Targets:", TARGETS)
print()


try:
    while True:
        frame = frame_read.frame
        if frame is None:
            continue

        raw_frame = frame.copy()
        h, w = raw_frame.shape[:2]

        # YOLO inference
        results = model.predict(raw_frame, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        r = results[0]

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale=1.35)

                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                name = model.names.get(cls_id, str(cls_id))
                label = f"{name} {conf:.2f}"

                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cls_id": cls_id, "conf": conf,
                    "name": name, "label": label
                })

        annotated = raw_frame.copy()
        draw_detections(annotated, detections)

        # FPS
        frames += 1
        now = time.time()
        if now - last_fps_t >= 1.0:
            fps = frames / (now - last_fps_t)
            frames = 0
            last_fps_t = now

        # HUD top (split into 2 shorter lines)
        cv2.putText(
            annotated,
            f"FPS:{fps:.1f}  Speed:{MANUAL_SPEED}  Air:{airborne}  FIND:{AUTO_FIND}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Target:{target_name}  Thresh:{CONF_THRESH:.2f}  Scan:{scan_rotated_deg}  Dir:{'CW' if scan_dir > 0 else 'CCW'}  Sweeps:{scan_sweep_count}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # HUD controls (multi-line)
        y = 90
        for line in HUD_LINES:
            cv2.putText(annotated, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            y += 22

        if now < status_msg_until and status_msg:
            cv2.putText(annotated, status_msg, (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Tello YOLO", annotated)
        key = cv2.waitKey(1) & 0xFF

        # ---------------------------
        # BASIC FLIGHT CONTROLS
        # ---------------------------
        if key == ord("t") and not airborne:
            preflight_report(tello)
            airborne, msg = safe_takeoff(tello, airborne)
            set_status(msg, 3.0)

            lr = fb = ud = yw = 0
            manual_until = 0.0
            stabilize_until = time.time() + STABILIZE_SEC

            AUTO_FIND = False
            found_streak = 0
            scan_rotated_deg = 0
            scan_next_step_at = time.time() + 0.6

            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass

        elif key == ord("l") and airborne:
            AUTO_FIND = False
            found_streak = 0
            lr = fb = ud = yw = 0
            manual_until = 0.0
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            time.sleep(0.2)
            try:
                tello.land()
            except Exception:
                pass
            airborne = False
            set_status("LAND ✅", 2.0)

        elif key == 32 and airborne:  # SPACE hover
            AUTO_FIND = False
            found_streak = 0
            lr = fb = ud = yw = 0
            manual_until = 0.0
            set_status("HOVER 🛑", 1.5)

        elif key == ord("q"):
            AUTO_FIND = False
            if airborne:
                lr = fb = ud = yw = 0
                manual_until = 0.0
                try:
                    tello.send_rc_control(0, 0, 0, 0)
                except Exception:
                    pass
                time.sleep(0.3)
                try:
                    tello.land()
                except Exception:
                    pass
            break

        elif key == 27:
            print("EMERGENCY STOP")
            try:
                tello.emergency()
            except Exception:
                pass
            break

        # ---------------------------
        # PHOTO / DATA CAPTURE
        # ---------------------------
        elif key == ord("p"):
            msg = bundle_save("bundle", raw_frame, annotated, detections, tello, airborne)
            set_status(msg, 2.0)

        elif key == ord("o"):
            ts = ts_now()
            best_path = save_best_crop(raw_frame, detections, PHOTO_DIR / f"{ts}_crops_best.jpg")
            crops_dir = save_all_crops(raw_frame, detections, PHOTO_DIR / f"{ts}_crops_all")
            try:
                battery = _call_any(tello, ["get_battery", "query_battery"])
            except Exception:
                battery = ""
            log_action(ts, "crops_only", "", "", best_path, crops_dir, battery, airborne, detections)
            set_status(f"SAVED: crops-only | det={len(detections)}", 2.0)

        elif key == ord("1"):
            ts = ts_now()
            best_path = save_best_crop(raw_frame, detections, PHOTO_DIR / f"{ts}_best_only.jpg")
            try:
                battery = _call_any(tello, ["get_battery", "query_battery"])
            except Exception:
                battery = ""
            log_action(ts, "best_only", "", "", best_path, "", battery, airborne, detections)
            set_status(f"SAVED: best-only | det={len(detections)}", 2.0)

        elif key == ord("2"):
            ts = ts_now()
            crops_dir = save_all_crops(raw_frame, detections, PHOTO_DIR / f"{ts}_all_only")
            try:
                battery = _call_any(tello, ["get_battery", "query_battery"])
            except Exception:
                battery = ""
            log_action(ts, "all_only", "", "", "", crops_dir, battery, airborne, detections)
            set_status(f"SAVED: all-crops-only | det={len(detections)}", 2.0)

        elif key == ord("3"):
            ts = ts_now()
            root = PHOTO_DIR / f"{ts}_class_crops"
            best_c, dir_c = save_crops_for_class(raw_frame, detections, root / "charles", "charles")
            best_k, dir_k = save_crops_for_class(raw_frame, detections, root / "katherine", "katherine")
            best_l, dir_l = save_crops_for_class(raw_frame, detections, root / "luna", "luna")
            best_r, dir_r = save_crops_for_class(raw_frame, detections, root / "rifkind", "rifkind")

            try:
                battery = _call_any(tello, ["get_battery", "query_battery"])
            except Exception:
                battery = ""
            log_action(
                ts, "class_crops",
                "", "",
                f"charles:{best_c} | katherine:{best_k} | luna:{best_l} | rifkind:{best_r}",
                f"charles_dir:{dir_c} | katherine_dir:{dir_k} | luna_dir:{dir_l} | rifkind_dir:{dir_r}",
                battery, airborne, detections
            )
            set_status("SAVED: class-crops", 2.0)

        # ---------------------------
        # AUTO_FIND CONTROLS
        # ---------------------------
        elif key == ord("b"):
            cycle_target()

        elif key == ord("z"):
            CONF_THRESH = clamp(CONF_THRESH - 0.05, 0.10, 0.95)
            found_streak = 0
            set_status(f"Thresh: {CONF_THRESH:.2f}", 1.2)

        elif key == ord("x"):
            CONF_THRESH = clamp(CONF_THRESH + 0.05, 0.10, 0.95)
            found_streak = 0
            set_status(f"Thresh: {CONF_THRESH:.2f}", 1.2)

        elif key == ord("f"):
            if airborne and time.time() >= stabilize_until:
                AUTO_FIND = not AUTO_FIND
                found_streak = 0
                lr = fb = ud = yw = 0
                manual_until = 0.0
                scan_rotated_deg = 0
                scan_next_step_at = time.time() + 0.2
                if AUTO_FIND:
                    set_status(f"AUTO_FIND ON → searching for: {target_name} (thresh={CONF_THRESH:.2f})", 3.0)
                else:
                    set_status("AUTO_FIND OFF", 2.0)
            else:
                set_status("AUTO_FIND requires airborne + stable", 2.0)

        # ---------------------------
        # MANUAL SPEED + ROTATE
        # ---------------------------
        elif key == ord("+") or key == ord("="):
            MANUAL_SPEED = min(80, MANUAL_SPEED + 5)
            set_status(f"Speed: {MANUAL_SPEED}", 1.2)

        elif key == ord("-") or key == ord("_"):
            MANUAL_SPEED = max(10, MANUAL_SPEED - 5)
            set_status(f"Speed: {MANUAL_SPEED}", 1.2)

        elif airborne and key == ord("[") and not AUTO_FIND and time.time() >= stabilize_until:
            try:
                tello.rotate_counter_clockwise(15)
                set_status("Rotate ⟲ 15°", 1.2)
            except Exception:
                set_status("Rotate fail", 2.0)

        elif airborne and key == ord("]") and not AUTO_FIND and time.time() >= stabilize_until:
            try:
                tello.rotate_clockwise(15)
                set_status("Rotate ⟳ 15°", 1.2)
            except Exception:
                set_status("Rotate fail", 2.0)

        # ---------------------------
        # MANUAL MOVEMENT (burst-based)
        # any manual input cancels AUTO_FIND
        # ---------------------------
        elif airborne and time.time() >= stabilize_until:
            # cancel auto-find on manual motion keys
            if key in (ord("w"), ord("s"), ord("a"), ord("d"), ord("i"), ord("k"), ord("u"), ord("o")):
                if AUTO_FIND:
                    AUTO_FIND = False
                    found_streak = 0
                    set_status("AUTO_FIND cancelled (manual input)", 1.5)

            if key == ord("w"):
                set_manual(fb_v=+MANUAL_SPEED)
            elif key == ord("s"):
                set_manual(fb_v=-MANUAL_SPEED)
            elif key == ord("a"):
                set_manual(lr_v=-MANUAL_SPEED)
            elif key == ord("d"):
                set_manual(lr_v=+MANUAL_SPEED)
            elif key == ord("i"):
                set_manual(ud_v=+MANUAL_SPEED)
            elif key == ord("k"):
                set_manual(ud_v=-MANUAL_SPEED)
            # elif key == ord("u"):   # yaw left
            #     set_manual(yw_v=-MANUAL_SPEED)
            # elif key == ord("o"):   # yaw right
            #     set_manual(yw_v=+MANUAL_SPEED)
            elif key == ord("c"):   # yaw left
                set_manual(yw_v=-MANUAL_SPEED)
            elif key == ord("v"):   # yaw right
                set_manual(yw_v=+MANUAL_SPEED)


        # ---------------------------
        # AUTO_FIND BEHAVIOR (360 scan + stop on found)
        # ---------------------------
        if airborne and AUTO_FIND and time.time() >= stabilize_until:
            # target selection
            if target_name.lower() == "best":
                good = [d for d in detections if d["conf"] >= CONF_THRESH]
            else:
                good = [d for d in detections
                        if d["name"].lower() == target_name.lower() and d["conf"] >= CONF_THRESH]

            if good:
                found_streak += 1

                # hold still while looking at target
                lr = fb = ud = yw = 0
                manual_until = 0.0

                if found_streak >= FOUND_STREAK_N and time.time() >= cooldown_until:
                    action = f"FOUND_{target_name}"
                    msg = bundle_save(action, raw_frame, annotated, detections, tello, airborne)
                    set_status(f"{msg} ✅", 3.0)

                    AUTO_FIND = False
                    found_streak = 0
                    lr = fb = ud = yw = 0
                    manual_until = 0.0
                    cooldown_until = time.time() + FOUND_COOLDOWN_SEC

            else:
                found_streak = 0

                # do step-rotations until we complete 360°, then reverse direction
                if time.time() >= scan_next_step_at and time.time() >= cooldown_until:
                    try:
                        # keep RC zero before rotation command
                        lr = fb = ud = yw = 0
                        manual_until = 0.0
                        try:
                            tello.send_rc_control(0, 0, 0, 0)
                        except Exception:
                            pass

                        if scan_dir > 0:
                            tello.rotate_clockwise(SCAN_STEP_DEG)
                        else:
                            tello.rotate_counter_clockwise(SCAN_STEP_DEG)

                        scan_rotated_deg += SCAN_STEP_DEG
                        if scan_rotated_deg >= 360:
                            scan_rotated_deg = 0
                            scan_dir *= -1
                            scan_sweep_count += 1
                            set_status(f"Scan completed 360° → reversing (sweeps={scan_sweep_count})", 1.2)

                    except Exception:
                        set_status("Scan rotate fail", 2.0)

                    scan_next_step_at = time.time() + SCAN_STEP_PAUSE_SEC

        # Stop burst when it expires
        if airborne and time.time() > manual_until and not AUTO_FIND:
            lr = fb = ud = yw = 0

        # Send RC continuously (only meaningful when not using rotate_* commands)
        now2 = time.time()
        if airborne and (now2 - rc_last_send) >= (1.0 / RC_SEND_HZ):
            try:
                tello.send_rc_control(lr, fb, ud, yw)
            except Exception:
                pass
            rc_last_send = now2

        if not airborne:
            lr = fb = ud = yw = 0

finally:
    print("Cleaning up resources...")
    try:
        if airborne:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            time.sleep(0.2)
            try:
                tello.land()
            except Exception:
                pass
    except Exception:
        pass

    try:
        tello.streamoff()
    except Exception:
        pass

    cv2.destroyAllWindows()

    try:
        tello.end()
    except Exception:
        pass
