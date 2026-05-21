import cv2
import pandas as pd
import numpy as np
import os
import sys
import shutil
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ============================================================
# vision-label-curator — bbox review & curation tool
# ============================================================
# Expected CSV columns: new_filename, class_name, cx, cy, w, h
# (boxes in YOLO-normalized cxcywh in [0, 1]).
# 'reviewed' and 'is_satellite' columns are created if missing.
#
# Controls:
#   D / Right Arrow     : next frame
#   A / Left Arrow      : previous frame
#   SPACE               : mark reviewed + next
#   T                   : trash frame (image + all CSV rows)
#   Click               : select box (image or right-side list panel)
#   X                   : delete selected box
#   R                   : remove every box in this frame
#   N                   : draw new box (mouse drag)
#   E                   : edit selected box (corners / edges / move)
#   C                   : change class of selected box
#   TAB                 : cycle box selection
#   F                   : jump to next unreviewed frame
#   U                   : toggle is_satellite flag for this frame
#   Z                   : undo last trash
#   S                   : save
#   Q                   : save and quit
#   H                   : help overlay
#
# After drawing a new box or pressing C:
#   0-9     : pick class 0-9 directly
#   1+0-9   : pick class 10-19
#   2+0     : pick class 20
#   ESC     : cancel
# ============================================================

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(WORKSPACE, "images")
TRASH_DIR = os.path.join(WORKSPACE, "_Trash")
CSV_PATH = os.path.join(WORKSPACE, "labels.csv")
PROGRESS_FILE = os.path.join(WORKSPACE, "_review_progress.json")
CONFIG_PATH = os.path.join(WORKSPACE, "workspace_config.json")


def _detect_display_size(min_w=1280, min_h=720, max_w=1920, max_h=1080, ratio=0.85):
    """Return (w, h) sized to the user's screen, clamped to sensible bounds.
    Falls back to (min_w, min_h) on systems without a display query."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        w = max(min_w, min(int(sw * ratio), max_w))
        h = max(min_h, min(int(sh * ratio), max_h))
        return w, h
    except Exception:
        return min_w, min_h


DISPLAY_W, DISPLAY_H = _detect_display_size()
AUTOSAVE_EVERY = 50

CLASS_COLORS = {
    0: (0, 255, 0), 1: (255, 200, 0), 2: (0, 100, 255),
    3: (0, 0, 255), 4: (200, 200, 0), 5: (255, 150, 150),
    6: (150, 0, 255), 7: (255, 0, 255), 8: (0, 255, 255),
    9: (255, 255, 100), 10: (100, 100, 255), 11: (200, 100, 0),
    12: (0, 200, 200), 13: (100, 0, 200), 14: (200, 200, 200),
    15: (0, 150, 100), 16: (150, 200, 255), 17: (0, 140, 255),
    18: (255, 100, 50), 19: (80, 120, 60), 20: (0, 200, 255),
}


def name_color(name):
    if not isinstance(name, str) or not name:
        return (200, 200, 200)
    h = hash(name) & 0xFFFFFF
    return (80 + (h & 0x7F), 80 + ((h >> 8) & 0x7F), 80 + ((h >> 16) & 0x7F))


def build_classes(df):
    """Load class id->name map from workspace_config.json, or build the top-21
    most-frequent class names from the CSV on first run and persist them."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            classes = {int(k): v for k, v in cfg.get("classes", {}).items()}
            if classes:
                print("[Config] Loaded " + str(len(classes)) + " classes: " +
                      str(list(classes.values())))
                return classes
        except Exception as e:
            print("Could not read workspace_config.json: " + str(e))
    if "class_name" in df.columns:
        vc = df["class_name"].dropna().astype(str)
        vc = vc[vc.str.strip() != ""]
        top = vc.value_counts().head(21)
        classes = {i: name for i, name in enumerate(top.index.tolist())}
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"classes": {str(k): v for k, v in classes.items()}},
                          f, ensure_ascii=False, indent=2)
            print("[Auto] Built top-" + str(len(classes)) +
                  " classes from CSV and wrote " + CONFIG_PATH)
        except Exception as e:
            print("Could not write workspace_config.json: " + str(e))
        print("[Classes] " + str(list(classes.values())))
        return classes
    return {}


class Reviewer:
    def __init__(self):
        os.makedirs(TRASH_DIR, exist_ok=True)
        if not os.path.exists(CSV_PATH):
            print("ERROR: " + CSV_PATH + " not found"); sys.exit(1)
        if not os.path.isdir(IMAGES_DIR):
            print("ERROR: " + IMAGES_DIR + " not found"); sys.exit(1)

        print("Loading CSV (large files may take a moment)...")
        self.df = pd.read_csv(CSV_PATH)
        print("  " + str(len(self.df)) + " rows")
        if "reviewed" not in self.df.columns:
            self.df["reviewed"] = False
        self.df["reviewed"] = self.df["reviewed"].fillna(False).astype(bool)
        if "is_satellite" not in self.df.columns:
            self.df["is_satellite"] = 0
        self.df["is_satellite"] = self.df["is_satellite"].fillna(0).astype(int)

        print("Scanning images folder...")
        files = sorted([f for f in os.listdir(IMAGES_DIR)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.frames = files
        print("  " + str(len(self.frames)) + " images")

        existing = set(self.frames)
        before = len(self.df)
        self.df = self.df[self.df["new_filename"].isin(existing)].reset_index(drop=True)
        if len(self.df) < before:
            print("  Removed " + str(before - len(self.df)) + " orphan rows")
            self.df.to_csv(CSV_PATH, index=False)

        self.CLASSES = build_classes(self.df)
        self.NAME_TO_ID = {v: k for k, v in self.CLASSES.items()}

        # State
        self.current_idx = 0
        self.selected_box = -1
        self.unsaved = False
        self.show_help_overlay = False
        self.undo_stack = []
        self.trash_since_save = 0
        self.session_trashed = 0
        self.session_passed = 0
        self.session_satellite = 0
        self.session_satellite_frames = set()
        self.box_list_rects = []
        self.box_list_panel_rect = None
        self.dirty = True

        # Edit / draw
        self.draw_mode = False
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.pending_box = None
        self.class_select_mode = False
        self.first_digit = None
        self.edit_mode = False
        self.edit_dragging = False
        self.edit_handle = None
        self.edit_drag_start = None

        # Display
        self.scale = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.orig_w = 0
        self.orig_h = 0

        self.load_progress()
        print("[" + str(self.current_idx + 1) + "/" + str(len(self.frames)) + "] starting")

    # ---------- progress ----------
    def load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                idx = d.get("current_idx", 0)
                if 0 <= idx < len(self.frames):
                    self.current_idx = idx
            except Exception:
                pass

    def save_progress(self):
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({"current_idx": self.current_idx}, f)

    def save_csv(self):
        self.df.to_csv(CSV_PATH, index=False)
        self.save_progress()
        self.unsaved = False
        self.trash_since_save = 0
        print("  Saved: " + str(len(self.df)) + " rows")

    # ---------- helpers ----------
    def get_boxes(self, frame_name):
        mask = self.df["new_filename"] == frame_name
        return self.df[mask], mask

    def to_orig(self, sx, sy):
        ox = (sx - self.x_offset) / self.scale if self.scale > 0 else 0
        oy = (sy - self.y_offset) / self.scale if self.scale > 0 else 0
        return int(ox), int(oy)

    def yolo_to_pixel(self, cx, cy, w, h, iw, ih):
        x1 = (cx - w / 2) * iw; y1 = (cy - h / 2) * ih
        x2 = (cx + w / 2) * iw; y2 = (cy + h / 2) * ih
        return x1, y1, x2, y2

    def pixel_to_yolo(self, x1, y1, x2, y2, iw, ih):
        if iw <= 0 or ih <= 0:
            return None
        cx = (x1 + x2) / 2.0 / iw
        cy = (y1 + y2) / 2.0 / ih
        w = abs(x2 - x1) / iw
        h = abs(y2 - y1) / ih
        return cx, cy, w, h

    def find_box_at(self, ox, oy, boxes_df):
        if len(boxes_df) == 0:
            return -1
        cands = []
        for i, (_, row) in enumerate(boxes_df.iterrows()):
            try:
                cx = float(row["cx"]); cy = float(row["cy"])
                w = float(row["w"]); h = float(row["h"])
            except (ValueError, TypeError):
                continue
            if any(pd.isna(v) for v in (cx, cy, w, h)):
                continue
            iw = self.orig_w if self.orig_w > 0 else 1
            ih = self.orig_h if self.orig_h > 0 else 1
            x1, y1, x2, y2 = self.yolo_to_pixel(cx, cy, w, h, iw, ih)
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                cands.append((i, (x2 - x1) * (y2 - y1)))
        if not cands:
            return -1
        cands.sort(key=lambda t: t[1])
        return cands[0][0]

    def get_edit_handle(self, ox, oy, x1, y1, x2, y2):
        margin = max(12, int(min(x2 - x1, y2 - y1) * 0.15))
        if abs(ox - x1) < margin and abs(oy - y1) < margin: return "tl"
        if abs(ox - x2) < margin and abs(oy - y1) < margin: return "tr"
        if abs(ox - x1) < margin and abs(oy - y2) < margin: return "bl"
        if abs(ox - x2) < margin and abs(oy - y2) < margin: return "br"
        if abs(oy - y1) < margin and x1 <= ox <= x2: return "t"
        if abs(oy - y2) < margin and x1 <= ox <= x2: return "b"
        if abs(ox - x1) < margin and y1 <= oy <= y2: return "l"
        if abs(ox - x2) < margin and y1 <= oy <= y2: return "r"
        if x1 <= ox <= x2 and y1 <= oy <= y2: return "move"
        return None

    def df_index_of_selected(self):
        frame = self.frames[self.current_idx]
        mask = self.df["new_filename"] == frame
        idxs = self.df[mask].index.tolist()
        if 0 <= self.selected_box < len(idxs):
            return idxs[self.selected_box]
        return None

    def apply_edit(self, handle, dx, dy):
        idx = self.df_index_of_selected()
        if idx is None:
            return
        try:
            cx = float(self.df.at[idx, "cx"]); cy = float(self.df.at[idx, "cy"])
            w = float(self.df.at[idx, "w"]); h = float(self.df.at[idx, "h"])
        except Exception:
            return
        if any(pd.isna(v) for v in (cx, cy, w, h)):
            return
        iw = self.orig_w if self.orig_w > 0 else 1
        ih = self.orig_h if self.orig_h > 0 else 1
        x1, y1, x2, y2 = self.yolo_to_pixel(cx, cy, w, h, iw, ih)
        if handle == "tl": x1 += dx; y1 += dy
        elif handle == "tr": x2 += dx; y1 += dy
        elif handle == "bl": x1 += dx; y2 += dy
        elif handle == "br": x2 += dx; y2 += dy
        elif handle == "t":  y1 += dy
        elif handle == "b":  y2 += dy
        elif handle == "l":  x1 += dx
        elif handle == "r":  x2 += dx
        elif handle == "move": x1 += dx; y1 += dy; x2 += dx; y2 += dy
        if x2 - x1 < 5 or y2 - y1 < 5:
            return
        x1 = max(0, min(iw, x1)); x2 = max(0, min(iw, x2))
        y1 = max(0, min(ih, y1)); y2 = max(0, min(ih, y2))
        if x2 <= x1 or y2 <= y1:
            return
        new = self.pixel_to_yolo(x1, y1, x2, y2, iw, ih)
        if new is None:
            return
        ncx, ncy, nw, nh = new
        self.df.at[idx, "cx"] = round(ncx, 6)
        self.df.at[idx, "cy"] = round(ncy, 6)
        self.df.at[idx, "w"] = round(nw, 6)
        self.df.at[idx, "h"] = round(nh, 6)
        self.unsaved = True

    # ---------- mouse ----------
    def mouse_cb(self, event, x, y, flags, param):
        # Box list panel click -> select
        if event == cv2.EVENT_LBUTTONDOWN and self.box_list_panel_rect is not None:
            px1, py1, px2, py2 = self.box_list_panel_rect
            if px1 <= x <= px2 and py1 <= y <= py2:
                for y_top, y_bot, bidx in self.box_list_rects:
                    if y_top <= y <= y_bot:
                        self.selected_box = bidx
                        self.dirty = True
                        return
                return

        ox, oy = self.to_orig(x, y)

        if self.class_select_mode:
            return

        if self.edit_mode and self.selected_box >= 0:
            idx = self.df_index_of_selected()
            if idx is None:
                return
            try:
                cx = float(self.df.at[idx, "cx"]); cy = float(self.df.at[idx, "cy"])
                w = float(self.df.at[idx, "w"]); h = float(self.df.at[idx, "h"])
            except Exception:
                return
            if any(pd.isna(v) for v in (cx, cy, w, h)):
                return
            iw = self.orig_w if self.orig_w > 0 else 1
            ih = self.orig_h if self.orig_h > 0 else 1
            x1, y1, x2, y2 = self.yolo_to_pixel(cx, cy, w, h, iw, ih)
            if event == cv2.EVENT_LBUTTONDOWN:
                handle = self.get_edit_handle(ox, oy, x1, y1, x2, y2)
                if handle:
                    self.edit_dragging = True
                    self.edit_handle = handle
                    self.edit_drag_start = (ox, oy)
            elif event == cv2.EVENT_MOUSEMOVE and self.edit_dragging:
                dx = ox - self.edit_drag_start[0]
                dy = oy - self.edit_drag_start[1]
                self.apply_edit(self.edit_handle, dx, dy)
                self.edit_drag_start = (ox, oy)
                self.dirty = True
            elif event == cv2.EVENT_LBUTTONUP and self.edit_dragging:
                self.edit_dragging = False
                self.edit_handle = None
                self.edit_drag_start = None
            return

        if self.draw_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.draw_start = (ox, oy)
                self.draw_end = (ox, oy)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.draw_end = (ox, oy)
                self.dirty = True
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                self.draw_end = (ox, oy)
                bx1 = min(self.draw_start[0], self.draw_end[0])
                by1 = min(self.draw_start[1], self.draw_end[1])
                bx2 = max(self.draw_start[0], self.draw_end[0])
                by2 = max(self.draw_start[1], self.draw_end[1])
                if (bx2 - bx1) >= 10 and (by2 - by1) >= 10:
                    self.pending_box = (bx1, by1, bx2, by2)
                    self.class_select_mode = True
                    self.first_digit = None
                else:
                    self.draw_start = None
                    self.draw_end = None
                self.dirty = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            frame = self.frames[self.current_idx]
            boxes_df, _ = self.get_boxes(frame)
            self.selected_box = self.find_box_at(ox, oy, boxes_df)
            self.dirty = True

    # ---------- pending box / class ----------
    def add_pending_box(self, class_id):
        if self.pending_box is None or class_id not in self.CLASSES:
            return
        bx1, by1, bx2, by2 = self.pending_box
        frame = self.frames[self.current_idx]
        iw, ih = self.orig_w, self.orig_h
        if iw <= 0 or ih <= 0:
            self.cancel_pending(); return
        new = self.pixel_to_yolo(bx1, by1, bx2, by2, iw, ih)
        if new is None:
            self.cancel_pending(); return
        cx, cy, w, h = new
        cname = self.CLASSES[class_id]
        new_row = {
            "new_filename": frame,
            "dataset_source": "manual",
            "original_filename": "",
            "class_name": cname,
            "cx": round(cx, 6), "cy": round(cy, 6),
            "w": round(w, 6), "h": round(h, 6),
            "img_width": iw, "img_height": ih,
            "status": "labeled",
            "notes": "manual_review",
            "reviewed": True,
        }
        for col in self.df.columns:
            if col not in new_row:
                new_row[col] = np.nan
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.unsaved = True
        self.pending_box = None
        self.class_select_mode = False
        self.first_digit = None
        self.draw_start = None
        self.draw_end = None
        print("  Box added: " + cname)

    def cancel_pending(self):
        self.pending_box = None
        self.class_select_mode = False
        self.first_digit = None
        self.draw_start = None
        self.draw_end = None

    # ---------- actions ----------
    def toggle_satellite(self):
        if len(self.frames) == 0:
            return
        frame = self.frames[self.current_idx]
        mask = self.df["new_filename"] == frame
        if not mask.any():
            print("  Satellite: no rows for this frame")
            return
        current = int(self.df.loc[mask, "is_satellite"].iloc[0]) if mask.any() else 0
        new_val = 0 if current == 1 else 1
        self.df.loc[mask, "is_satellite"] = new_val
        if new_val == 1:
            if frame not in self.session_satellite_frames:
                self.session_satellite_frames.add(frame)
                self.session_satellite += 1
            print("  SATELLITE flagged: " + frame)
        else:
            if frame in self.session_satellite_frames:
                self.session_satellite_frames.discard(frame)
                self.session_satellite = max(0, self.session_satellite - 1)
            print("  Satellite cleared: " + frame)
        self.unsaved = True

    def trash_frame(self):
        if len(self.frames) == 0:
            return
        frame = self.frames[self.current_idx]
        path = os.path.join(IMAGES_DIR, frame)
        trash_path = os.path.join(TRASH_DIR, frame)
        mask = self.df["new_filename"] == frame
        saved_rows = self.df[mask].copy()
        self.df = self.df[~mask].reset_index(drop=True)
        if os.path.exists(path):
            try:
                shutil.move(path, trash_path)
            except Exception as e:
                print("  Move error: " + str(e))
                self.df = pd.concat([self.df, saved_rows], ignore_index=True)
                return
        self.undo_stack.append((frame, trash_path, path, saved_rows, self.current_idx))
        self.frames.pop(self.current_idx)
        if self.current_idx >= len(self.frames):
            self.current_idx = max(0, len(self.frames) - 1)
        self.unsaved = True
        self.selected_box = -1
        self.session_trashed += 1
        self.trash_since_save += 1
        print("  TRASH: " + frame + " (-" + str(len(saved_rows)) + " label)")
        if self.trash_since_save >= AUTOSAVE_EVERY:
            print("  [auto-save]")
            self.save_csv()

    def undo(self):
        if not self.undo_stack:
            print("  Nothing to undo"); return
        frame, trash_path, orig_path, saved_rows, _ = self.undo_stack.pop()
        if os.path.exists(trash_path):
            try:
                shutil.move(trash_path, orig_path)
            except Exception as e:
                print("  Undo error: " + str(e)); return
        self.df = pd.concat([self.df, saved_rows], ignore_index=True)
        self.frames.append(frame); self.frames.sort()
        self.current_idx = self.frames.index(frame)
        self.unsaved = True
        self.session_trashed = max(0, self.session_trashed - 1)
        print("  UNDO: " + frame)

    def delete_selected_box(self):
        idx = self.df_index_of_selected()
        if idx is None:
            return
        self.df = self.df.drop(idx).reset_index(drop=True)
        self.selected_box = -1
        self.unsaved = True
        print("  Box deleted")

    def clear_frame_boxes(self):
        frame = self.frames[self.current_idx]
        mask = self.df["new_filename"] == frame
        n = int(mask.sum())
        if n == 0:
            return
        self.df = self.df[~mask].reset_index(drop=True)
        self.selected_box = -1
        self.unsaved = True
        print("  Cleared frame boxes (-" + str(n) + ")")

    def jump_unreviewed(self):
        for i in range(self.current_idx + 1, len(self.frames)):
            frame = self.frames[i]
            mask = self.df["new_filename"] == frame
            sub = self.df[mask]
            if len(sub) == 0 or not sub["reviewed"].any():
                self.current_idx = i
                self.selected_box = -1
                return
        print("  No unreviewed frame ahead")

    def mark_reviewed_and_next(self):
        frame = self.frames[self.current_idx]
        mask = self.df["new_filename"] == frame
        was_reviewed = False
        if mask.any():
            was_reviewed = bool(self.df.loc[mask, "reviewed"].all())
            if not was_reviewed:
                self.df.loc[mask, "reviewed"] = True
                self.unsaved = True
        if not was_reviewed:
            self.session_passed += 1
        self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
        self.selected_box = -1

    def change_selected_class(self):
        """Copy selected box into pending, delete it, enter class-select mode."""
        idx = self.df_index_of_selected()
        if idx is None:
            print("  Select a box first"); return
        try:
            cx = float(self.df.at[idx, "cx"]); cy = float(self.df.at[idx, "cy"])
            w = float(self.df.at[idx, "w"]); h = float(self.df.at[idx, "h"])
        except Exception:
            return
        if any(pd.isna(v) for v in (cx, cy, w, h)):
            return
        iw = self.orig_w if self.orig_w > 0 else 1
        ih = self.orig_h if self.orig_h > 0 else 1
        x1, y1, x2, y2 = self.yolo_to_pixel(cx, cy, w, h, iw, ih)
        self.pending_box = (int(x1), int(y1), int(x2), int(y2))
        self.df = self.df.drop(idx).reset_index(drop=True)
        self.class_select_mode = True
        self.first_digit = None
        self.selected_box = -1

    def cycle_selection(self):
        frame = self.frames[self.current_idx]
        boxes_df, _ = self.get_boxes(frame)
        n = len(boxes_df)
        if n == 0:
            return
        self.selected_box = (self.selected_box + 1) % n

    # ---------- render ----------
    def render(self):
        frame = self.frames[self.current_idx]
        path = os.path.join(IMAGES_DIR, frame)
        img = imread_unicode(path)
        if img is None:
            img = 64 * np.ones((480, 640, 3), dtype="uint8")
            cv2.putText(img, "Image failed to load", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        self.orig_h, self.orig_w = img.shape[:2]
        boxes_df, _ = self.get_boxes(frame)

        for i, (_, row) in enumerate(boxes_df.iterrows()):
            try:
                cx = float(row["cx"]); cy = float(row["cy"])
                w = float(row["w"]); h = float(row["h"])
            except (ValueError, TypeError):
                continue
            if any(pd.isna(v) for v in (cx, cy, w, h)):
                continue
            iw = self.orig_w if self.orig_w > 0 else 1
            ih = self.orig_h if self.orig_h > 0 else 1
            x1, y1, x2, y2 = self.yolo_to_pixel(cx, cy, w, h, iw, ih)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cname = str(row.get("class_name", "")) if not pd.isna(row.get("class_name", "")) else ""
            cid = self.NAME_TO_ID.get(cname, -1)
            color = CLASS_COLORS.get(cid, name_color(cname))
            if i == self.selected_box:
                color = (255, 255, 255)
            thick = 3 if i == self.selected_box else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            label = cname if cname else "?"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - ts[1] - 6), (x1 + ts[0] + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if self.edit_mode and i == self.selected_box:
                hs = 6
                ccx, ccy = (x1 + x2) // 2, (y1 + y2) // 2
                handles = [(x1, y1), (ccx, y1), (x2, y1),
                           (x1, ccy), (x2, ccy),
                           (x1, y2), (ccx, y2), (x2, y2)]
                for hx, hy in handles:
                    cv2.rectangle(img, (hx - hs, hy - hs), (hx + hs, hy + hs), (0, 255, 0), -1)
                    cv2.rectangle(img, (hx - hs, hy - hs), (hx + hs, hy + hs), (255, 255, 255), 1)

        if self.draw_mode and self.drawing and self.draw_start and self.draw_end:
            cv2.rectangle(img, self.draw_start, self.draw_end, (0, 255, 0), 2)
        if self.pending_box:
            bx1, by1, bx2, by2 = self.pending_box
            cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 3)

        sx = DISPLAY_W / max(1, self.orig_w)
        sy = DISPLAY_H / max(1, self.orig_h)
        scale = min(sx, sy)
        self.scale = scale
        new_w = int(self.orig_w * scale); new_h = int(self.orig_h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        disp = cv2.resize(img, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype="uint8")
        self.x_offset = (DISPLAY_W - new_w) // 2
        self.y_offset = (DISPLAY_H - new_h) // 2
        canvas[self.y_offset:self.y_offset + new_h,
               self.x_offset:self.x_offset + new_w] = disp

        # Top bar
        cv2.rectangle(canvas, (0, 0), (DISPLAY_W, 50), (30, 30, 30), -1)
        is_reviewed = bool(boxes_df["reviewed"].any()) if len(boxes_df) > 0 else False
        is_sat = bool((boxes_df["is_satellite"] == 1).any()) if len(boxes_df) > 0 else False
        status = "REVIEWED" if is_reviewed else "PENDING"
        scol = (0, 255, 0) if is_reviewed else (0, 200, 255)
        info = ("[" + str(self.current_idx + 1) + "/" + str(len(self.frames)) + "] " +
                frame + "  Box:" + str(len(boxes_df)) +
                "  Sel:" + (str(self.selected_box + 1) if self.selected_box >= 0 else "-") +
                "  " + status +
                ("  SAT" if is_sat else "") +
                "  Pass:" + str(self.session_passed) +
                "  Trash:" + str(self.session_trashed) +
                "  Sat:" + str(self.session_satellite) +
                ("  UNSAVED" if self.unsaved else ""))
        cv2.putText(canvas, info, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, scol, 1, cv2.LINE_AA)

        # Mode bar
        cv2.rectangle(canvas, (0, DISPLAY_H - 30), (DISPLAY_W, DISPLAY_H), (30, 30, 30), -1)
        if self.class_select_mode:
            mode = "PICK CLASS: 0-9 direct, 1+0-9=10-19, 2+0=20, ESC=cancel"
            if self.first_digit is not None:
                mode = str(self.first_digit) + "_  second digit or ESC"
        elif self.edit_mode:
            mode = "EDIT MODE: drag corners/edges, center=move, ESC=exit"
        elif self.draw_mode:
            mode = "DRAW MODE: mouse drag, ESC=exit"
        else:
            mode = "D:next A:prev SPACE:review T:trash N:draw E:edit C:class TAB:cycle X:del R:clear F:unreviewed U:satellite Z:undo S:save Q:quit H:help"
        cv2.putText(canvas, mode, (10, DISPLAY_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        if self.class_select_mode:
            self.draw_class_panel(canvas)
            self.box_list_rects = []
            self.box_list_panel_rect = None
        else:
            self.draw_box_list_panel(canvas, boxes_df)
        if self.show_help_overlay:
            self.draw_help(canvas)
        return canvas

    def draw_box_list_panel(self, canvas, boxes_df):
        self.box_list_rects = []
        self.box_list_panel_rect = None
        if boxes_df is None or len(boxes_df) == 0:
            return
        row_h = 22
        max_rows = min(len(boxes_df), 24)
        panel_w = 260
        panel_h = 40 + max_rows * row_h
        px = DISPLAY_W - panel_w - 12
        py = 60
        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (80, 200, 255), 1)
        cv2.putText(canvas, "BOXES (" + str(len(boxes_df)) + ")", (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        self.box_list_panel_rect = (px, py, px + panel_w, py + panel_h)
        for i, (_, row) in enumerate(boxes_df.iterrows()):
            if i >= max_rows:
                break
            y_top = py + 32 + i * row_h
            y_bot = y_top + row_h - 2
            cname = str(row.get("class_name", "")) if not pd.isna(row.get("class_name", "")) else "?"
            cid = self.NAME_TO_ID.get(cname, -1)
            color = CLASS_COLORS.get(cid, name_color(cname))
            if i == self.selected_box:
                cv2.rectangle(canvas, (px + 4, y_top), (px + panel_w - 4, y_bot), (50, 50, 50), -1)
                cv2.rectangle(canvas, (px + 4, y_top), (px + panel_w - 4, y_bot), (255, 255, 255), 1)
            cv2.rectangle(canvas, (px + 10, y_top + 4), (px + 24, y_bot - 4), color, -1)
            text = str(i + 1) + ": " + cname
            cv2.putText(canvas, text, (px + 32, y_top + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            self.box_list_rects.append((y_top, y_bot, i))

    def draw_class_panel(self, canvas):
        if not self.CLASSES:
            return
        panel_w = 320; panel_h = min(580, 50 + len(self.CLASSES) * 22)
        px = DISPLAY_W - panel_w - 20; py = 60
        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (0, 255, 0), 2)
        cv2.putText(canvas, "PICK CLASS:", (px + 10, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        for i, (cid, cname) in enumerate(sorted(self.CLASSES.items())):
            ypos = py + 45 + i * 22
            color = CLASS_COLORS.get(cid, name_color(cname))
            key = str(cid) if cid < 10 else ("1+" + str(cid - 10) if cid < 20 else "2+0")
            line = key + " : " + cname
            cv2.putText(canvas, line, (px + 12, ypos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def draw_help(self, canvas):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (50, 50), (DISPLAY_W - 50, DISPLAY_H - 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        lines = [
            "CONTROLS",
            "",
            "D / Right Arrow    : next frame",
            "A / Left Arrow     : previous frame",
            "SPACE              : mark reviewed + next",
            "T                  : TRASH frame (image + all CSV rows)",
            "Click              : select box (image or right-side panel)",
            "TAB                : cycle box selection",
            "X                  : delete selected box",
            "R                  : remove every box in this frame",
            "N                  : draw new box (mouse drag)",
            "E                  : edit selected box (corners / edges)",
            "C                  : change class of selected box",
            "F                  : jump to next unreviewed frame",
            "U                  : toggle is_satellite flag",
            "Z                  : undo last trash",
            "S / Q              : save / save and quit",
            "ESC                : cancel edit / draw / class-pick mode",
            "",
            "After draw or class change:",
            "  0-9 direct | 1+0-9 = 10-19 | 2+0 = 20",
            "",
            "Auto-save: every " + str(AUTOSAVE_EVERY) + " trashes the CSV is flushed.",
        ]
        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(canvas, line, (80, 90 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ---------- main loop ----------
    def run(self):
        wname = "vision-label-curator"
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, DISPLAY_W, DISPLAY_H)
        cv2.setMouseCallback(wname, self.mouse_cb)

        while True:
            if len(self.frames) == 0:
                print("No frames left"); break
            if self.dirty:
                cv2.imshow(wname, self.render())
                self.dirty = False
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue
            self.dirty = True

            # Class-pick mode
            if self.class_select_mode:
                if key == 27:
                    self.cancel_pending()
                elif self.first_digit is not None:
                    if ord('0') <= key <= ord('9'):
                        cid = self.first_digit * 10 + (key - ord('0'))
                        if cid in self.CLASSES:
                            self.add_pending_box(cid)
                        else:
                            self.add_pending_box(self.first_digit)
                    else:
                        self.add_pending_box(self.first_digit)
                elif key in (ord('1'), ord('2')):
                    self.first_digit = key - ord('0')
                elif ord('0') <= key <= ord('9'):
                    self.add_pending_box(key - ord('0'))
                continue

            if key == ord('q'):
                self.save_csv(); break
            elif key == ord('s'):
                self.save_csv()
            elif key in (ord('d'), 83):
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
                self.selected_box = -1; self.edit_mode = False
            elif key in (ord('a'), 81):
                self.current_idx = max(0, self.current_idx - 1)
                self.selected_box = -1; self.edit_mode = False
            elif key == ord(' '):
                self.mark_reviewed_and_next()
            elif key == ord('t'):
                self.trash_frame()
            elif key == ord('z'):
                self.undo()
            elif key == ord('x'):
                self.delete_selected_box()
            elif key == ord('r'):
                self.clear_frame_boxes()
            elif key == ord('e'):
                if self.selected_box >= 0:
                    self.edit_mode = not self.edit_mode
                    if self.edit_mode:
                        self.draw_mode = False
                        print("Edit ON")
                    else:
                        print("Edit OFF"); self.edit_dragging = False
                else:
                    print("  Select a box first")
            elif key == ord('n'):
                self.draw_mode = not self.draw_mode
                self.edit_mode = False
                if self.draw_mode:
                    print("Draw ON")
                else:
                    print("Draw OFF")
                    self.drawing = False
                    self.draw_start = None
                    self.draw_end = None
            elif key == ord('c'):
                self.change_selected_class()
            elif key == 9:  # TAB
                self.cycle_selection()
            elif key == ord('f'):
                self.jump_unreviewed()
            elif key == ord('u'):
                self.toggle_satellite()
            elif key == ord('h'):
                self.show_help_overlay = not self.show_help_overlay
            elif key == 27:
                if self.edit_mode:
                    self.edit_mode = False; self.edit_dragging = False
                    print("Edit OFF")
                elif self.draw_mode:
                    self.draw_mode = False; self.drawing = False
                    self.draw_start = None; self.draw_end = None
                    print("Draw OFF")
                else:
                    self.show_help_overlay = False

        cv2.destroyAllWindows()
        print("\nSession: Pass=" + str(self.session_passed) +
              " Trash=" + str(self.session_trashed))


if __name__ == "__main__":
    Reviewer().run()
