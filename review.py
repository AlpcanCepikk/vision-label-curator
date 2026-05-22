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
# data-cleaner-cv — bbox review & curation tool
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

SUPPORTED_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class Paths:
    """All filesystem paths used by the tool, resolved from CLI args / wizard."""
    def __init__(self, workspace, images_dir, csv_path,
                 trash_dir=None, progress_file=None, config_path=None):
        self.workspace = workspace
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.trash_dir = trash_dir or os.path.join(workspace, "_Trash")
        self.progress_file = progress_file or os.path.join(workspace, "_review_progress.json")
        self.config_path = config_path or os.path.join(workspace, "workspace_config.json")


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


def build_classes(df, config_path):
    """Load class id->name map from workspace_config.json, or build the top-21
    most-frequent class names from the CSV on first run and persist them."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
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
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"classes": {str(k): v for k, v in classes.items()}},
                          f, ensure_ascii=False, indent=2)
            print("[Auto] Built top-" + str(len(classes)) +
                  " classes from CSV and wrote " + config_path)
        except Exception as e:
            print("Could not write workspace_config.json: " + str(e))
        print("[Classes] " + str(list(classes.values())))
        return classes
    return {}


# ============================================================
# Format autodetect + loaders
# ============================================================

CANONICAL_COLUMNS = [
    "new_filename", "class_name", "cx", "cy", "w", "h",
    "img_width", "img_height", "reviewed", "is_satellite",
]


def detect_label_format(labels_path):
    """Return one of: 'csv', 'coco', 'yolo', 'voc', 'unknown'."""
    if os.path.isfile(labels_path):
        ext = os.path.splitext(labels_path)[1].lower()
        if ext == ".csv":
            return "csv"
        if ext == ".json":
            return "coco"
        if ext in (".xml",):
            return "voc"
        return "unknown"
    if os.path.isdir(labels_path):
        # Scan a few files to guess
        for entry in os.listdir(labels_path):
            low = entry.lower()
            if low.endswith(".xml"):
                return "voc"
            if low.endswith(".txt") and not low.startswith("classes"):
                return "yolo"
        return "unknown"
    return "unknown"


def load_csv(labels_path):
    return pd.read_csv(labels_path)


def load_yolo(labels_dir, images_dir):
    """YOLO: one .txt per image, each line `class_id cx cy w h` (normalized).
    Class names from classes.txt or names.txt next to labels (if present)."""
    rows = []
    class_names = {}
    for cand in ("classes.txt", "names.txt", "labels.txt"):
        p = os.path.join(labels_dir, cand)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        class_names[i] = name
            break
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)
                   if f.lower().endswith(SUPPORTED_IMG_EXTS)}
    for entry in os.listdir(labels_dir):
        if not entry.lower().endswith(".txt"):
            continue
        if entry.lower() in ("classes.txt", "names.txt", "labels.txt"):
            continue
        stem = os.path.splitext(entry)[0]
        image_name = image_files.get(stem)
        if image_name is None:
            continue
        try:
            with open(os.path.join(labels_dir, entry), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    cx, cy, w, h = [float(x) for x in parts[1:5]]
                    cname = class_names.get(cid, str(cid))
                    rows.append({
                        "new_filename": image_name, "class_name": cname,
                        "cx": cx, "cy": cy, "w": w, "h": h,
                    })
        except Exception as e:
            print("Skipped " + entry + ": " + str(e))
    return pd.DataFrame(rows)


def load_coco(json_path, images_dir):
    """COCO JSON: images[], annotations[] with bbox=[x, y, w, h] in pixels."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgs = {img["id"]: img for img in data.get("images", [])}
    rows = []
    for ann in data.get("annotations", []):
        img = imgs.get(ann.get("image_id"))
        if img is None:
            continue
        iw, ih = float(img.get("width", 0)), float(img.get("height", 0))
        if iw <= 0 or ih <= 0:
            continue
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        cx = (x + w / 2.0) / iw
        cy = (y + h / 2.0) / ih
        nw, nh = w / iw, h / ih
        rows.append({
            "new_filename": img.get("file_name", ""),
            "class_name": cats.get(ann.get("category_id"), str(ann.get("category_id"))),
            "cx": cx, "cy": cy, "w": nw, "h": nh,
            "img_width": iw, "img_height": ih,
        })
    return pd.DataFrame(rows)


def load_voc(xml_dir, images_dir):
    """Pascal VOC: per-image .xml with bndbox xmin/ymin/xmax/ymax."""
    import xml.etree.ElementTree as ET
    rows = []
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)
                   if f.lower().endswith(SUPPORTED_IMG_EXTS)}
    for entry in os.listdir(xml_dir):
        if not entry.lower().endswith(".xml"):
            continue
        stem = os.path.splitext(entry)[0]
        image_name = image_files.get(stem)
        if image_name is None:
            continue
        try:
            tree = ET.parse(os.path.join(xml_dir, entry))
            root = tree.getroot()
            size = root.find("size")
            iw = float(size.findtext("width", "0")) if size is not None else 0.0
            ih = float(size.findtext("height", "0")) if size is not None else 0.0
            if iw <= 0 or ih <= 0:
                continue
            for obj in root.findall("object"):
                cname = obj.findtext("name", "unknown")
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
                cx = (xmin + xmax) / 2.0 / iw
                cy = (ymin + ymax) / 2.0 / ih
                w = (xmax - xmin) / iw
                h = (ymax - ymin) / ih
                rows.append({
                    "new_filename": image_name, "class_name": cname,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "img_width": iw, "img_height": ih,
                })
        except Exception as e:
            print("Skipped " + entry + ": " + str(e))
    return pd.DataFrame(rows)


def export_yolo(df, images_dir, out_dir, class_order=None):
    """Write YOLO-format .txt files (one per image) and classes.txt to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    if class_order is None:
        class_order = sorted(df["class_name"].dropna().astype(str).unique().tolist())
    name_to_id = {n: i for i, n in enumerate(class_order)}
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for name in class_order:
            f.write(name + "\n")
    n_files = 0
    for frame, sub in df.groupby("new_filename"):
        stem = os.path.splitext(str(frame))[0]
        lines = []
        for _, r in sub.iterrows():
            cname = str(r.get("class_name", ""))
            if cname not in name_to_id:
                continue
            try:
                cx, cy, w, h = float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"])
            except Exception:
                continue
            if any(pd.isna(v) for v in (cx, cy, w, h)):
                continue
            lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                name_to_id[cname], cx, cy, w, h))
        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        n_files += 1
    print("[Export YOLO] " + str(n_files) + " files + classes.txt -> " + out_dir)


def export_coco(df, images_dir, out_file):
    """Write a minimal COCO JSON to out_file."""
    class_order = sorted(df["class_name"].dropna().astype(str).unique().tolist())
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_order)]
    name_to_cat = {n: i + 1 for i, n in enumerate(class_order)}
    images = []
    annotations = []
    image_id_map = {}
    next_img_id = 1
    next_ann_id = 1
    for frame, sub in df.groupby("new_filename"):
        img_path = os.path.join(images_dir, str(frame))
        iw = ih = 0
        if pd.notna(sub.iloc[0].get("img_width", np.nan)) and pd.notna(sub.iloc[0].get("img_height", np.nan)):
            iw = int(sub.iloc[0]["img_width"]); ih = int(sub.iloc[0]["img_height"])
        if (iw <= 0 or ih <= 0) and os.path.exists(img_path):
            im = imread_unicode(img_path)
            if im is not None:
                ih, iw = im.shape[:2]
        if iw <= 0 or ih <= 0:
            continue
        image_id_map[frame] = next_img_id
        images.append({"id": next_img_id, "file_name": str(frame),
                       "width": int(iw), "height": int(ih)})
        for _, r in sub.iterrows():
            cname = str(r.get("class_name", ""))
            if cname not in name_to_cat:
                continue
            try:
                cx, cy, w, h = float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"])
            except Exception:
                continue
            if any(pd.isna(v) for v in (cx, cy, w, h)):
                continue
            x = (cx - w / 2.0) * iw
            y = (cy - h / 2.0) * ih
            bw = w * iw; bh = h * ih
            annotations.append({
                "id": next_ann_id, "image_id": next_img_id,
                "category_id": name_to_cat[cname],
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0,
            })
            next_ann_id += 1
        next_img_id += 1
    os.makedirs(os.path.dirname(os.path.abspath(out_file)) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": categories}, f, ensure_ascii=False, indent=2)
    print("[Export COCO] " + str(len(images)) + " images, " +
          str(len(annotations)) + " annotations -> " + out_file)


def load_labels(labels_path, images_dir):
    fmt = detect_label_format(labels_path)
    print("[Loader] Detected format: " + fmt)
    if fmt == "csv":
        return load_csv(labels_path), fmt
    if fmt == "yolo":
        return load_yolo(labels_path, images_dir), fmt
    if fmt == "coco":
        return load_coco(labels_path, images_dir), fmt
    if fmt == "voc":
        return load_voc(labels_path, images_dir), fmt
    print("ERROR: Could not detect label format for: " + labels_path)
    sys.exit(1)


# ============================================================
# CLI args + first-run wizard
# ============================================================

def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(
        prog="data-cleaner-cv",
        description="Review, clean, and edit object-detection datasets.")
    p.add_argument("--images", help="Directory containing image files.")
    p.add_argument("--labels", help="Path to labels: .csv, .json (COCO), folder of .txt (YOLO), or folder of .xml (VOC).")
    p.add_argument("--workspace", help="Workspace dir (where the normalized labels.csv, trash, and progress files live). Defaults to alongside --labels.")
    p.add_argument("--autosave", type=int, default=50, help="Auto-save the CSV every N trashes (default 50).")
    p.add_argument("--export-yolo", metavar="OUTDIR",
                   help="Convert the workspace's labels.csv to a YOLO label folder, then exit.")
    p.add_argument("--export-coco", metavar="OUTFILE",
                   help="Convert the workspace's labels.csv to a COCO JSON file, then exit.")
    return p.parse_args(argv)


def _prompt(prompt_text, default=None):
    suffix = (" [" + default + "]") if default else ""
    val = input(prompt_text + suffix + ": ").strip()
    return val or default


def run_wizard():
    """Interactive prompts when CLI args are missing."""
    print("")
    print("=== data-cleaner-cv: first-run setup ===")
    print("(answers are remembered in the workspace; you can also pass them as CLI args)")
    images = ""
    while not images or not os.path.isdir(images):
        images = _prompt("Images folder (absolute path)")
        if not images:
            print("  Required."); continue
        if not os.path.isdir(images):
            print("  Not a directory: " + images)
    labels = ""
    while not labels or not os.path.exists(labels):
        labels = _prompt("Labels (CSV file, COCO .json, YOLO folder, or VOC folder)")
        if not labels:
            print("  Required."); continue
        if not os.path.exists(labels):
            print("  Not found: " + labels)
    workspace_default = os.path.dirname(os.path.abspath(labels)) \
        if os.path.isfile(labels) else os.path.abspath(labels)
    workspace = _prompt("Workspace folder (defaults to labels' directory)", workspace_default)
    if not os.path.isdir(workspace):
        os.makedirs(workspace, exist_ok=True)
    return images, labels, workspace


def resolve_paths(args):
    """Resolve CLI args (with wizard fallback) into a Paths object plus the
    raw labels-path the user gave us (which may not be the canonical CSV)."""
    images = args.images
    labels = args.labels
    workspace = args.workspace
    # Backward compat: if no args, fall back to script_dir/{images,labels.csv}
    if not images and not labels:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        legacy_images = os.path.join(script_dir, "images")
        legacy_csv = os.path.join(script_dir, "labels.csv")
        if os.path.isdir(legacy_images) and os.path.exists(legacy_csv):
            print("[Compat] No CLI args; using script_dir/images and script_dir/labels.csv")
            images, labels, workspace = legacy_images, legacy_csv, script_dir
    if not images or not labels:
        images, labels, workspace = run_wizard()
    if not workspace:
        workspace = os.path.dirname(os.path.abspath(labels)) \
            if os.path.isfile(labels) else os.path.abspath(labels)
    workspace = os.path.abspath(workspace)
    os.makedirs(workspace, exist_ok=True)
    csv_path = os.path.join(workspace, "labels.csv")
    return Paths(workspace=workspace, images_dir=os.path.abspath(images),
                 csv_path=csv_path), os.path.abspath(labels)


class Reviewer:
    def __init__(self, paths, autosave_every=50):
        self.paths = paths
        self.autosave_every = autosave_every
        os.makedirs(paths.trash_dir, exist_ok=True)
        if not os.path.exists(paths.csv_path):
            print("ERROR: " + paths.csv_path + " not found"); sys.exit(1)
        if not os.path.isdir(paths.images_dir):
            print("ERROR: " + paths.images_dir + " not found"); sys.exit(1)

        print("Loading CSV (large files may take a moment)...")
        self.df = pd.read_csv(paths.csv_path)
        print("  " + str(len(self.df)) + " rows")
        if "reviewed" not in self.df.columns:
            self.df["reviewed"] = False
        self.df["reviewed"] = self.df["reviewed"].fillna(False).astype(bool)
        if "is_satellite" not in self.df.columns:
            self.df["is_satellite"] = 0
        self.df["is_satellite"] = self.df["is_satellite"].fillna(0).astype(int)

        print("Scanning images folder...")
        files = sorted([f for f in os.listdir(paths.images_dir)
                        if f.lower().endswith(SUPPORTED_IMG_EXTS)])
        self.frames = files
        print("  " + str(len(self.frames)) + " images")

        existing = set(self.frames)
        before = len(self.df)
        self.df = self.df[self.df["new_filename"].isin(existing)].reset_index(drop=True)
        if len(self.df) < before:
            print("  Removed " + str(before - len(self.df)) + " orphan rows")
            self.df.to_csv(paths.csv_path, index=False)

        self.CLASSES = build_classes(self.df, paths.config_path)
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
        self.selected_boxes = set()  # multi-select: set of box indices (within current frame)
        self.class_filter = None     # if set, box list panel only shows this class_name
        self.show_stats_overlay = False
        self.show_thumbs = False
        self.thumb_cache = {}        # frame_name -> small BGR ndarray

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
        if os.path.exists(self.paths.progress_file):
            try:
                with open(self.paths.progress_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                idx = d.get("current_idx", 0)
                if 0 <= idx < len(self.frames):
                    self.current_idx = idx
            except Exception:
                pass

    def save_progress(self):
        with open(self.paths.progress_file, "w", encoding="utf-8") as f:
            json.dump({"current_idx": self.current_idx}, f)

    def _rotate_backup(self, keep=3):
        """Rotate up to `keep` timestamped backups of the labels CSV alongside it."""
        if not os.path.exists(self.paths.csv_path):
            return
        try:
            backup_dir = os.path.join(self.paths.workspace, "_backups")
            os.makedirs(backup_dir, exist_ok=True)
            import time
            stamp = time.strftime("%Y%m%d_%H%M%S")
            target = os.path.join(backup_dir, "labels_" + stamp + ".csv")
            shutil.copy2(self.paths.csv_path, target)
            backups = sorted(
                [f for f in os.listdir(backup_dir) if f.startswith("labels_") and f.endswith(".csv")],
                reverse=True,
            )
            for stale in backups[keep:]:
                try:
                    os.remove(os.path.join(backup_dir, stale))
                except Exception:
                    pass
        except Exception as e:
            print("  Backup error: " + str(e))

    def save_csv(self):
        self._rotate_backup(keep=3)
        self.df.to_csv(self.paths.csv_path, index=False)
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
            hit = self.find_box_at(ox, oy, boxes_df)
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                if hit >= 0:
                    if hit in self.selected_boxes:
                        self.selected_boxes.discard(hit)
                    else:
                        self.selected_boxes.add(hit)
                    self.selected_box = hit
            else:
                self.selected_box = hit
                self.selected_boxes = {hit} if hit >= 0 else set()
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
        path = os.path.join(self.paths.images_dir, frame)
        trash_path = os.path.join(self.paths.trash_dir, frame)
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
        if self.trash_since_save >= self.autosave_every:
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
        # If multi-selection has more than one entry, delete all of them.
        if len(self.selected_boxes) > 1:
            frame = self.frames[self.current_idx]
            mask = self.df["new_filename"] == frame
            idxs = self.df[mask].index.tolist()
            to_drop = [idxs[i] for i in sorted(self.selected_boxes) if 0 <= i < len(idxs)]
            if not to_drop:
                return
            self.df = self.df.drop(to_drop).reset_index(drop=True)
            n = len(to_drop)
            self.selected_box = -1
            self.selected_boxes = set()
            self.unsaved = True
            print("  Deleted " + str(n) + " boxes")
            return
        idx = self.df_index_of_selected()
        if idx is None:
            return
        self.df = self.df.drop(idx).reset_index(drop=True)
        self.selected_box = -1
        self.selected_boxes = set()
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

    def validate_csv(self, fix=False):
        """Check cx/cy/w/h ∈ [0,1], NaN box values, duplicate rows.
        Returns a summary dict. If fix=True, drop invalid/duplicate rows."""
        df = self.df
        report = {"total": int(len(df)), "nan_box": 0, "out_of_range": 0,
                  "duplicates": 0, "fixed": 0}
        for col in ("cx", "cy", "w", "h"):
            if col not in df.columns:
                print("  [Validator] Missing column: " + col)
                return report
        nan_mask = df[["cx", "cy", "w", "h"]].isna().any(axis=1)
        report["nan_box"] = int(nan_mask.sum())
        oor_mask = (
            (df["cx"] < 0) | (df["cx"] > 1) | (df["cy"] < 0) | (df["cy"] > 1) |
            (df["w"] <= 0) | (df["w"] > 1) | (df["h"] <= 0) | (df["h"] > 1)
        ) & ~nan_mask
        report["out_of_range"] = int(oor_mask.sum())
        dup_subset = [c for c in ("new_filename", "class_name", "cx", "cy", "w", "h") if c in df.columns]
        dup_mask = df.duplicated(subset=dup_subset, keep="first")
        report["duplicates"] = int(dup_mask.sum())
        print("[Validator] " + str(report["total"]) + " rows | NaN box: " +
              str(report["nan_box"]) + " | out-of-range: " +
              str(report["out_of_range"]) + " | duplicates: " + str(report["duplicates"]))
        if fix:
            drop_mask = nan_mask | oor_mask | dup_mask
            n = int(drop_mask.sum())
            if n > 0:
                self.df = df[~drop_mask].reset_index(drop=True)
                self.unsaved = True
                report["fixed"] = n
                print("  [Validator] Dropped " + str(n) + " rows.")
            else:
                print("  [Validator] Nothing to fix.")
        return report

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
        path = os.path.join(self.paths.images_dir, frame)
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
            elif i in self.selected_boxes:
                color = (0, 255, 255)
            thick = 3 if (i == self.selected_box or i in self.selected_boxes) else 2
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
            mode = "D:next A:prev SPACE:review T:trash N:draw E:edit C:class TAB:cycle X:del R:clear F:unreviewed U:sat V:validate Z:undo S:save Q:quit H:help"
        cv2.putText(canvas, mode, (10, DISPLAY_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        if self.show_thumbs:
            self.draw_thumb_strip(canvas)
        if self.class_select_mode:
            self.draw_class_panel(canvas)
            self.box_list_rects = []
            self.box_list_panel_rect = None
        else:
            self.draw_box_list_panel(canvas, boxes_df)
        if self.show_help_overlay:
            self.draw_help(canvas)
        if self.show_stats_overlay:
            self.draw_stats(canvas)
        return canvas

    def draw_box_list_panel(self, canvas, boxes_df):
        self.box_list_rects = []
        self.box_list_panel_rect = None
        if boxes_df is None or len(boxes_df) == 0:
            return
        # Apply class filter (panel-only, image still shows all)
        visible = []
        for i, (_, row) in enumerate(boxes_df.iterrows()):
            cname = str(row.get("class_name", "")) if not pd.isna(row.get("class_name", "")) else "?"
            if self.class_filter and cname != self.class_filter:
                continue
            visible.append((i, cname))
        row_h = 22
        max_rows = min(len(visible), 24)
        panel_w = 260
        panel_h = 40 + max(1, max_rows) * row_h
        px = DISPLAY_W - panel_w - 12
        py = 60
        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (80, 200, 255), 1)
        header = "BOXES " + str(len(visible)) + "/" + str(len(boxes_df))
        if self.class_filter:
            header += "  [filter: " + self.class_filter + "]"
        cv2.putText(canvas, header, (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        self.box_list_panel_rect = (px, py, px + panel_w, py + panel_h)
        for n, (i, cname) in enumerate(visible[:max_rows]):
            y_top = py + 32 + n * row_h
            y_bot = y_top + row_h - 2
            cid = self.NAME_TO_ID.get(cname, -1)
            color = CLASS_COLORS.get(cid, name_color(cname))
            highlighted = (i in self.selected_boxes) or (i == self.selected_box)
            if highlighted:
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

    def _get_thumb(self, frame_name, tw=120, th=68):
        cached = self.thumb_cache.get(frame_name)
        if cached is not None and cached.shape[1] == tw and cached.shape[0] == th:
            return cached
        path = os.path.join(self.paths.images_dir, frame_name)
        img = imread_unicode(path)
        if img is None:
            img = 32 * np.ones((th, tw, 3), dtype="uint8")
        else:
            ih, iw = img.shape[:2]
            scale = min(tw / max(1, iw), th / max(1, ih))
            new_w = max(1, int(iw * scale)); new_h = max(1, int(ih * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pad = np.zeros((th, tw, 3), dtype="uint8")
            ox = (tw - new_w) // 2; oy = (th - new_h) // 2
            pad[oy:oy + new_h, ox:ox + new_w] = img
            img = pad
        if len(self.thumb_cache) > 80:
            self.thumb_cache.clear()
        self.thumb_cache[frame_name] = img
        return img

    def draw_thumb_strip(self, canvas):
        tw, th = 120, 68
        strip_h = th + 18  # text below
        bottom_mode_h = 30
        strip_y = DISPLAY_H - bottom_mode_h - strip_h
        cv2.rectangle(canvas, (0, strip_y), (DISPLAY_W, strip_y + strip_h), (15, 15, 15), -1)
        n_total = len(self.frames)
        if n_total == 0:
            return
        # show 5 before + current + 5 after
        radius = 5
        slots = 2 * radius + 1
        spacing = 4
        total_w = slots * tw + (slots - 1) * spacing
        start_x = max(8, (DISPLAY_W - total_w) // 2)
        for k in range(slots):
            idx = self.current_idx - radius + k
            x = start_x + k * (tw + spacing)
            if 0 <= idx < n_total:
                fr = self.frames[idx]
                thumb = self._get_thumb(fr, tw, th)
                canvas[strip_y + 2:strip_y + 2 + th, x:x + tw] = thumb
                col = (0, 255, 255) if idx == self.current_idx else (90, 90, 90)
                cv2.rectangle(canvas, (x - 1, strip_y + 1),
                              (x + tw, strip_y + 2 + th), col,
                              2 if idx == self.current_idx else 1)
                label = str(idx + 1)
                cv2.putText(canvas, label, (x + 4, strip_y + th + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (x, strip_y + 2),
                              (x + tw, strip_y + 2 + th), (25, 25, 25), -1)

    def draw_stats(self, canvas):
        df = self.df
        total_rows = int(len(df))
        total_frames = len(self.frames)
        reviewed_frames = 0
        if "reviewed" in df.columns and total_frames > 0:
            reviewed_frames = int(df[df["reviewed"] == True]["new_filename"].nunique())
        sat_frames = 0
        if "is_satellite" in df.columns:
            sat_frames = int(df[df["is_satellite"] == 1]["new_filename"].nunique())
        if "class_name" in df.columns:
            vc = df["class_name"].dropna().astype(str).value_counts().head(20)
        else:
            vc = pd.Series(dtype=int)
        lines = [
            "STATISTICS",
            "",
            "Total rows         : " + str(total_rows),
            "Total frames       : " + str(total_frames),
            "Reviewed frames    : " + str(reviewed_frames) +
              ("  (" + str(int(100 * reviewed_frames / max(1, total_frames))) + "%)"),
            "Satellite frames   : " + str(sat_frames),
            "Session pass       : " + str(self.session_passed),
            "Session trash      : " + str(self.session_trashed),
            "",
            "Top classes:",
        ]
        for cname, count in vc.items():
            lines.append("  " + str(cname)[:24].ljust(24) + " " + str(int(count)))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (50, 50), (DISPLAY_W - 50, DISPLAY_H - 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(canvas, line, (80, 90 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def cycle_class_filter(self):
        if "class_name" not in self.df.columns:
            return
        names = sorted(self.df["class_name"].dropna().astype(str).unique().tolist())
        if not names:
            return
        if self.class_filter is None:
            self.class_filter = names[0]
        else:
            try:
                i = names.index(self.class_filter)
                self.class_filter = names[i + 1] if i + 1 < len(names) else None
            except ValueError:
                self.class_filter = None
        print("  Class filter: " + (self.class_filter or "<all>"))

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
            "V / Shift+V        : validate CSV (report / drop bad rows)",
            "P                  : statistics overlay",
            "/                  : cycle class filter (box list panel only)",
            "Shift+Click        : multi-select boxes (X deletes all)",
            "[                  : toggle thumbnail strip (prev/next preview)",
            "Z                  : undo last trash",
            "S / Q              : save / save and quit",
            "ESC                : cancel edit / draw / class-pick mode",
            "",
            "After draw or class change:",
            "  0-9 direct | 1+0-9 = 10-19 | 2+0 = 20",
            "",
            "Auto-save: every " + str(self.autosave_every) + " trashes the CSV is flushed.",
        ]
        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(canvas, line, (80, 90 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ---------- main loop ----------
    def run(self):
        wname = "data-cleaner-cv"
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
            elif key == ord('v'):
                self.validate_csv(fix=False)
            elif key == ord('V'):
                self.validate_csv(fix=True)
            elif key == ord('p'):
                self.show_stats_overlay = not self.show_stats_overlay
            elif key == ord('/'):
                self.cycle_class_filter()
            elif key == ord('['):
                self.show_thumbs = not self.show_thumbs
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
                    self.show_stats_overlay = False

        cv2.destroyAllWindows()
        print("\nSession: Pass=" + str(self.session_passed) +
              " Trash=" + str(self.session_trashed))


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    paths, source_labels = resolve_paths(args)
    if not os.path.exists(paths.csv_path):
        if not os.path.exists(source_labels):
            print("ERROR: labels path not found: " + source_labels); sys.exit(1)
        df, fmt = load_labels(source_labels, paths.images_dir)
        if df is None or len(df) == 0:
            print("ERROR: loader returned no rows from " + source_labels); sys.exit(1)
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = 0 if col == "is_satellite" else (False if col == "reviewed" else np.nan)
        df.to_csv(paths.csv_path, index=False)
        print("[Loader] Wrote normalized " + str(len(df)) + " rows to " + paths.csv_path
              + " (source format: " + fmt + ")")
    if getattr(args, "export_yolo", None):
        df_now = pd.read_csv(paths.csv_path)
        export_yolo(df_now, paths.images_dir, args.export_yolo)
        return
    if getattr(args, "export_coco", None):
        df_now = pd.read_csv(paths.csv_path)
        export_coco(df_now, paths.images_dir, args.export_coco)
        return
    Reviewer(paths, autosave_every=args.autosave).run()


if __name__ == "__main__":
    main()
