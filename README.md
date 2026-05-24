# data-cleaner-cv

**Curate million-row object-detection datasets with your keyboard.**
A keyboard-driven desktop tool for reviewing, cleaning, editing, and organizing
image + bounding-box datasets — built for humans, not for batch scripts.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/downloads/)
[![Status: Working tool](https://img.shields.io/badge/status-working_tool-success.svg)]()

![Review window with bounding boxes and box-list panel](docs/screenshot-review.png)

---

## Why use it

- **Built for scale.** Tested on a 1M-row CSV. Auto-save batches, no per-keypress disk thrash.
- **Keyboard-only.** No mouse hunting. Trash a frame in one keystroke; sweep hundreds in two.
- **Schema-agnostic.** Drop in CSV, YOLO `.txt` folder, COCO JSON, or Pascal VOC XML — auto-detected.
- **Crash-safe.** Rotating backups, resume from last frame, single-key undo.
- **Smart batch operations.** Anchor-distance counter + `B` to sweep unreviewed frames back to your last known-good point.

---

## Quick start

```bash
git clone https://github.com/AlpcanCepikk/data-cleaner-cv.git
cd data-cleaner-cv
pip install -r requirements.txt
python review.py --images /path/to/images --labels /path/to/labels
```

No arguments? A wizard asks for paths. Drop `review.py` next to `images/` +
`labels.csv` and it just runs.

![Thumbnail strip showing neighbouring frames](docs/screenshot-thumbnails.png)

---

## What it does

- Displays each image with its bounding boxes drawn on top.
- Lets you **trash** an entire frame in one keystroke — image moves to
  `_Trash/`, rows drop from the CSV in memory, periodic auto-save.
- Lets you **edit** existing boxes (drag corners/edges, move), **draw** new
  boxes, **delete** individual boxes, and **change the class** of selected boxes.
- Maintains an `is_satellite` metadata column you can toggle per frame.
- Persists progress in `_review_progress.json` so closing resumes exactly
  where you left off.
- Per-frame **box list panel** on the right — click a row to select the
  corresponding box (helpful when boxes are tiny or overlapping).
- **Anchor counter** in the info bar (`Anchor:+N / -N / 0`): shows how far
  you've drifted from your last run of 5+ consecutive reviewed frames.

---

## Controls

| key                | action                                                  |
|--------------------|---------------------------------------------------------|
| `D` / `→`          | next frame                                              |
| `A` / `←`          | previous frame                                          |
| `SPACE`            | mark current frame reviewed, advance                    |
| `T`                | trash frame                                             |
| `Shift+T`          | trash next N frames, **skipping reviewed** (type count) |
| `B`                | trash every unreviewed frame between current and anchor |
| `Z`                | undo last trash / crop / batch                          |
| `Click`            | select box (on image or right-side list panel)          |
| `Shift+Click`      | add / remove box from a multi-selection                 |
| `TAB`              | cycle box selection                                     |
| `X`                | delete selected box(es)                                 |
| `R`                | remove every box in this frame                          |
| `N`                | draw new box (drag, then pick class)                    |
| `E`                | edit selected box (drag corners / edges / move)         |
| `C`                | change class of selected box                            |
| `K`                | crop a region (drag → `ENTER` apply, `ESC` cancel)      |
| `F`                | jump to next unreviewed frame                           |
| `U`                | toggle `is_satellite` flag for current frame            |
| `V` / `Shift+V`    | validate CSV (report only / drop bad rows)              |
| `P`                | toggle statistics overlay                               |
| `/`                | cycle class filter (box-list panel only)                |
| `[`                | toggle thumbnail strip                                  |
| `S`                | save CSV                                                |
| `Q`                | save and quit                                           |
| `H`                | show in-app help overlay                                |
| `ESC`              | cancel edit / draw / crop / class-pick mode             |

Class picker: type the class id with the digits `0–9`; for ids `10–19`
press `1` then a digit; for id `20` press `2` then `0`.

---

## The anchor (and `B`)

While reviewing, the info bar shows `Anchor:+N`, `-N`, `0`, or `-`:

- **Anchor** = the end of the most recent run of **5+ consecutive reviewed
  frames**. A single stray `SPACE` doesn't count — needs a real streak.
- `+N` → you've gone N frames forward from your last known-good zone.
- `B` → trashes every **unreviewed** frame between your current position and
  the anchor, then jumps you back to the anchor.

Typical use: hammer `D` through a bad stretch, `SPACE` the few keepers you
spot. When you've gone too far, press `B` — the garbage is gone, your
keepers stay, and you're back at the anchor. One `Z` undoes the whole sweep.

---

## Expected data layout

```
your_workspace/
├── images/                 # all image files (.png, .jpg, .jpeg)
├── labels.csv              # one row per bounding box
├── workspace_config.json   # optional — class id ↔ name (auto-built if missing)
├── _Trash/                 # created by the tool; trashed images land here
└── _review_progress.json   # created by the tool; resume state
```

### `labels.csv` columns

| column          | meaning                                                |
|-----------------|--------------------------------------------------------|
| `new_filename`  | image filename inside `images/`                        |
| `class_name`    | textual class label (e.g. `car`, `person`)             |
| `cx`, `cy`      | YOLO-normalized box center in `[0, 1]`                 |
| `w`, `h`        | YOLO-normalized box width/height in `[0, 1]`           |
| `img_width`     | optional; falls back to the actual image size on load  |
| `img_height`    | optional; same fallback                                |
| `reviewed`      | boolean; set to `True` when you mark a frame reviewed  |
| `is_satellite`  | 0/1 metadata flag, toggled with `U`                    |

`reviewed` and `is_satellite` are created automatically if missing.

---

## Install (detailed)

### Prerequisites

- **Python 3.8 or newer.** Check with `python --version`. On Windows, tick
  **"Add python.exe to PATH"** in the installer.
- **A desktop OpenCV build.** The tool opens a native window via
  `cv2.imshow`, so headless servers (no display) are not supported.

### Optional: virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Optional: pre-define your class list

Copy `workspace_config.example.json` to `workspace_config.json` inside your
workspace folder and edit the `classes` map. Otherwise the tool builds one
automatically from the most-frequent labels in your dataset on first run.

---

## Run

```bash
python review.py --images /path/to/images --labels /path/to/labels
```

`--labels` accepts any of:
- a `.csv` file,
- a `.json` COCO file,
- a folder of YOLO `.txt` files,
- a folder of Pascal VOC `.xml` files.

The format is auto-detected and normalized into a working `labels.csv` inside
the workspace (`--workspace DIR`, defaults to the labels' directory).
Press `H` any time for an in-app help overlay. Press `Q` to save and exit —
the next run resumes from the same frame index.

### Export

```bash
python review.py --images <dir> --labels <workspace>/labels.csv --export-yolo out_dir
python review.py --images <dir> --labels <workspace>/labels.csv --export-coco out.json
```

---

## Cropping

Press `K`, drag a region, press `ENTER`. The cropped image overwrites the
original on disk (the original is backed up under `workspace/_originals/`
so `Z` can undo). All boxes are re-normalized to the new image size; a box
that straddles the crop edge is clipped, and dropped entirely if less than
30% of its original area remains inside. `ESC` cancels.

---

## Auto-save & backups

Auto-saves every **50** trashes by default (`--autosave N` to change). Every
save also writes a timestamped backup under `workspace/_backups/`, keeping
the most recent 3. Force-save with `S` or `Q`.

---

## Tips

- Use `F` to jump to the next unreviewed frame — much faster than walking.
- Use `Z` immediately after a wrong trash to undo it.
- When boxes overlap densely, click the **right-side list** instead of the
  image.
- After a session, split by satellite flag with pandas:
  `df[df["is_satellite"] == 1]["new_filename"].unique()`.

---

## Roadmap

**Done**
- English-only UI; CLI args + first-run wizard
- Schema autodetect (CSV, YOLO `.txt`, COCO JSON, Pascal VOC XML)
- YOLO + COCO export
- Thumbnail strip, statistics overlay, CSV validator, rotating backups
- Region crop, batch trash, anchor counter, smart back-to-anchor sweep
- Bulk-trash O(N→1) optimization (single isin mask, single df rebuild)

**Still planned**
- Universal class discovery on `N` / `C`
- Export back to Pascal VOC
- Bounding-box snapping and keyboard nudging in edit mode
- Tk / PyQt rewrite for a proper menu bar and file pickers

If you have a wishlist item, [open an issue](https://github.com/AlpcanCepikk/data-cleaner-cv/issues).

---

## License

GPL-3.0 — see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). You may use,
modify, and redistribute under the terms of the GPL; redistributions must
remain open source under GPL-3.0 and must keep the original author and
project attribution visible.
