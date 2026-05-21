# vision-label-curator

A keyboard-driven desktop tool for **reviewing, cleaning, editing, and organizing
object-detection datasets** that combine images with a CSV of bounding-box
labels. It is designed for human-in-the-loop curation of large datasets
(hundreds of thousands of rows), not for autonomous cleaning.

> **Status:** working in-house tool — currently being generalized for public use.
> The first public release will add automatic schema detection, full English
> UI, and a guided first-run setup. See the [Roadmap](#roadmap) below.

---

## What it does

- Displays each image with its bounding boxes drawn on top.
- Lets you **trash** an entire frame in one keystroke — the image is moved to
  `_Trash/` and every matching row is removed from the CSV in memory, with
  periodic auto-save so a 1M-row CSV is not rewritten on every keypress.
- Lets you **edit** existing boxes (drag corners/edges, move), **draw** new
  boxes, **delete** individual boxes, and **change the class** of the selected
  box.
- Maintains an `is_satellite` metadata column you can toggle per frame, so you
  can later split satellite/aerial imagery from ground/air-to-air imagery
  without touching the training labels.
- Persists progress in `_review_progress.json` so closing the window resumes
  exactly where you left off.
- Per-frame **box list panel** on the right — click a row to select the
  corresponding box (helpful when boxes are small or overlapping).

---

## Expected data layout

```
your_workspace/
├── images/                 # all image files (.png, .jpg, .jpeg)
├── labels.csv              # one row per bounding box
├── workspace_config.json   # optional — class id ↔ name mapping (auto-built if missing)
├── _Trash/                 # created by the tool; trashed images land here
└── _review_progress.json   # created by the tool; resume state
```

### `labels.csv` columns

The tool currently expects a CSV with at least these columns:

| column          | meaning                                                |
|-----------------|--------------------------------------------------------|
| `new_filename`  | image filename inside `images/` (e.g. `00042.jpg`)     |
| `class_name`    | textual class label (e.g. `car`, `person`)             |
| `cx`, `cy`      | YOLO-normalized box center in `[0, 1]`                 |
| `w`, `h`        | YOLO-normalized box width/height in `[0, 1]`           |
| `img_width`     | optional; falls back to the actual image size on load  |
| `img_height`    | optional; same fallback                                |
| `reviewed`      | boolean; set to `True` when you mark a frame reviewed  |
| `is_satellite`  | 0/1 metadata flag, toggled with `U`                    |

`reviewed` and `is_satellite` are created automatically if missing.

> Generalizing this schema (autodetect Pascal/YOLO/COCO; rename columns at load;
> allow xyxy or cxcywh in pixels) is part of the roadmap.

---

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and a desktop OpenCV build (the GUI uses `cv2.imshow`).

## Run

From your workspace directory (the one that contains `images/` and
`labels.csv`):

```bash
python /path/to/vision-label-curator/review.py
```

Currently the tool resolves `images/` and `labels.csv` relative to its own
location. **Until the upcoming CLI-args refactor (see roadmap)**, the simplest
way to use it is to drop `review.py` next to your data, or symlink it.

---

## Controls

| key                | action                                                  |
|--------------------|---------------------------------------------------------|
| `D` / `→`          | next frame                                              |
| `A` / `←`          | previous frame                                          |
| `SPACE`            | mark current frame reviewed, advance                    |
| `T`                | trash frame (image → `_Trash/`, rows → dropped)         |
| `Z`                | undo last trash                                         |
| `Click`            | select box (on image or right-side list panel)          |
| `TAB`              | cycle box selection                                     |
| `X`                | delete selected box                                     |
| `R`                | remove every box in this frame                          |
| `N`                | draw new box (drag, then pick class)                    |
| `E`                | edit selected box (drag corners / edges / move)         |
| `C`                | change class of selected box                            |
| `F`                | jump to next unreviewed frame                           |
| `U`                | toggle `is_satellite` flag for current frame            |
| `S`                | save CSV                                                |
| `Q`                | save and quit                                           |
| `H`                | show in-app help overlay                                |
| `ESC`              | cancel edit / draw / class-pick mode                    |

When a class is being assigned (after drawing a new box or after pressing `C`):
type the class id with the digits `0–9`; for ids `10–19` press `1` then a
digit; for id `20` press `2` then `0`.

---

## Auto-save

The tool calls auto-save every **50** trashes by default (`AUTOSAVE_EVERY`
constant in `review.py`). This avoids rewriting a million-row CSV on every
keystroke while still being durable against crashes. You can also force a save
with `S` or `Q`.

---

## Roadmap

The current `review.py` reflects how the tool is used internally on a single
large dataset. To make it a clean public tool, the following work is planned:

- [ ] **English-only UI** (current build has some Turkish strings).
- [ ] **CLI args**: `python review.py --images <dir> --labels <csv>` instead of
      hard-coded relative paths.
- [ ] **First-run setup wizard** that detects the schema automatically:
      Pascal VOC XML, YOLO `.txt`, COCO JSON, or generic CSV — and proposes a
      column mapping.
- [ ] **Universal class discovery**: on `N` (draw new box) and `C` (change
      class), classes are pulled from the live CSV *plus* user-defined ones,
      so any dataset works out of the box.
- [ ] Export back to YOLO `.txt` / COCO JSON / Pascal VOC.
- [ ] Optional thumbnail strip showing nearby frames.

If you have a wishlist item, open an issue.

---

## License

TBD — to be added before the first tagged release.
