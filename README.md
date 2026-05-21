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

### 1. Prerequisites

- **Python 3.8 or newer.** Check with:

  ```bash
  python --version
  ```

  If Python is not installed, get it from [python.org](https://www.python.org/downloads/).
  On Windows, make sure to tick **"Add python.exe to PATH"** in the installer.
- **A desktop OpenCV build.** The tool opens a native window via
  `cv2.imshow`, so headless servers (no display) are not supported.

### 2. Clone the repo

```bash
git clone https://github.com/AlpcanCepikk/vision-label-curator.git
cd vision-label-curator
```

(If you don't have git, you can also download the ZIP from the GitHub page
and extract it.)

### 3. (Recommended) Create a virtual environment

Keeps the tool's dependencies isolated from your system Python:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `opencv-python`, `numpy`, and `pandas`.

### 5. Place your dataset next to `review.py`

Until the upcoming CLI-args refactor (see roadmap), the tool resolves
`images/` and `labels.csv` **relative to its own location**. The simplest
setup is to copy `review.py` (and optionally `workspace_config.example.json`)
into your dataset folder:

```
your_workspace/
├── images/              # your images
├── labels.csv           # your label rows
└── review.py            # copied from this repo
```

Alternatively, you can leave `review.py` where it is and run it with the
working directory set to your dataset folder — but the **`images/` and
`labels.csv` paths are read relative to `review.py`'s own folder**, so
running it from elsewhere will not work yet.

### 6. (Optional) Pre-define your class list

If you already know the class set you want, copy
`workspace_config.example.json` to `workspace_config.json` next to
`review.py` and edit the `classes` map. Otherwise the tool will build one
automatically from the most-frequent labels in your CSV on first run.

## Run

```bash
python review.py
```

A native window opens with the first frame; press `H` any time for an
in-app help overlay. Press `Q` to save and exit — the next run resumes from
the same frame index.

---

## Getting started

A typical first session looks like this:

1. **Lay out your data** as shown in [Expected data layout](#expected-data-layout):
   put all images in `images/` and your label rows in `labels.csv` (YOLO-normalized
   `cx, cy, w, h`). Drop `review.py` next to them.
2. **Install dependencies** with `pip install -r requirements.txt`.
3. **Run** `python review.py`. On the very first run, the tool:
   - scans `images/` and drops any CSV rows that point to missing images;
   - builds a class id ↔ name map from the top 21 most-frequent `class_name`
     values in your CSV, and writes it to `workspace_config.json` so the
     mapping stays stable across runs. Edit that file by hand if you want
     a different ordering or set of classes.
4. **Review frames one by one.** For each image:
   - press `D` (or `→`) to go to the next frame without changing anything;
   - press `SPACE` to mark the current frame as reviewed and advance;
   - press `T` to trash the whole frame — the image moves to `_Trash/` and
     all rows for that filename are removed from the CSV in memory;
   - click a box on the image (or its row in the right-side panel) to select
     it, then press `E` to drag its corners/edges, `X` to delete it, or
     `C` to change its class;
   - press `N` to draw a brand-new box: drag a rectangle, then pick the class
     by typing the class id;
   - press `U` to flag the frame as a satellite/aerial shot (metadata only;
     does not affect labels).
5. **Save and quit** with `Q` (or press `S` to save mid-session). The CSV is
   also auto-saved every `AUTOSAVE_EVERY` trashes (default `50`).
6. **Resume any time.** Progress is stored in `_review_progress.json`; the
   next run picks up at the same frame index.

### Tips

- Use `F` to jump straight to the next frame that has not been reviewed yet —
  much faster than walking through reviewed frames one at a time.
- Use `Z` immediately after `T` to undo a wrong trash; the image is moved back
  out of `_Trash/` and the rows are restored.
- If many small boxes overlap, the **right-side box list panel** is faster
  than clicking on the image — each row shows the box index and class, and
  clicking it selects that box.
- `is_satellite` is just a column on each row. After a session, you can split
  the dataset by filtering `labels.csv` with pandas:
  `df[df["is_satellite"] == 1]["new_filename"].unique()`.

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
