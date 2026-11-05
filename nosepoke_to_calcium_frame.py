# =============================================================================
# 1) IMPORTS
# =============================================================================
import os
import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# =============================================================================
# 2) DEFAULT SETTINGS (can also be changed in the window)
# =============================================================================
# Timestamp unit, used ONLY if it cannot be guessed from the column name
# (e.g. "Time Stamp (ms)" is recognised as ms). Possible values: "ms" or "s".
TIME_UNIT = "ms"

# TIME offset added to the behavior timestamp to get the calcium one.
# Leave 0.0 if both files use the same clock.
# Example: if calcium starts 2 s after behavior, use 2000 (ms) or 2.0 (s).
OFFSET = 0.0

# ROW shift: video frame n corresponds to row (n + ROW_SHIFT) of the behavior file.
# Leave 0 if row 0 of the file is video frame 0.
# Example: if the file has 31 extra rows at the start, use 31.
ROW_SHIFT = 0

# Maximum size (pixels) of the displayed video; larger videos are scaled down.
MAX_W, MAX_H = 960, 600

# Names (without extension) of the files to find in the folder.
# NOTE: NAME_BEHAVIOR keeps its French spelling so that it matches your existing file names.
NAME_VIDEO = "video"
NAME_BEHAVIOR = "comportement"
NAME_CALCIUM = "calcium"

# Accepted extensions
VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv")
TABLE_EXT = (".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".txt")

# Experiment names recognised in the folder name or its parent folder name (case-insensitive).
# e.g. "Reversal_J2", "training 3" or ".../Shift/1". Add new ones here.
EXPERIENCES = ("reversal", "shift", "training")

# Results file, created in the experiment folder. The name is deliberately fixed
# (it does not contain the folder name) so that renaming the folder does not lose your marks.
RESULT_FILE = "nosepokes.xlsx"

# Information kept for each nose poke (see compute_mark).
# These keys are also the column names of the results file: they are kept in French
# so that existing results files and downstream analysis scripts keep working.
MARK_KEYS = ["i", "type", "frame_comportement", "t_comportement",
             "frame_calcium", "t_calcium", "ecart", "hors_plage"]

# The two nose poke TYPES ("" = unclassified, for marks from an older version).
REWARD, NO_REWARD = "reward", "no_reward"
KIND_LABELS = {REWARD: "reward", NO_REWARD: "no reward", "": "unclassified"}

# Displayed table headings. The keys (column identifiers, also used in files) stay in French.
HEADINGS = {"n": "#", "frame_video": "video frame", "type": "type",
            "frame_comportement": "behavior frame", "t_comportement": "behavior time",
            "frame_calcium": "calcium frame", "t_calcium": "calcium time",
            "ecart": "time diff", "hors_plage": "out of range"}

# Colours
GREEN, ORANGE, RED, BLUE = "#2e7d32", "#e65100", "#c62828", "#1565c0"
PALE_GREEN, PALE_RED = "#c8e6c9", "#ffcdd2"
PALE_GREEN_HOVER, PALE_RED_HOVER = "#a5d6a7", "#ef9a9a"


# =============================================================================
# 3) HELPER FUNCTIONS
# =============================================================================
def find_file(folder, name, extensions):
    """Return the file in 'folder' whose name (without extension) is 'name' and whose
    extension is in 'extensions' (e.g. calcium.xlsx or Calcium.CSV), or None."""
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.stem.lower() == name and f.suffix.lower() in extensions:
            return f
    return None


def find_col(df, keyword):
    """Return the first column of 'df' whose name contains 'keyword'."""
    for c in df.columns:
        if keyword in str(c).lower():
            return c
    raise ValueError(f"No column containing '{keyword}' among {list(df.columns)}")


def detect_unit(column_name, default):
    """Guess the timestamp unit from the column name ("Time Stamp (ms)" -> "ms"),
    otherwise return 'default'."""
    name = str(column_name).lower().replace(" ", "")
    if "(ms)" in name or "[ms]" in name:
        return "ms"
    if "(s)" in name or "[s]" in name or "(sec)" in name:
        return "s"
    return default


def to_numeric_time(s):
    """Convert a timestamp column to numbers. Return (numbers, were_dates).
    Numbers are kept as is; dates/times are converted to seconds since 1970-01-01."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.to_numpy(float), False
    dt = pd.to_datetime(s)
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(float), True


def load_table(path):
    """Read an Excel/CSV file. Return (frame numbers, timestamps as numbers,
    timestamps as written in the file, unit)."""
    if Path(path).suffix.lower() in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep=None, engine="python")  # pandas guesses the separator
    else:
        df = pd.read_excel(path)
    fcol, tcol = find_col(df, "frame"), find_col(df, "time")
    df = df.dropna(subset=[fcol, tcol]).reset_index(drop=True)
    t, is_date = to_numeric_time(df[tcol])
    # Dates are converted to SECONDS; numbers keep the unit written in the header.
    unit = "s" if is_date else detect_unit(tcol, TIME_UNIT)
    return df[fcol].to_numpy(), t, df[tcol].to_numpy(), unit


def usable_range(n_video, n_rows, shift):
    """Range of usable video frames i0..i1 (inclusive): video frame i corresponds to
    row (i + shift) of the behavior file, which must exist."""
    i0 = max(0, -shift)
    i1 = min(n_video - 1, n_rows - 1 - shift)
    return i0, i1


def find_gaps(t):
    """Positions just after each timestamp jump (interval > 1.5 x the median),
    i.e. probable lost frames. Return None if the timestamps are not increasing."""
    d = np.diff(t)
    if len(d) == 0:
        return []
    med = np.median(d)
    if med <= 0:
        return None
    return (np.where(d > 1.5 * med)[0] + 1).tolist()


def gap_text(g):
    """Turn the result of find_gaps into readable text."""
    if g is None:
        return "timestamps not increasing ⚠"
    if not g:
        return "0"
    # k + 2 = Excel row number (+1 for the header row, +1 because Excel counts from 1)
    shown = ", ".join(str(k + 2) for k in g[:5])
    return f"{len(g)} (Excel rows ≈ {shown}{'…' if len(g) > 5 else ''})"


def clean_kind(x):
    """Normalise a type read from a results file: "reward", "no_reward" or "" (unclassified)."""
    k = str(x).strip().lower().replace(" ", "_")
    return k if k in (REWARD, NO_REWARD) else ""


def experiment_name(folder):
    """Return (display name, recognised).
    Name = parent folder name + folder name, unless the folder name already contains a keyword.
      .../Shift/1           -> ("Shift 1", True)
      .../Reversal_J2       -> ("Reversal_J2", True)
      .../Mouse3/1          -> ("Mouse3 1", False)   (no reversal / shift / training keyword)
    Never raises an error: an unexpected name blocks nothing."""
    name = folder.name or str(folder)  # empty if the root of a drive is chosen
    parent = folder.parent.name

    def keywords(text):
        return [e for e in EXPERIENCES if e in text.lower()]

    own, up = keywords(name), keywords(parent)
    if own:
        label = name
    elif parent:
        label = f"{parent} {name}"
    else:
        label = name
    known = len(own) == 1 or (not own and len(up) == 1)
    return label, known


def time_diff_tolerance(t):
    """'Normal' time difference between a timestamp and the nearest calcium frame:
    0.6 x the median calcium interval (0.5 + a margin for timestamp jitter).
    Also used as the margin to decide that a timestamp is out of range."""
    d = np.diff(t)
    if len(d) == 0:
        return 0.0
    med = float(np.median(d))
    return 0.6 * med if med > 0 else 0.0


# =============================================================================
# 4) THE "App" CLASS: EVERYTHING RELATED TO THE WINDOW
# =============================================================================
class App:
    COLS = ("n", "frame_video", "type", "frame_comportement", "t_comportement",
            "frame_calcium", "t_calcium", "ecart", "hors_plage")

    # -------------------------------------------------------------------------
    # 4.1) __init__
    # -------------------------------------------------------------------------
    def __init__(self, root):
        self.root = root
        root.title("Nose poke → calcium frame")

        # State: nothing is loaded at startup
        self.loaded = False
        self.cap = None
        self.video_path = None
        self.beh_file = None
        self.ca_file = None
        self.save_path = None
        self.save_ok = True
        self.fps = 30.0
        self.n_video = 0
        self.unit = TIME_UNIT
        self.tol = 0.0  # normal nose poke -> calcium frame time difference
        self.experiment = ""
        self.experiment_known = False
        self.exp_recovered = False  # name taken from the results file (folder renamed)
        self.details_visible = False
        self.offset = OFFSET
        self.shift = ROW_SHIFT
        self.i0, self.i1 = 0, 0  # first and last usable frames
        self.marks = []
        self.pos = -1  # displayed frame (-1 = none yet)
        self.playing = False

        self.pause_on_mark = tk.BooleanVar(value=True)
        self.offset_var = tk.StringVar(value=str(OFFSET))
        self.shift_var = tk.StringVar(value=str(ROW_SHIFT))
        self.offset_text = tk.StringVar(value=f"Time offset (OFFSET) in {TIME_UNIT}:")
        self.hint_text = tk.StringVar(value=f"(in {TIME_UNIT}, same as your timestamps; press Enter to apply)")

        self.build_ui()
        self.root.focus_set()
        # Let the window appear first, then offer to choose a folder
        self.root.after(200, self.open_folder)

    # -------------------------------------------------------------------------
    # 4.2) build_ui
    # -------------------------------------------------------------------------
    def build_ui(self):
        r = self.root

        # Row 0: top bar
        top = tk.Frame(r)
        top.grid(row=0, column=0, columnspan=2, pady=4, sticky="w")
        tk.Button(top, text="📁 Open a folder", takefocus=0,
                  command=self.open_folder).pack(side="left", padx=3)
        tk.Button(top, text="🔢 Recount frames (exact)", takefocus=0,
                  command=self.count_exact).pack(side="left", padx=3)
        self.exp_label = tk.Label(top, text="", font=("Segoe UI", 10, "bold"), fg=BLUE)
        self.exp_label.pack(side="left", padx=(10, 0))
        self.folder_label = tk.Label(top, text="No folder open", fg="gray")
        self.folder_label.pack(side="left", padx=10)

        # Row 1: settings (time OFFSET and row shift)
        params = tk.Frame(r)
        params.grid(row=1, column=0, columnspan=2, pady=2, sticky="w")
        tk.Label(params, textvariable=self.offset_text).pack(side="left", padx=(6, 2))
        e1 = tk.Entry(params, textvariable=self.offset_var, width=10)
        e1.pack(side="left")
        tk.Label(params, text=" Row shift:").pack(side="left", padx=(10, 2))
        e2 = tk.Entry(params, textvariable=self.shift_var, width=6)
        e2.pack(side="left")
        tk.Button(params, text="✔ Apply", takefocus=0,
                  command=self.apply_params).pack(side="left", padx=8)
        tk.Label(params, textvariable=self.hint_text, fg="gray").pack(side="left")
        for e in (e1, e2):
            e.bind("<Return>", self.apply_params)
            e.bind("<Escape>", lambda ev: r.focus_set())

        # Row 2: check panel (green = OK, blue = information, orange = to check)
        panel = tk.Frame(r)
        panel.grid(row=2, column=0, columnspan=2, sticky="w", padx=6)
        self.chk1 = tk.Label(panel, text="", justify="left", wraplength=900, font=("Segoe UI", 10))
        self.chk2 = tk.Label(panel, text="", justify="left", wraplength=900, font=("Segoe UI", 10))
        self.chk3 = tk.Label(panel, text="", justify="left", wraplength=900, font=("Segoe UI", 10))
        for lab in (self.chk1, self.chk2, self.chk3):
            lab.pack(anchor="w")
        self.details_btn = tk.Button(panel, text="▸ Technical details", takefocus=0, relief="flat",
                                     fg="gray", command=self.toggle_details)
        self.details_btn.pack(anchor="w")
        self.details = tk.Label(panel, text="", justify="left", wraplength=900,
                                font=("Consolas", 9), fg="gray")  # hidden until toggled

        # Row 3: video display
        self.canvas = tk.Label(r, text="No folder open.\nClick “📁 Open a folder”.", pady=60)
        self.canvas.grid(row=3, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", lambda e: r.focus_set())

        # Row 4: info text under the video
        self.info = tk.Label(r, font=("Consolas", 11))
        self.info.grid(row=4, column=0, columnspan=2, pady=4)

        # Row 5: slider
        self.slider = tk.Scale(r, from_=0, to=1, orient="horizontal", length=900,
                               showvalue=False, takefocus=0, command=self.on_slider)
        self.slider.grid(row=5, column=0, columnspan=2)
        self.slider.bind("<ButtonRelease-1>", lambda e: r.focus_set())

        # Row 6: action buttons
        bar = tk.Frame(r)
        bar.grid(row=6, column=0, columnspan=2, pady=4)
        tk.Button(bar, text="⏯ Play/Pause", takefocus=0, command=self.toggle).pack(side="left", padx=3)
        tk.Button(bar, text="No reward nose poke (N)", bg=PALE_RED, activebackground=PALE_RED_HOVER,
                  takefocus=0, command=lambda: self.mark(NO_REWARD)).pack(side="left", padx=3)
        tk.Button(bar, text="Reward nose poke (R)", bg=PALE_GREEN, activebackground=PALE_GREEN_HOVER,
                  takefocus=0, command=lambda: self.mark(REWARD)).pack(side="left", padx=3)
        tk.Button(bar, text="↶ Undo last", takefocus=0, command=self.undo).pack(side="left", padx=3)
        tk.Button(bar, text="↪ Move selection here", takefocus=0,
                  command=self.move_selected).pack(side="left", padx=3)
        tk.Button(bar, text="🗑 Delete selection", takefocus=0,
                  command=self.delete_selected).pack(side="left", padx=3)
        tk.Button(bar, text="💾 Export", takefocus=0, command=self.export).pack(side="left", padx=3)

        # Row 7: playback speed and auto-pause option
        bar2 = tk.Frame(r)
        bar2.grid(row=7, column=0, columnspan=2, pady=2)
        tk.Label(bar2, text="Speed:").pack(side="left", padx=(0, 2))
        self.speed = ttk.Combobox(bar2, values=["0.25", "0.5", "1", "2", "4"], width=5,
                                  state="readonly", takefocus=0)
        self.speed.set("1")
        self.speed.pack(side="left")
        self.speed.bind("<<ComboboxSelected>>", lambda e: r.focus_set())
        tk.Checkbutton(bar2, text="⏸ Auto-pause when marking (to adjust the frame)",
                       variable=self.pause_on_mark, takefocus=0).pack(side="left", padx=20)

        # Row 8: table of marked nose pokes
        self.tree = ttk.Treeview(r, columns=self.COLS, show="headings", height=8, takefocus=0)
        for c in self.COLS:
            self.tree.heading(c, text=HEADINGS[c])
            self.tree.column(c, width=120, anchor="center")
        self.tree.column("n", width=40)
        self.tree.column("type", width=80)
        self.tree.tag_configure(REWARD, background=PALE_GREEN)
        self.tree.tag_configure(NO_REWARD, background=PALE_RED)
        self.tree.grid(row=8, column=0, columnspan=2, pady=4, sticky="ew")
        self.tree.bind("<Double-1>", self.goto_selected)
        self.update_unit_labels()

        # Row 9: status messages
        self.status = tk.Label(r, text="", fg="gray", wraplength=900)
        self.status.grid(row=9, column=0, columnspan=2)

        # Row 10: save status (always visible)
        self.save_label = tk.Label(r, text="", fg="gray", wraplength=900)
        self.save_label.grid(row=10, column=0, columnspan=2, pady=(0, 4))

        # Keyboard shortcuts
        self.bind_key("<space>", self.toggle)
        for key in ("r", "R"):
            self.bind_key(key, lambda: self.mark(REWARD))
        for key in ("n", "N"):
            self.bind_key(key, lambda: self.mark(NO_REWARD))
        for key in ("m", "M"):
            self.bind_key(key, self.move_selected)
        self.bind_key("<Left>", lambda: self.step(-1))
        self.bind_key("<Right>", lambda: self.step(1))
        self.bind_key("<Down>", lambda: self.step(-10))
        self.bind_key("<Up>", lambda: self.step(10))
        self.bind_key("<BackSpace>", self.undo)
        self.bind_key("<Delete>", self.delete_selected)

    def bind_key(self, key, func):
        """Bind a key to a function, ignored while typing in a text box
        (otherwise Backspace would undo a nose poke)."""
        def handler(event):
            if isinstance(event.widget, tk.Entry):
                return
            func()
        self.root.bind(key, handler)

    def update_unit_labels(self):
        """Write the timestamp unit in the OFFSET label and the table headings."""
        u = self.unit
        self.offset_text.set(f"Time offset (OFFSET) in {u}:")
        self.hint_text.set(f"(in {u}, same as your timestamps; press Enter to apply)")
        for col in ("t_comportement", "t_calcium", "ecart"):
            self.tree.heading(col, text=f"{HEADINGS[col]} ({u})")

    def update_experiment_labels(self):
        """Show the experiment name at the top of the window and in the window title."""
        self.exp_label.config(text=f"🧪 {self.experiment}")
        self.root.title(f"Nose poke → calcium frame — {self.experiment}")

    # -------------------------------------------------------------------------
    # 4.3) Opening a folder and recognising the files
    # -------------------------------------------------------------------------
    def open_folder(self):
        """Ask for a folder, find the video and the two timestamp files, and load everything."""
        # Marks are saved automatically and reloaded, so we only ask for confirmation
        # if the last save failed.
        if self.marks and not self.save_ok and not messagebox.askyesno(
                "Save failed",
                "The last save failed: your latest marks may not have been saved.\n"
                "Open another folder anyway?"):
            return

        folder = filedialog.askdirectory(title="Choose the experiment folder")
        if not folder:
            self.root.focus_set()
            return
        folder = Path(folder)

        video = find_file(folder, NAME_VIDEO, VIDEO_EXT)
        beh_file = find_file(folder, NAME_BEHAVIOR, TABLE_EXT)
        ca_file = find_file(folder, NAME_CALCIUM, TABLE_EXT)

        missing = []
        if video is None:
            missing.append(f"• the video: “{NAME_VIDEO}” with extension {', '.join(VIDEO_EXT)}")
        if beh_file is None:
            missing.append(f"• the behavior timestamps: “{NAME_BEHAVIOR}” with extension {', '.join(TABLE_EXT)}")
        if ca_file is None:
            missing.append(f"• the calcium timestamps: “{NAME_CALCIUM}” with extension {', '.join(TABLE_EXT)}")
        if missing:
            present = sorted(f.name for f in folder.iterdir() if f.is_file())[:15]
            messagebox.showerror(
                "Files not found",
                f"In the folder:\n{folder}\nmissing:\n" + "\n".join(missing) +
                "\n\nFiles present in the folder:\n" + "\n".join(present) +
                "\n\nTip: check that the extensions are not doubled (e.g. video.mp4.mp4).")
            self.root.focus_set()
            return

        # Load into TEMPORARY variables first: the current state is only replaced once
        # everything has succeeded, so a read failure does not break the open folder.
        try:
            b_frames, b_t, b_raw, b_unit = load_table(beh_file)
            c_frames, c_t, c_raw, c_unit = load_table(ca_file)
            if len(b_t) == 0 or len(c_t) == 0:
                raise ValueError("one of the files has no usable row")
            # Mixing ms and seconds would silently give completely wrong matches.
            if b_unit != c_unit:
                raise ValueError(
                    f"the two files do not use the same time unit "
                    f"(behavior: {b_unit}, calcium: {c_unit}).\n"
                    "Convert one of them so that both use the same unit.")
        except Exception as e:
            messagebox.showerror("Read error", f"Could not read the timestamp files:\n{e}")
            self.root.focus_set()
            return

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            messagebox.showerror(
                "Unreadable video",
                f"Could not open:\n{video}\nIf the path contains accents or special characters, "
                "try moving the folder to a simple path (e.g. C:\\data).")
            self.root.focus_set()
            return

        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_video <= 0:  # some formats do not report it: count frame by frame
            n_video = self.count_frames(video)
        if n_video <= 0:
            cap.release()
            messagebox.showerror("Empty video", f"No readable frame in:\n{video}")
            self.root.focus_set()
            return

        # Everything succeeded: replace the previous state
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.playing = False
        self.video_path, self.beh_file, self.ca_file = video, beh_file, ca_file
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_video = n_video
        self.b_frames, self.b_t, self.b_raw = b_frames, b_t, b_raw
        self.c_frames, self.c_t, self.c_raw = c_frames, c_t, c_raw
        self.tol = time_diff_tolerance(c_t)
        self.unit = b_unit
        self.update_unit_labels()

        # Starting settings; if the default row shift is incompatible with this folder, reset it to 0
        self.offset, self.shift = OFFSET, ROW_SHIFT
        i0, i1 = usable_range(self.n_video, len(self.b_t), self.shift)
        if i1 < i0:
            self.shift = 0
        self.marks = []
        self.pos = -1
        self.save_path = folder / RESULT_FILE
        self.save_ok = True
        self.loaded = True
        self.folder_label.config(text=f"📂 {folder}", fg="black")

        # The folder name is only used for the experiment name: files are found by name
        # inside the folder and the results file has a fixed name.
        self.experiment, self.experiment_known = experiment_name(folder)
        self.exp_recovered = False

        # Reload the marks of a previous session, if a results file exists
        message = self.load_previous()
        self.update_experiment_labels()
        if self.exp_recovered:
            message += (f" ℹ Neither the folder name nor its parent folder name contains reversal / shift / training: "
                        f"“{self.experiment}” was taken from the results file.")
        elif not self.experiment_known:
            message += (f" ℹ Neither the folder name nor its parent folder name contains reversal / shift / training: "
                        f"“{self.experiment}” is used as is for the experiment name.")

        self.sync_param_fields()
        self.refresh_all()
        self.show(self.i0)
        self.status.config(text=message)
        self.root.focus_set()

    def load_previous(self):
        """Reload the marks (and settings) of a previous session if the results file exists.
        Return a message to display."""
        if not self.save_path.exists():
            self.save_label.config(text="No save yet in this folder.", fg="gray")
            return "New folder loaded. Results will be saved automatically in "\
                   f"{self.save_path.name}."
        try:
            # Sheet "nose_pokes"; files from an older version have no such sheet: read the first one
            try:
                df = pd.read_excel(self.save_path, sheet_name="nose_pokes")
            except ValueError:
                df = pd.read_excel(self.save_path)

            # Settings of the previous session (sheet "parametres"), if any
            params = {}
            try:
                p = pd.read_excel(self.save_path, sheet_name="parametres")
                params = dict(zip(p["parametre"], p["valeur"]))
            except ValueError:
                pass

            # Renamed folder: if its name no longer says the experiment, take the one saved in the file
            saved_exp = str(params.get("experience", "")).strip()
            if not self.experiment_known and any(e in saved_exp.lower() for e in EXPERIENCES):
                self.experiment, self.experiment_known = saved_exp, True
                self.exp_recovered = True

            # Restore OFFSET and row shift if still compatible with this folder.
            # The OFFSET is not restored if the unit differs from the previous session (factor 1000).
            try:
                same_unit = str(params.get("unite_temps", self.unit)) == self.unit
                offset = float(params.get("offset_temps", self.offset)) if same_unit else self.offset
                shift = int(params.get("decalage_lignes", self.shift))
                i0, i1 = usable_range(self.n_video, len(self.b_t), shift)
                if i1 >= i0:
                    self.offset, self.shift = offset, shift
            except (TypeError, ValueError):
                pass

            # Marked video frames with their type (reward / no_reward) if recorded.
            # Older files have no "type" column: marks are then "unclassified".
            has_type = "type" in df.columns
            idx, types = [], []
            if "frame_video" in df.columns:
                fv = pd.to_numeric(df["frame_video"], errors="coerce")
                for k in range(len(df)):
                    if pd.notna(fv.iloc[k]):
                        idx.append(int(fv.iloc[k]))
                        types.append(clean_kind(df["type"].iloc[k]) if has_type else "")
            else:
                # Older file: find the video frame from the behavior frame number
                for k, f in enumerate(df["frame_comportement"]):
                    rows = np.where(self.b_frames == f)[0]
                    if len(rows):
                        idx.append(int(rows[0]) - self.shift)
                        types.append(clean_kind(df["type"].iloc[k]) if has_type else "")

            # Keep only usable frames, without duplicates (first occurrence wins)
            i0, i1 = usable_range(self.n_video, len(self.b_t), self.shift)
            by_frame = {}
            for i, kd in zip(idx, types):
                if i0 <= i <= i1 and i not in by_frame:
                    by_frame[i] = kd
            dropped = len(idx) - len(by_frame)

            # Recompute the calcium frames with the current settings (the type is kept)
            self.marks = [self.compute_mark(i, kd) for i, kd in by_frame.items()]
        except Exception as e:
            # Existing but unreadable file: back it up before the next save overwrites it
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = self.save_path.with_name(f"nosepokes_UNREADABLE_{stamp}.xlsx")
            try:
                shutil.copy2(self.save_path, backup)
                extra = f"A backup copy was created: {backup.name}"
            except Exception:
                extra = "The backup copy failed: copy the file yourself before continuing!"
            messagebox.showwarning(
                "Unreadable previous results file",
                f"Could not re-read {self.save_path.name}:\n{e}\n\n{extra}")
            self.save_label.config(text="⚠ Previous file unreadable (backup copy made).", fg=ORANGE)
            return "Folder loaded, but the previous results file was unreadable."

        self.save_label.config(text=f"💾 Existing file reloaded ({self.save_path.name})", fg=GREEN)
        msg = f"Folder loaded: {len(self.marks)} nose poke(s) reloaded from {self.save_path.name}."
        if dropped:
            msg += f" ⚠ {dropped} ignored (outside the usable range)."
        unclassified = sum(1 for m in self.marks if not m["type"])
        if unclassified:
            msg += (f" ⚠ {unclassified} nose poke(s) without a type (older version): go to their frame "
                    "and click “Reward” or “No reward” to classify them.")
        return msg

    # -------------------------------------------------------------------------
    # 4.4) Settings, frame counting and check panel
    # -------------------------------------------------------------------------
    def sync_param_fields(self):
        """Show the values actually used in the two settings boxes."""
        self.offset_var.set(str(self.offset))
        self.shift_var.set(str(self.shift))

    def apply_params(self, _event=None):
        """'Apply' button (or Enter in a box): read the boxes and apply."""
        if not self.loaded:
            self.status.config(text="Open a folder first.")
            self.root.focus_set()
            return
        try:
            offset = float(self.offset_var.get().replace(",", "."))  # accepts a decimal comma
            shift = int(self.shift_var.get())
        except ValueError:
            messagebox.showerror("Invalid value",
                                 f"The time offset must be a number in {self.unit} (e.g. 0 or 2.5)\n"
                                 "and the row shift a whole number (e.g. 0 or 31).")
            self.sync_param_fields()
            self.root.focus_set()
            return
        if self.set_params(offset, shift):
            self.status.config(
                text=f"Settings applied: OFFSET = {self.offset} {self.unit}, shift = {self.shift} row(s). "
                     f"{len(self.marks)} nose poke(s) recalculated.")
        else:
            self.sync_param_fields()
        self.root.focus_set()

    def set_params(self, offset, shift):
        """Apply new settings and recompute all marks. Return True if accepted, False otherwise."""
        i0, i1 = usable_range(self.n_video, len(self.b_t), shift)
        if i1 < i0:
            messagebox.showerror("Shift not possible",
                                 "With this row shift, no video frame has a corresponding row.")
            return False
        # Refuse if an already-marked nose poke would leave the usable range
        outside = [m["i"] for m in self.marks if not (i0 <= m["i"] <= i1)]
        if outside:
            messagebox.showerror(
                "Shift not possible",
                f"{len(outside)} already-marked nose poke(s) (e.g. frame {outside[0]}) "
                "would fall outside the usable range with this shift.\n"
                "Delete them first, or choose another value.")
            return False
        self.offset, self.shift = offset, shift
        self.refresh_all()
        self.show(self.pos)
        if self.marks or self.save_path.exists():
            self.autosave()  # settings are also saved in the results file
        return True

    def refresh_all(self):
        """Recompute everything that depends on the frame count, the shift and the OFFSET:
        usable range, slider, marks, check panel and table."""
        self.i0, self.i1 = usable_range(self.n_video, len(self.b_t), self.shift)
        self.marks = [self.compute_mark(m["i"], m["type"]) for m in self.marks if self.i0 <= m["i"] <= self.i1]
        self.slider.config(from_=self.i0, to=self.i1)
        self.update_check()
        self.refresh_tree()

    def count_frames(self, path):
        """Count the frames of a video one by one (slow but exact)."""
        cap2 = cv2.VideoCapture(str(path))
        n = 0
        while cap2.grab():
            n += 1
            if n % 500 == 0:
                self.status.config(text=f"Counting... {n} frames")
                self.root.update_idletasks()
        cap2.release()
        return n

    def count_exact(self):
        """'Recount' button: count the frames one by one, because the number read at loading
        comes from the file metadata, which can be approximate depending on the format."""
        if not self.loaded:
            return
        self.playing = False
        n = self.count_frames(self.video_path)
        if n <= 0:
            messagebox.showerror("Cannot count", "No readable frame in the video.")
            return
        # If the new count makes the current row shift impossible, reset it to 0
        i0, i1 = usable_range(n, len(self.b_t), self.shift)
        if i1 < i0:
            self.shift = 0
            self.sync_param_fields()
            messagebox.showwarning("Shift reset to 0",
                                   "With the exact number of frames, the row shift was no longer possible.")
        announced = self.n_video
        self.n_video = n
        before = len(self.marks)
        self.refresh_all()
        if self.pos < self.i0 or self.pos > self.i1:
            self.show(self.i0)
        if n == announced:
            msg = f"Exact count finished: {n} frames ✅ (same as announced by the video file)."
        else:
            msg = f"Exact count finished: {n} frames ⚠ (the video file announced {announced}: the previous number was wrong)."
        if len(self.marks) < before:
            msg += f" ⚠ {before - len(self.marks)} out-of-range nose poke(s) removed."
        self.autosave()
        self.status.config(text=msg)
        self.root.focus_set()

    def toggle_details(self):
        """'Technical details' button: show or hide the detailed numbers of the panel."""
        self.details_visible = not self.details_visible
        if self.details_visible:
            self.details.pack(anchor="w", pady=(2, 0))
            self.details_btn.config(text="▾ Hide details")
        else:
            self.details.pack_forget()
            self.details_btn.config(text="▸ Technical details")
        self.root.focus_set()

    def update_check(self):
        """Fill the check panel: 3 short colour-coded lines (green = OK, blue = harmless
        information, orange = to check) and the detailed text hidden behind the details button."""
        n_vid = self.n_video
        n_beh, n_ca = len(self.b_t), len(self.c_t)
        D = self.shift
        u = self.unit
        nd = 0 if u == "ms" else 3  # decimals to display
        # After the shift, the last file row matches frame (n_beh - 1 - D).
        # diff > 0: extra rows at the end of the file; diff < 0: rows missing.
        diff = (n_beh - D) - n_vid

        # LINE 1: video frame count vs behavior file row count
        if diff == 0 and D >= 0:
            text1, color1 = "✅ Frames: the video and the behavior file are aligned", GREEN
            if D > 0:
                text1 += f" ({D} row(s) at the start of the file ignored)"
        else:
            parts = []
            if D < 0:
                parts.append(f"the first {-D} video frame(s) have no row in the file")
            if D > 0:
                parts.append(f"{D} row(s) at the start of the file ignored")
            if diff > 0:
                parts.append(f"the file has {diff} more row(s) than the video (ignored at the end)")
            if diff < 0:
                parts.append(f"the video has {-diff} more frame(s) than the file (no timestamp)")
            # Only harmless case: extra rows at the end of the file, start correctly aligned
            harmless = (D >= 0 and diff > 0)
            if harmless:
                text1, color1 = "ℹ Frames: " + "; ".join(parts) + " — harmless if the video was simply cut", BLUE
            else:
                text1, color1 = "⚠ Frames: " + "; ".join(parts), ORANGE
        self.chk1.config(text=text1, fg=color1)

        # LINE 2: timestamp jumps = probably lost frames during acquisition
        g_b, g_c = find_gaps(self.b_t), find_gaps(self.c_t)
        if g_b == [] and g_c == []:
            self.chk2.config(text="✅ No lost frames (regular timestamps)", fg=GREEN)
        else:
            self.chk2.config(
                text=f"⚠ Lost frames? Timestamp jumps — behavior: {gap_text(g_b)}"
                     f" | calcium: {gap_text(g_c)}", fg=ORANGE)

        # LINE 3: do the covered time periods overlap? (only the rows actually used)
        used = self.b_t[self.i0 + D: self.i1 + D + 1]
        ts = used + self.offset
        b0, b1 = ts.min(), ts.max()
        c0, c1 = self.c_t.min(), self.c_t.max()
        # Out of range only if beyond the calcium recording by MORE than the normal difference
        n_out = int(np.sum((ts < c0 - self.tol) | (ts > c1 + self.tol)))
        if n_out == 0:
            self.chk3.config(text="✅ Time: the behavior is covered by the calcium recording", fg=GREEN)
        else:
            self.chk3.config(
                text=f"⚠ Time: {n_out} behavior frame(s) are outside the calcium time range "
                     "(no reliable match for them)", fg=ORANGE)

        # DETAILS (hidden by default)
        p_b = float(np.median(np.diff(self.b_t))) if n_beh > 1 else float("nan")
        p_c = float(np.median(np.diff(self.c_t))) if n_ca > 1 else float("nan")
        self.details.config(text=(
            f"Frames: video = {n_vid} | behavior file = {n_beh} rows | row shift = {D}\n"
            f"Timestamp jumps: behavior = {gap_text(g_b)} | calcium = {gap_text(g_c)}\n"
            f"Behavior time [{b0:.{nd}f} ; {b1:.{nd}f}] {u} | calcium [{c0:.{nd}f} ; {c1:.{nd}f}] {u} ({n_ca} rows)\n"
            f"Typical interval: behavior ≈ {p_b:.4g} {u} | calcium ≈ {p_c:.4g} {u} → "
            f"normal time difference between a nose poke and its calcium frame: up to ≈ ± {self.tol:.4g} {u}"))

    # -------------------------------------------------------------------------
    # 4.5) Video handling: display, navigation, playback
    # -------------------------------------------------------------------------
    def show(self, i):
        """Display video frame number 'i'."""
        i = max(self.i0, min(self.i1, i))
        # Reading the NEXT frame is fast; jumping elsewhere is slower
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.pos = i

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        s = min(MAX_W / w, MAX_H / h, 1.0)  # reduction factor (never enlarge)
        if s != 1.0:
            frame = cv2.resize(frame, (int(w * s), int(h * s)))

        # Keep a reference in self.img, otherwise the image disappears from the screen
        self.img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.configure(image=self.img, text="", pady=0)

        r = i + self.shift  # behavior file row matching this frame
        self.info.config(
            text=f"video frame {i} | Excel frame {self.b_frames[r]} | timestamp {self.b_raw[r]} {self.unit}")
        self.slider.set(i)

    def on_slider(self, val):
        """Called when the slider is moved."""
        if not self.loaded:
            return
        i = int(float(val))
        # When the PROGRAM moves the slider (during playback), tkinter still calls this
        # function with the frame already displayed: ignore it, otherwise playback would pause.
        if i == self.pos:
            return
        self.playing = False
        self.show(i)

    def step(self, d):
        """Move forward (d > 0) or backward (d < 0) by 'd' frames."""
        if not self.loaded:
            return
        self.playing = False
        self.show(self.pos + d)

    def toggle(self):
        """Switch between play and pause."""
        if not self.loaded:
            return
        self.playing = not self.playing
        if self.playing:
            self.tick()

    def tick(self):
        """Display the next frame, then reschedule itself (playback loop)."""
        if not self.playing or self.pos + 1 > self.i1:
            self.playing = False
            return
        self.show(self.pos + 1)
        delay = int(1000 / self.fps / float(self.speed.get()))
        self.root.after(max(1, delay), self.tick)

    # -------------------------------------------------------------------------
    # 4.6) Core: marking a nose poke and finding the calcium frame
    # -------------------------------------------------------------------------
    def compute_mark(self, i, kind):
        """Build the full record of a nose poke from its video frame 'i' and its type 'kind'
        (REWARD, NO_REWARD, or "" if unclassified): behavior frame and timestamp, then the
        matching calcium frame and timestamp. Everything is derived from 'i', so all marks
        can be recomputed when the OFFSET or the row shift changes.

        'ecart' = t_calcium - t_comportement (positive = the calcium frame is AFTER the nose poke).
        Calcium is sampled more slowly than the behavior camera, so a nose poke almost always
        falls BETWEEN two calcium frames: the nearest one is taken, and 'ecart' is at most half
        a calcium period (≈ 0.6 period with the margin self.tol). This is the temporal resolution
        of calcium imaging, not an error. A much larger value signals a real problem (wrong OFFSET,
        lost frames, or nose poke outside the calcium range)."""
        r = i + self.shift

        # Behavior frame timestamp (+ time offset)
        t = self.b_t[r] + self.offset

        # Calcium frame with the nearest timestamp
        j = int(np.argmin(np.abs(self.c_t - t)))

        # Outside the calcium period? A overshoot smaller than the normal difference is tolerated.
        out_of_range = not (self.c_t.min() - self.tol <= t <= self.c_t.max() + self.tol)

        return dict(
            i=i,                                    # video frame number
            type=kind,                              # "reward" or "no_reward"
            frame_comportement=self.b_frames[r],    # behavior frame (in the Excel file)
            t_comportement=self.b_raw[r],           # behavior timestamp
            frame_calcium=self.c_frames[j],         # matching calcium frame  <-- THE RESULT
            t_calcium=self.c_raw[j],                # matching calcium timestamp
            ecart=self.c_t[j] - t,                  # time difference between the two
            hors_plage=bool(out_of_range))          # True if outside the calcium range

    def mark(self, kind):
        """Record a nose poke at the displayed frame. 'kind' = REWARD or NO_REWARD.
        If the frame is already marked with the OTHER type, the type is simply corrected."""
        if not self.loaded:
            return
        i = self.pos
        label = KIND_LABELS[kind]

        for k, old in enumerate(self.marks):
            if old["i"] == i:
                if old["type"] == kind:
                    self.status.config(text=f"Frame {i} is already marked as {label}.")
                    return
                before = KIND_LABELS[old["type"]]
                self.marks[k] = self.compute_mark(i, kind)
                self.refresh_tree()
                self.tree.selection_set(self.tree.get_children()[k])
                self.autosave()
                self.status.config(text=f"Nose poke #{k + 1} (video frame {i}): type changed {before} → {label}.")
                return

        m = self.compute_mark(i, kind)
        self.marks.append(m)
        self.refresh_tree()
        kids = self.tree.get_children()
        self.tree.see(kids[-1])
        self.autosave()

        # Auto-pause (if ticked) to leave time to adjust the frame
        if self.pause_on_mark.get():
            self.playing = False

        out = " ⚠ outside the calcium time range!" if m["hors_plage"] else ""
        verdict = "normal time difference" if abs(m["ecart"]) <= self.tol else "⚠ abnormal time difference"
        n_r = sum(1 for x in self.marks if x["type"] == REWARD)
        n_nr = sum(1 for x in self.marks if x["type"] == NO_REWARD)
        self.status.config(
            text=f"Nose poke #{len(self.marks)} ({label}): video frame {i} (behavior {m['frame_comportement']}) → "
                 f"calcium frame {m['frame_calcium']} ({verdict}: {m['ecart']:.4g} {self.unit}){out}"
                 f" | total: {n_r} reward, {n_nr} no reward")

    def move_selected(self):
        """Move the nose poke selected in the table to the displayed frame
        (double-click the row, go to the right frame, then press M)."""
        if not self.loaded:
            return
        sel = self.tree.selection()
        if not sel:
            self.status.config(text="First select a row in the table, move the video to the right frame, then try again.")
            return
        k = self.tree.index(sel[0])
        i = self.pos
        if any(m["i"] == i for n, m in enumerate(self.marks) if n != k):
            self.status.config(text=f"Frame {i} is already marked by another nose poke.")
            return
        old = self.marks[k]["i"]
        self.marks[k] = self.compute_mark(i, self.marks[k]["type"])  # the type is kept
        self.refresh_tree()
        self.tree.selection_set(self.tree.get_children()[k])
        self.autosave()
        self.status.config(text=f"Nose poke #{k + 1} moved: video frame {old} → {i}.")

    def undo(self):
        """Remove the last marked nose poke."""
        if self.marks:
            self.marks.pop()
            self.refresh_tree()
            self.autosave()

    def delete_selected(self):
        """Delete the rows selected in the table."""
        sel = [self.tree.index(s) for s in self.tree.selection()]
        # Delete from the end so that the remaining indices do not shift
        for k in sorted(sel, reverse=True):
            self.marks.pop(k)
        self.refresh_tree()
        self.autosave()

    def goto_selected(self, _):
        """Double-click on a table row: jump to the corresponding frame."""
        sel = self.tree.selection()
        if sel:
            self.playing = False
            self.show(self.marks[self.tree.index(sel[0])]["i"])

    def refresh_tree(self):
        """Clear and redraw the whole table from self.marks."""
        self.tree.delete(*self.tree.get_children())
        for k, m in enumerate(self.marks, 1):
            self.tree.insert("", "end", tags=(m["type"],), values=(
                k, m["i"], KIND_LABELS[m["type"]], m["frame_comportement"], m["t_comportement"],
                m["frame_calcium"], m["t_calcium"], f"{m['ecart']:.4g}",
                "yes ⚠" if m["hors_plage"] else "no"))
        self.root.focus_set()

    # -------------------------------------------------------------------------
    # 4.7) Saving the results
    # -------------------------------------------------------------------------
    def dataframe(self):
        """Turn the list of marks into a pandas table ready to be saved."""
        u = self.unit
        # Time columns receive the unit (e.g. t_comportement -> t_comportement_ms)
        df = pd.DataFrame(self.marks, columns=MARK_KEYS).rename(columns={
            "i": "frame_video",
            "t_comportement": f"t_comportement_{u}",
            "t_calcium": f"t_calcium_{u}",
            "ecart": f"ecart_{u}"})
        df.insert(0, "nose_poke", range(1, len(df) + 1))
        df.insert(1, "experience", self.experiment)
        return df

    def params_dataframe(self):
        """'parameter / value' table: everything needed to know HOW the results were obtained."""
        rows = [
            ("experience", self.experiment),
            ("dossier", str(self.save_path.parent)),
            ("video", self.video_path.name),
            ("fichier_comportement", self.beh_file.name),
            ("fichier_calcium", self.ca_file.name),
            ("unite_temps", self.unit),
            ("offset_temps", self.offset),
            ("decalage_lignes", self.shift),
            ("frames_video", self.n_video),
            ("lignes_comportement", len(self.b_t)),
            ("lignes_calcium", len(self.c_t)),
            ("nombre_nose_pokes", len(self.marks)),
            ("nombre_reward", sum(1 for m in self.marks if m["type"] == REWARD)),
            ("nombre_no_reward", sum(1 for m in self.marks if m["type"] == NO_REWARD)),
            ("date_sauvegarde", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        return pd.DataFrame(rows, columns=["parametre", "valeur"])

    def write_results(self, path):
        """Write the results to 'path' (2-sheet Excel, or CSV if the name ends with .csv)."""
        path = Path(path)
        df = self.dataframe()
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
            return
        # Write to a TEMPORARY file first, then swap it in, so the old file is never half-written
        tmp = path.with_name(path.stem + "_tmp.xlsx")
        try:
            with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="nose_pokes", index=False)
                self.params_dataframe().to_excel(writer, sheet_name="parametres", index=False)
            os.replace(tmp, path)  # fails (PermissionError) if the file is open in Excel
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            raise

    def autosave(self):
        """Automatic save, with its status always shown at the bottom of the window."""
        if self.save_path is None:
            return True
        try:
            self.write_results(self.save_path)
            self.save_ok = True
            self.save_label.config(
                text=f"💾 Saved at {datetime.now():%H:%M:%S} ({self.save_path.name})", fg=GREEN)
        except Exception as e:  # typically: the file is open and locked in Excel
            self.save_ok = False
            self.save_label.config(
                text=f"⚠ NOT SAVED ({e}). Close the file if it is open in Excel: "
                     "saving will be retried at the next change.", fg=RED)
        return self.save_ok

    def export(self):
        """Export button: save the results wherever you want, as Excel or CSV."""
        if not self.loaded or not self.marks:
            self.status.config(text="Nothing to export yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", initialfile=f"nosepokes_{self.experiment.replace(' ', '_')}.xlsx",
            initialdir=self.save_path.parent,
            filetypes=[("Excel", ".xlsx"), ("CSV", ".csv")])
        if path:
            try:
                self.write_results(path)
                self.status.config(text=f"Exported: {path}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))
        self.root.focus_set()


# =============================================================================
# 5) ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
