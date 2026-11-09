"""
Animal Detection & Thermal Vision System
---------------------------------------
A desktop presentation app and headless CLI for animal detection:
- Supports Image & Video uploads.
- Converts video feeds to Thermal Infrared (IR) heatmaps (Inferno, Jet, Turbo, Bone, Hot).
- Detects animals across video frames and images with spatial IoU deduplication.
- Displays multi-animal count summary (e.g. 3 Animals: 2 Dogs, 1 Cow).
- Triggers non-blocking Voice Alert warnings ("Warning! 3 Animals are approaching!").

Run desktop UI:
    python final.py

Run image detection in terminal:
    python final.py --image animal_detector/samples/test_00005_out.png --headless

Run video thermal detection in terminal:
    python final.py --video path/to/sample_video.mp4 --headless --thermal-palette inferno
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

ROOT_DIR = Path(__file__).resolve().parent
DETECTOR_SRC_DIR = ROOT_DIR / "animal_detector" / "src"
if str(DETECTOR_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_SRC_DIR))

from inference import detect_animal, detect_animal_image, format_animal_counts  # noqa: E402
from video_processor import process_video, convert_frame_to_thermal, draw_thermal_overlay, speak_animal_warning, COLORMAPS  # noqa: E402


PALETTE = {
    "background": "#F8F8FF",
    "surface": "#FFFFFF",
    "ink": "#172033",
    "muted": "#657086",
    "subtle": "#8B94A5",
    "primary": "#1560BD",
    "primary_dark": "#104F9F",
    "primary_soft": "#EAF2FF",
    "success": "#217346",
    "success_soft": "#F2FBF5",
    "alert": "#D71868",
    "alert_soft": "#FFF7FB",
    "border": "#DFE2EB",
    "border_soft": "#EDF0F5",
    "preview": "#0B0E14",
    "error": "#A83232",
    "error_soft": "#FFF2F2",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
PREVIEW_SIZE = (680, 500)


class AnimalDetectionApp:
    """Tkinter UI presentation layer for Image & Thermal Video Animal Detection."""

    def __init__(self, root: tk.Tk, initial_file: str | None = None) -> None:
        self.root = root
        self.root.title("Thermal Vision & Animal Detection System")
        self.root.geometry("1240x840")
        self.root.minsize(980, 680)
        self.root.configure(bg=PALETTE["background"])

        self.selected_path: Path | None = None
        self.is_video: bool = False
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.output_path: Path | None = None
        self.is_processing_video: bool = False

        self.file_name_var = tk.StringVar(value="No file selected")
        self.file_status_var = tk.StringVar(value="Choose an image or video file.")
        self.status_var = tk.StringVar(value="Ready for file upload")
        self.result_title_var = tk.StringVar(value="No detection run yet")
        self.result_detail_var = tk.StringVar(value="Upload an image or video to begin detection.")
        self.count_var = tk.StringVar(value="—")
        self.animal_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.box_var = tk.StringVar(value="—")
        self.thermal_palette_var = tk.StringVar(value="inferno")
        self.voice_alert_var = tk.BooleanVar(value=True)

        self._configure_styles()
        self._build_layout()

        if initial_file:
            self.select_file(initial_file)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=PALETTE["background"])
        style.configure("Card.TFrame", background=PALETTE["surface"])
        style.configure(
            "Primary.TButton",
            background=PALETTE["primary"],
            foreground=PALETTE["surface"],
            borderwidth=0,
            focusthickness=0,
            padding=(14, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Primary.TButton", background=[("active", PALETTE["primary_dark"])])
        style.configure(
            "Secondary.TButton",
            background=PALETTE["surface"],
            foreground=PALETTE["primary"],
            bordercolor=PALETTE["border"],
            padding=(12, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Secondary.TButton", background=[("active", PALETTE["primary_soft"])])

    @staticmethod
    def _card(parent: tk.Widget, **kwargs: Any) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=PALETTE["surface"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            bd=0,
            **kwargs,
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # Header
        header = self._card(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)

        title_group = tk.Frame(header, bg=PALETTE["surface"])
        title_group.grid(row=0, column=0, sticky="w", padx=20, pady=14)
        tk.Label(
            title_group,
            text="Thermal Vision & Animal Detection System",
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_group,
            text="Multi-Animal Counting & Thermal Heatmap Transformation · Real-Time Voice Warnings",
            bg=PALETTE["surface"],
            fg=PALETTE["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        status_pill = tk.Label(
            header,
            text="● THERMAL VISION READY",
            bg=PALETTE["success_soft"],
            fg=PALETTE["success"],
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
        )
        status_pill.grid(row=0, column=1, sticky="e", padx=20, pady=14)

        # Main Workspace
        workspace = tk.Frame(outer, bg=PALETTE["background"])
        workspace.grid(row=1, column=0, sticky="nsew")
        workspace.columnconfigure(0, minsize=340, weight=0)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(0, weight=1)

        self._build_sidebar(workspace)
        self._build_result_panel(workspace)

        # Footer
        footer = tk.Label(
            outer,
            text="Supported Formats: Video (MP4, AVI, MOV, MKV, WEBM) & Image (PNG, JPG, WEBP) | Multi-Animal Spatial IoU NMS & Voice Announcements",
            bg=PALETTE["background"],
            fg=PALETTE["subtle"],
            font=("Segoe UI", 8),
            anchor="w",
        )
        footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))

    def _build_sidebar(self, parent: tk.Widget) -> None:
        sidebar = tk.Frame(parent, bg=PALETTE["background"], width=340)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        # Upload Card
        upload_card = self._card(sidebar)
        upload_card.pack(fill="x", pady=(0, 14))

        header = tk.Frame(upload_card, bg=PALETTE["surface"])
        header.pack(fill="x", padx=16, pady=(16, 10))
        tk.Label(
            header,
            text="STEP 01",
            bg=PALETTE["surface"],
            fg=PALETTE["subtle"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Upload Image or Video",
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(2, 0))

        drop_zone = tk.Frame(
            upload_card,
            bg="#FBFCFF",
            highlightbackground="#B7C0D1",
            highlightthickness=2,
            cursor="hand2",
        )
        drop_zone.pack(fill="x", padx=16, pady=(0, 10))
        drop_zone.bind("<Button-1>", lambda _event: self.choose_file())

        tk.Label(
            drop_zone,
            text="📹 / 🖼️",
            bg=PALETTE["primary_soft"],
            fg=PALETTE["primary"],
            font=("Segoe UI", 18),
            width=3,
            pady=2,
        ).pack(pady=(14, 6))
        prompt = tk.Label(
            drop_zone,
            text="Click to choose Video or Image",
            bg="#FBFCFF",
            fg="#27324A",
            font=("Segoe UI", 9, "bold"),
        )
        prompt.pack()
        prompt.bind("<Button-1>", lambda _event: self.choose_file())
        browse = tk.Label(
            drop_zone,
            text="MP4, AVI, MOV, PNG, JPG, WEBP",
            bg="#FBFCFF",
            fg=PALETTE["subtle"],
            font=("Segoe UI", 8),
        )
        browse.pack(pady=(3, 14))
        browse.bind("<Button-1>", lambda _event: self.choose_file())

        self.choose_button = ttk.Button(
            upload_card,
            text="Browse files...",
            command=self.choose_file,
            style="Secondary.TButton",
        )
        self.choose_button.pack(fill="x", padx=16, pady=(0, 10))

        # Selected File Info
        file_row = tk.Frame(upload_card, bg=PALETTE["surface"])
        file_row.pack(fill="x", padx=16, pady=(0, 4))
        tk.Label(
            file_row,
            text="SELECTED FILE",
            bg=PALETTE["surface"],
            fg=PALETTE["subtle"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            file_row,
            textvariable=self.file_name_var,
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))
        tk.Label(
            file_row,
            textvariable=self.file_status_var,
            bg=PALETTE["surface"],
            fg=PALETTE["muted"],
            font=("Segoe UI", 8),
            anchor="w",
            wraplength=280,
            justify="left",
        ).pack(fill="x", pady=(2, 0))

        # Options Card (Thermal Colormap & Voice Warning)
        options_card = self._card(sidebar)
        options_card.pack(fill="x", pady=(0, 14))
        
        tk.Label(
            options_card,
            text="Thermal & Voice Settings",
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 8))

        # Thermal Colormap Dropdown
        cmap_row = tk.Frame(options_card, bg=PALETTE["surface"])
        cmap_row.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(
            cmap_row,
            text="Thermal Colormap:",
            bg=PALETTE["surface"],
            fg=PALETTE["muted"],
            font=("Segoe UI", 9),
        ).pack(side="left")
        cmap_menu = ttk.Combobox(
            cmap_row,
            textvariable=self.thermal_palette_var,
            values=["inferno", "jet", "turbo", "bone", "hot", "magma"],
            state="readonly",
            width=12,
        )
        cmap_menu.pack(side="right")
        cmap_menu.bind("<<ComboboxSelected>>", lambda _e: self._on_colormap_changed())

        # Voice Alert Checkbox
        voice_chk = tk.Checkbutton(
            options_card,
            text="🔊 Enable Voice Warning Alert",
            variable=self.voice_alert_var,
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            activebackground=PALETTE["surface"],
            font=("Segoe UI", 9, "bold"),
        )
        voice_chk.pack(anchor="w", padx=16, pady=(0, 14))

        # Action Buttons Row
        btn_frame = tk.Frame(upload_card, bg=PALETTE["surface"])
        btn_frame.pack(fill="x", padx=16, pady=(12, 16))

        self.run_button = ttk.Button(
            btn_frame,
            text="▶ Run Detection & Thermal Vision",
            command=self.run_detection,
            style="Primary.TButton",
            state="disabled",
        )
        self.run_button.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.stop_button = tk.Button(
            btn_frame,
            text="⏹ Stop Video",
            command=self.stop_detection,
            bg=PALETTE["error"],
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            state="disabled",
            padx=10,
            pady=6,
        )
        self.stop_button.pack(side="right", padx=(4, 0))

        # System Capability Info Card
        metrics_card = self._card(sidebar)
        metrics_card.pack(fill="x")
        tk.Label(
            metrics_card,
            text="System Capabilities",
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 4))

        caps = (
            ("Input Support", "Video & Static Images"),
            ("Animal Counting", "Multi-Species Tracker"),
            ("Thermal Colormaps", "Inferno / Jet / Turbo"),
            ("Voice Synthesizer", "Windows SAPI Speech"),
        )
        for index, (label, value) in enumerate(caps):
            row = tk.Frame(metrics_card, bg=PALETTE["surface"])
            row.pack(fill="x", padx=16, pady=(0, 1))
            if index:
                separator = tk.Frame(row, bg=PALETTE["border_soft"], height=1)
                separator.pack(fill="x", pady=(0, 8))
            body = tk.Frame(row, bg=PALETTE["surface"])
            body.pack(fill="x", pady=(0, 8))
            tk.Label(body, text=label, bg=PALETTE["surface"], fg=PALETTE["muted"], font=("Segoe UI", 8), anchor="w").pack(side="left")
            tk.Label(body, text=value, bg=PALETTE["surface"], fg=PALETTE["ink"], font=("Segoe UI", 8, "bold"), anchor="e").pack(side="right")

    def _build_result_panel(self, parent: tk.Widget) -> None:
        result_card = self._card(parent)
        result_card.grid(row=0, column=1, sticky="nsew")
        result_card.rowconfigure(2, weight=1)
        result_card.columnconfigure(0, weight=1)

        heading = tk.Frame(result_card, bg=PALETTE["surface"])
        heading.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 12))
        heading.columnconfigure(0, weight=1)
        tk.Label(
            heading,
            text="STEP 02",
            bg=PALETTE["surface"],
            fg=PALETTE["subtle"],
            font=("Segoe UI", 8, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            heading,
            text="Thermal Feed & Multi-Animal Counter Display",
            bg=PALETTE["surface"],
            fg=PALETTE["ink"],
            font=("Segoe UI", 13, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        tk.Label(
            heading,
            textvariable=self.status_var,
            bg=PALETTE["surface"],
            fg=PALETTE["subtle"],
            font=("Segoe UI", 9),
        ).grid(row=1, column=1, sticky="e", pady=(2, 0))

        # Alert Box Banner
        self.alert_frame = tk.Frame(result_card, bg=PALETTE["primary_soft"], highlightthickness=1, highlightbackground="#C8DBF5")
        self.alert_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))
        self.alert_icon = tk.Label(self.alert_frame, text="ⓘ", bg=PALETTE["primary_soft"], fg=PALETTE["primary"], font=("Segoe UI", 15, "bold"))
        self.alert_icon.pack(side="left", padx=(12, 8), pady=10)
        alert_copy = tk.Frame(self.alert_frame, bg=PALETTE["primary_soft"])
        alert_copy.pack(side="left", fill="x", expand=True, pady=8)
        self.alert_title = tk.Label(alert_copy, textvariable=self.result_title_var, bg=PALETTE["primary_soft"], fg=PALETTE["primary"], font=("Segoe UI", 9, "bold"), anchor="w")
        self.alert_title.pack(anchor="w")
        self.alert_detail = tk.Label(alert_copy, textvariable=self.result_detail_var, bg=PALETTE["primary_soft"], fg=PALETTE["muted"], font=("Segoe UI", 8), anchor="w")
        self.alert_detail.pack(anchor="w", pady=(1, 0))

        # Preview Frame (Canvas / Display)
        preview_frame = tk.Frame(result_card, bg=PALETTE["preview"], highlightbackground=PALETTE["border"], highlightthickness=1)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=20)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            preview_frame,
            text="Thermal preview and detected objects will display here",
            bg=PALETTE["preview"],
            fg="#7A8B9E",
            font=("Segoe UI", 10),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Details Metrics Row (4 Metrics)
        details = tk.Frame(result_card, bg=PALETTE["surface"])
        details.grid(row=3, column=0, sticky="ew", padx=20, pady=(14, 18))
        for column in range(4):
            details.columnconfigure(column, weight=1)
        self._detail_cell(details, 0, "ANIMAL COUNT", self.count_var)
        self._detail_cell(details, 1, "SPECIES DETECTED", self.animal_var)
        self._detail_cell(details, 2, "CONFIDENCE LEVEL", self.confidence_var)
        self._detail_cell(details, 3, "BOUNDING BOX", self.box_var)

    def _detail_cell(self, parent: tk.Widget, column: int, label: str, variable: tk.StringVar) -> None:
        cell = tk.Frame(parent, bg=PALETTE["surface"], highlightbackground=PALETTE["border_soft"], highlightthickness=1)
        cell.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
        tk.Label(cell, text=label, bg=PALETTE["surface"], fg=PALETTE["subtle"], font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(cell, textvariable=variable, bg=PALETTE["surface"], fg=PALETTE["ink"], font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 8))

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an Image or Video",
            filetypes=[
                ("Supported files", "*.png *.jpg *.jpeg *.webp *.mp4 *.avi *.mov *.mkv *.webm"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Image files", "*.png *.jpg *.jpeg *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.select_file(path)

    def select_file(self, path: str) -> None:
        # Stop any active video processing thread from previous run
        self.is_processing_video = False

        file_path = Path(path).expanduser().resolve()
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            self._show_error("Unsupported format", f"Extension {ext} is not supported. Choose a video or image.")
            return

        self.selected_path = file_path
        self.is_video = ext in VIDEO_EXTENSIONS
        self.output_path = None
        self.file_name_var.set(file_path.name)

        if self.is_video:
            self.file_status_var.set("Video selected · Ready to convert to Thermal Vision & count animals")
            self.status_var.set("Video ready")
            self.result_title_var.set("Video loaded for Thermal Vision & Counting")
            self.result_detail_var.set("Press 'Run Detection & Thermal Vision' to stream video and count approaching animals.")
            # Load first video frame preview safely
            try:
                cap = cv2.VideoCapture(str(file_path))
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        thermal_preview = convert_frame_to_thermal(frame, colormap_name=self.thermal_palette_var.get())
                        rgb_preview = cv2.cvtColor(thermal_preview, cv2.COLOR_BGR2RGB)
                        self.current_image = Image.fromarray(rgb_preview)
                        self._display_image(self.current_image)
            except Exception as err:
                print(f"[Preview Error] {err}")
        else:
            self.file_status_var.set("Image selected · Ready to run detection")
            self.status_var.set("Image ready")
            self.result_title_var.set("Image ready for thermal detection")
            self.result_detail_var.set("Press 'Run Detection' to count and detect animals.")
            try:
                with Image.open(file_path) as img:
                    self.current_image = img.convert("RGB")
                    self._display_image(self.current_image)
            except Exception as err:
                self._show_error("Could not open image", str(err))
                return

        self.count_var.set("—")
        self.animal_var.set("—")
        self.confidence_var.set("—")
        self.box_var.set("—")
        self._set_alert("info")
        self.run_button.configure(state="normal")

    def _on_colormap_changed(self) -> None:
        if self.selected_path and self.is_video and not self.is_processing_video:
            try:
                cap = cv2.VideoCapture(str(self.selected_path))
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        thermal_preview = convert_frame_to_thermal(frame, colormap_name=self.thermal_palette_var.get())
                        rgb_preview = cv2.cvtColor(thermal_preview, cv2.COLOR_BGR2RGB)
                        self.current_image = Image.fromarray(rgb_preview)
                        self._display_image(self.current_image)
            except Exception as err:
                print(f"[Colormap Update Error] {err}")

    def stop_detection(self) -> None:
        """Immediately interrupts active video processing and resets button state."""
        self.is_processing_video = False
        self.status_var.set("Video Detection Stopped")
        self.file_status_var.set("Video detection stopped by user.")
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.result_title_var.set("Detection Cancelled")
        self.result_detail_var.set("Video processing was stopped by user.")
        self._set_alert("info")

    def run_detection(self) -> None:
        if self.selected_path is None:
            self._show_error("Choose a file first", "Please select an image or video file before running detection.")
            return

        self.run_button.configure(state="disabled")

        if self.is_video:
            self.stop_button.configure(state="normal")
            self._run_video_detection()
        else:
            self._run_image_detection()

    def _run_image_detection(self) -> None:
        self.status_var.set("Running detection…")
        self.file_status_var.set("Processing selected image…")
        self.root.update_idletasks()

        output_path = Path(tempfile.gettempdir()) / f"animal_detector_{uuid.uuid4().hex}.png"
        try:
            result = detect_animal(str(self.selected_path), out_path=str(output_path))
            self.output_path = output_path
            
            # Convert result image to thermal preview if animal detected
            with Image.open(output_path) as annotated:
                self.current_image = annotated.convert("RGB")
            
            # Display image & render counts
            self._display_image(self.current_image)
            self._render_result(result)

            # Trigger voice alert if animal detected
            if result.get("animal_detected") and self.voice_alert_var.get():
                summary = result.get("count_summary") or str(result.get("class"))
                speak_animal_warning(summary)

        except Exception as error:
            self.status_var.set("Inference failed")
            self.file_status_var.set("Error running image detection.")
            self._show_error("Detection error", str(error))
        finally:
            self.run_button.configure(state="normal")

    def _run_video_detection(self) -> None:
        # Reset detection metrics and alert state for fresh video run
        self.is_processing_video = True
        self.count_var.set("—")
        self.animal_var.set("—")
        self.confidence_var.set("—")
        self.box_var.set("—")
        self.status_var.set("Processing Thermal Video…")
        self.file_status_var.set("Streaming video through thermal engine and counting animals…")
        self.result_title_var.set("Scanning Thermal Video...")
        self.result_detail_var.set("Searching frames for animal heat signatures.")
        self._set_alert("info")

        output_path = Path(tempfile.gettempdir()) / f"thermal_animal_video_{uuid.uuid4().hex}.mp4"
        self.output_path = output_path

        def _worker():
            try:
                def _on_frame(thermal_bgr: np.ndarray, info: dict[str, Any]):
                    if not self.root.winfo_exists() or not self.is_processing_video:
                        return False
                    rgb_frame = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    frame_idx = info.get("frame_index", 0)
                    total_f = info.get("total_frames", 1)
                    top_a = info.get("top_animal")
                    dets = info.get("detections", [])
                    counts = info.get("counts", {})
                    tot_count = info.get("total_count", 0)

                    def _ui_update():
                        if not self.is_processing_video:
                            return
                        self._display_image(pil_img)
                        self.status_var.set(f"Frame {frame_idx}/{total_f}")
                        if tot_count > 0 and top_a:
                            summary_txt = format_animal_counts(counts)
                            self.count_var.set(f"{tot_count} Animal(s)")
                            self.animal_var.set(", ".join(c.capitalize() for c in counts.keys()))
                            self.result_title_var.set(f"🚨 ALERT: {summary_txt.upper()} APPROACHING!")
                            self.result_detail_var.set(f"Thermal signature confirmed on frame {frame_idx}.")
                            self._set_alert("success")
                            if dets:
                                top_det = max(dets, key=lambda d: d.get("confidence", 0.0))
                                self.confidence_var.set(f"{top_det.get('confidence', 0.0) * 100:.1f}%")
                                self.box_var.set(self._format_box(top_det.get("box")))
                        else:
                            if self.animal_var.get() == "—":
                                self.result_title_var.set("Scanning Thermal Video...")
                                self.result_detail_var.set("Searching frames for animal heat signatures.")
                                self._set_alert("info")

                    self.root.after(0, _ui_update)

                res = process_video(
                    video_path=str(self.selected_path),
                    output_path=str(output_path),
                    colormap=self.thermal_palette_var.get(),
                    conf_threshold=0.40,
                    sample_stride=2,
                    enable_voice=self.voice_alert_var.get(),
                    frame_callback=_on_frame,
                )

                def _finished():
                    self.is_processing_video = False
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set("Video Processing Complete")
                    
                    detected = res.get("animal_detected")
                    if detected:
                        detected_list = res.get("detected_animals", [])
                        animals_str = ", ".join(str(a).capitalize() for a in detected_list)
                        self.animal_var.set(animals_str)
                        self.count_var.set(f"{len(detected_list)} Species Detected")
                        self.result_title_var.set(f"🚨 ALERT: {animals_str.upper()} DETECTED IN VIDEO!")
                        self.result_detail_var.set(f"Annotated thermal video saved to: {output_path.name}")
                        self._set_alert("success")
                    else:
                        self.count_var.set("0")
                        self.animal_var.set("None detected")
                        self.result_title_var.set("No animal thermal signatures detected")
                        self.result_detail_var.set(f"Processed {res.get('processed_frames')} frames successfully.")
                        self._set_alert("neutral")
                    
                    messagebox.showinfo("Thermal Video Finished", f"Video processing completed!\nOutput saved to:\n{output_path}")

                self.root.after(0, _finished)

            except Exception as err:
                err_msg = str(err)
                def _error_ui(msg=err_msg):
                    self.is_processing_video = False
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self._show_error("Video Processing Error", msg)

                self.root.after(0, _error_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def _render_result(self, result: dict[str, Any]) -> None:
        detected = bool(result.get("animal_detected"))
        confidence = float(result.get("confidence", 0.0))
        class_name = result.get("class")
        box = result.get("box")
        tot_count = result.get("total_count", 1 if detected else 0)
        count_summary = result.get("count_summary") or (str(class_name).capitalize() if class_name else "—")

        self.status_var.set("Detection complete")
        self.file_status_var.set("Annotated output generated successfully.")
        self.count_var.set(f"{tot_count} Animal(s)" if detected else "0")
        self.confidence_var.set(f"{confidence * 100:.1f}%")
        self.box_var.set(self._format_box(box))

        if detected:
            readable_class = str(class_name).capitalize()
            self.animal_var.set(readable_class)
            self.result_title_var.set(f"🚨 ALERT: {count_summary.upper()} DETECTED!")
            self.result_detail_var.set(f"Total count: {tot_count} animal(s) found with {confidence * 100:.1f}% max confidence.")
            self._set_alert("success")
        else:
            self.animal_var.set("None detected")
            self.result_title_var.set("No animals detected")
            self.result_detail_var.set(f"The model returned {confidence * 100:.1f}% confidence.")
            self._set_alert("neutral")

    def _display_image(self, image: Image.Image) -> None:
        preview = ImageOps.contain(image, PREVIEW_SIZE)
        self.current_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.current_photo, text="")

    def _set_alert(self, state: str) -> None:
        colors = {
            "info": (PALETTE["primary_soft"], PALETTE["primary"], "ⓘ"),
            "success": (PALETTE["alert_soft"], PALETTE["alert"], "🚨"),
            "neutral": ("#F4F6F9", PALETTE["muted"], "○"),
            "error": (PALETTE["error_soft"], PALETTE["error"], "×"),
        }
        background, foreground, icon = colors[state]
        self.alert_frame.configure(bg=background, highlightbackground=foreground)
        self.alert_icon.configure(bg=background, fg=foreground, text=icon)
        self.alert_title.configure(bg=background, fg=foreground)
        self.alert_detail.configure(bg=background, fg=PALETTE["muted"])

    def _show_error(self, title: str, message: str) -> None:
        self.status_var.set("Needs attention")
        self.result_title_var.set(title)
        self.result_detail_var.set(message)
        self._set_alert("error")
        messagebox.showerror(title, message, parent=self.root)

    @staticmethod
    def _format_box(box: Any) -> str:
        if not box:
            return "—"
        return "(" + ", ".join(f"{float(value):.1f}" for value in box) + ")"


def run_headless_image(image_path: str) -> int:
    """Run image detection in terminal without opening Tkinter UI."""
    image = Path(image_path).expanduser().resolve()
    if not image.exists():
        print(f"Error: image not found: {image}")
        return 1
    
    output = Path(tempfile.gettempdir()) / f"animal_detector_smoke_{uuid.uuid4().hex}.png"
    result = detect_animal(str(image), out_path=str(output))
    print(f"Animal detected: {result['animal_detected']}")
    print(f"Total count: {result.get('total_count', 0)}")
    print(f"Count summary: {result.get('count_summary', 'None')}")
    print(f"Class: {result['class'] or 'None'}")
    print(f"Confidence: {float(result['confidence']) * 100:.1f}%")
    print(f"Box: {result['box'] or 'None'}")
    print(f"Alert: {result['alert']}")
    print(f"Annotated output: {output}")
    return 0


def run_headless_video(video_path: str, palette: str = "inferno", no_voice: bool = False) -> int:
    """Run video thermal detection in terminal without opening Tkinter UI."""
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        print(f"Error: video not found: {video}")
        return 1

    output = Path(tempfile.gettempdir()) / f"thermal_video_smoke_{uuid.uuid4().hex}.mp4"
    print(f"Processing video: {video}")
    print(f"Thermal palette: {palette}")
    print("Running detection and multi-animal counting across video frames...")

    result = process_video(
        video_path=str(video),
        output_path=str(output),
        colormap=palette,
        conf_threshold=0.40,
        enable_voice=not no_voice,
    )

    print("\n--- Video Detection Summary ---")
    print(f"Processed Frames: {result['processed_frames']}/{result['total_frames']}")
    print(f"Animal Detected: {result['animal_detected']}")
    print(f"Detected Classes: {result['detected_animals']}")
    print(f"Top Class: {result['top_class'] or 'None'}")
    print(f"Max Confidence: {float(result['max_confidence']) * 100:.1f}%")
    print(f"Annotated Output Video: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermal Vision & Multi-Animal Counting System")
    parser.add_argument("--image", help="Image path for preselection or headless detection")
    parser.add_argument("--video", help="Video path for preselection or headless thermal detection")
    parser.add_argument("--headless", action="store_true", help="Run detection in command line without opening UI")
    parser.add_argument("--thermal-palette", default="inferno", choices=list(COLORMAPS.keys()), help="Thermal colormap palette")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice warning announcements")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.headless:
        if args.video:
            return run_headless_video(args.video, palette=args.thermal_palette, no_voice=args.no_voice)
        elif args.image:
            return run_headless_image(args.image)
        else:
            print("Error: --headless requires --image <path> or --video <path>")
            return 2

    root = tk.Tk()
    initial_file = args.video or args.image
    AnimalDetectionApp(root, initial_file=initial_file)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
