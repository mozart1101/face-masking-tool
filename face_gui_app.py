import os
import sys
import threading
import tkinter as tkinter
from tkinter import filedialog, messagebox, ttk
import cv2
import lap
import numpy as np
import torch
import torchvision
from moviepy.editor import VideoFileClip
from ultralytics import YOLO

try:
  from tkinterdnd2 import DND_FILES, TkinterDnD

  HAS_DND = True
except ImportError:
  HAS_DND = False


def resource_path(relative_path):
  """Get absolute path to resource, works for dev and for PyInstaller"""
  try:
    base_path = sys._MEIPASS
  except AttributeError:
    base_path = os.path.abspath(".")

  local_path = os.path.join(
      os.path.dirname(
          os.path.abspath(
              __file__ if "__file__" in locals() else sys.executable
          )
      ),
      relative_path,
  )
  if os.path.exists(local_path):
    return local_path

  return os.path.join(base_path, relative_path)


class FaceMaskApp:

  def __init__(self, root):
    self.root = root
    self.root.title("AI Reel Side-by-Side Reference Masker")
    self.root.geometry("1280x800")
    self.root.configure(bg="#0f0f0f")

    self.style = ttk.Style()
    self.style.theme_use("clam")

    self.bg_color = "#0f0f0f"
    self.card_color = "#1a1a1a"
    self.accent_color = "#ff007f"
    self.text_color = "#ffffff"

    self.style.configure(
        "TLabel",
        background=self.bg_color,
        foreground=self.text_color,
        font=("Segoe UI", 10),
    )
    self.style.configure(
        "Header.TLabel",
        background=self.bg_color,
        foreground=self.accent_color,
        font=("Segoe UI", 16, "bold"),
    )

    self.input_path = tkinter.StringVar()
    self.output_dir = tkinter.StringVar()
    self.mask_target = tkinter.StringVar(value="Whole Subject")
    self.mask_style = tkinter.StringVar(value="black")
    self.audio_option = tkinter.StringVar(value="Keep Audio")
    self.dilation_val = tkinter.IntVar(value=20)
    self.start_frame_val = tkinter.IntVar(value=0)

    self.cap = None
    self.total_frames = 0
    self.fps = 30
    self.current_frame_idx = 0
    self.is_playing = False
    self.selected_click_points = []
    self.current_frame_img = None
    self.orig_width = 640
    self.orig_height = 520

    self.create_widgets()

  def create_widgets(self):
    header = ttk.Label(
        self.root,
        text="AI Reel Side-by-Side Reference Masker",
        style="Header.TLabel",
    )
    header.pack(pady=10)

    main_container = tkinter.Frame(self.root, bg=self.bg_color)
    main_container.pack(fill="both", expand=True, padx=15, pady=5)

    left_frame = tkinter.Frame(
        main_container, bg=self.card_color, padx=15, pady=15
    )
    left_frame.pack(side="left", fill="y", padx=5, pady=5)

    right_frame = tkinter.Frame(
        main_container, bg=self.card_color, padx=15, pady=15
    )
    right_frame.pack(
        side="right", fill="both", expand=True, padx=5, pady=5
    )

    self.build_config_panel(left_frame)
    self.build_video_panel(right_frame)

    self.lbl_status = ttk.Label(
        self.root,
        text="Status: Ready. Load or drag a video to begin.",
        background=self.bg_color,
        font=("Segoe UI", 9, "italic"),
    )
    self.lbl_status.pack(pady=5)

    self.btn_run = tkinter.Button(
        self.root,
        text="🚀 Start Rendering Masked Video",
        command=self.start_processing_thread,
        bg=self.accent_color,
        fg="white",
        relief="flat",
        font=("Segoe UI", 11, "bold"),
        padx=20,
        pady=8,
    )
    self.btn_run.pack(pady=10)

  def build_config_panel(self, parent):
    lbl_cfg = ttk.Label(
        parent,
        text="⚙️ Masking Configuration",
        font=("Segoe UI", 12, "bold"),
        background=self.card_color,
        foreground=self.accent_color,
    )
    lbl_cfg.grid(row=0, column=0, columnspan=2, sticky="w", pady=10)

    ttk.Label(
        parent, text="Reference Video:", background=self.card_color
    ).grid(row=1, column=0, sticky="w", pady=8)

    self.ent_file = tkinter.Entry(
        parent,
        textvariable=self.input_path,
        width=23,
        bg="#2a2a2a",
        fg=self.text_color,
        insertbackground="white",
        relief="flat",
        font=("Segoe UI", 10),
    )
    self.ent_file.grid(row=1, column=1, padx=5, pady=8)

    if HAS_DND:
      self.ent_file.drop_target_register(DND_FILES)
      self.ent_file.dnd_bind("<<Drop>>", self.on_file_drop)

    tkinter.Button(
        parent,
        text="Browse",
        command=self.browse_file,
        bg=self.accent_color,
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=6,
        pady=2,
    ).grid(row=1, column=2, padx=5, pady=8)

    lbl_tip = ttk.Label(
        parent,
        text="(You can also drag & drop video file here)",
        background=self.card_color,
        font=("Segoe UI", 8, "italic"),
        foreground="#888888",
    )
    lbl_tip.grid(row=2, column=1, columnspan=2, sticky="w", padx=5)

    ttk.Label(parent, text="Save To Folder:", background=self.card_color).grid(
        row=3, column=0, sticky="w", pady=8
    )
    self.ent_out = tkinter.Entry(
        parent,
        textvariable=self.output_dir,
        width=23,
        bg="#2a2a2a",
        fg=self.text_color,
        insertbackground="white",
        relief="flat",
        font=("Segoe UI", 10),
    )
    self.ent_out.grid(row=3, column=1, padx=5, pady=8)
    tkinter.Button(
        parent,
        text="Folder",
        command=self.browse_output_folder,
        bg="#444444",
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=6,
        pady=2,
    ).grid(row=3, column=2, padx=5, pady=8)

    ttk.Label(parent, text="Mask Target:", background=self.card_color).grid(
        row=4, column=0, sticky="w", pady=8
    )
    ttk.Combobox(
        parent,
        textvariable=self.mask_target,
        values=["Whole Subject", "Face & Hair Only", "Body Only"],
        state="readonly",
        width=20,
    ).grid(row=4, column=1, columnspan=2, sticky="w", padx=5, pady=8)

    ttk.Label(parent, text="Mask Effect:", background=self.card_color).grid(
        row=5, column=0, sticky="w", pady=8
    )
    ttk.Combobox(
        parent,
        textvariable=self.mask_style,
        values=["black", "blur"],
        state="readonly",
        width=20,
    ).grid(row=5, column=1, columnspan=2, sticky="w", padx=5, pady=8)

    ttk.Label(parent, text="Audio Track:", background=self.card_color).grid(
        row=6, column=0, sticky="w", pady=8
    )
    ttk.Combobox(
        parent,
        textvariable=self.audio_option,
        values=["Keep Audio", "Mute (No Audio)"],
        state="readonly",
        width=20,
    ).grid(row=6, column=1, columnspan=2, sticky="w", padx=5, pady=8)

    ttk.Label(
        parent, text="Mask Expansion Buffer:", background=self.card_color
    ).grid(row=7, column=0, sticky="w", pady=8)
    tkinter.Scale(
        parent,
        from_=0,
        to=40,
        resolution=1,
        orient="horizontal",
        variable=self.dilation_val,
        bg=self.card_color,
        fg=self.text_color,
        highlightthickness=0,
        troughcolor="#2a2a2a",
    ).grid(row=7, column=1, columnspan=2, sticky="ew", padx=5, pady=8)

    ttk.Label(parent, text="Start Mask Frame:", background=self.card_color).grid(
        row=8, column=0, sticky="w", pady=8
    )
    self.lbl_start_frame_disp = ttk.Label(
        parent, textvariable=self.start_frame_val, background=self.card_color
    )
    self.lbl_start_frame_disp.grid(row=8, column=1, sticky="w", padx=5, pady=8)

  def build_video_panel(self, parent):
    lbl_vid_header = ttk.Label(
        parent,
        text="🎬 Video Reference & Subject Selector",
        font=("Segoe UI", 12, "bold"),
        background=self.card_color,
        foreground=self.accent_color,
    )
    lbl_vid_header.pack(anchor="w", pady=5)

    ttk.Label(
        parent,
        text=(
            "Scrub through video, set start frame, and CLICK subjects to"
            " mask:"
        ),
        background=self.card_color,
        font=("Segoe UI", 9),
    ).pack(anchor="w", pady=5)

    self.canvas = tkinter.Canvas(
        parent, width=640, height=520, bg="#000000", highlightthickness=0
    )
    self.canvas.pack(pady=10)
    self.canvas.bind("<Button-1>", self.on_canvas_click)

    if HAS_DND:
      self.canvas.drop_target_register(DND_FILES)
      self.canvas.dnd_bind("<<Drop>>", self.on_file_drop)

    self.seek_slider = ttk.Scale(
        parent, from_=0, to=100, orient="horizontal", command=self.on_seek_slider
    )
    self.seek_slider.pack(fill="x", padx=10, pady=5)

    ctrl_frame = tkinter.Frame(parent, bg=self.card_color)
    ctrl_frame.pack(pady=5)

    tkinter.Button(
        ctrl_frame,
        text="▶ Play/Pause",
        command=self.toggle_play_pause,
        bg="#333333",
        fg="white",
        relief="flat",
        padx=10,
        pady=4,
    ).pack(side="left", padx=5)

    tkinter.Button(
        ctrl_frame,
        text="📌 Set Current Frame as Start",
        command=self.set_start_frame,
        bg=self.accent_color,
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=10,
        pady=4,
    ).pack(side="left", padx=5)

    tkinter.Button(
        ctrl_frame,
        text="🔄 Clear Selections",
        command=self.clear_selections,
        bg="#444444",
        fg="white",
        relief="flat",
        padx=10,
        pady=4,
    ).pack(side="left", padx=5)

    self.lbl_video_info = ttk.Label(
        parent,
        text="No video loaded. Browse or drag & drop a file here.",
        background=self.card_color,
        font=("Segoe UI", 9),
    )
    self.lbl_video_info.pack(pady=5)

  def browse_file(self):
    filename = filedialog.askopenfilename(
        title="Select Reference Video",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")],
    )
    if filename:
      self.set_video_path(filename)

  def on_file_drop(self, event):
    file_path = event.data.strip("{}")
    if os.path.exists(file_path) and file_path.lower().endswith(
        (".mp4", ".mov", ".avi", ".mkv")
    ):
      self.set_video_path(file_path)
    else:
      messagebox.showerror(
          "Error", "Please drop a valid video file (.mp4, .mov, .avi, .mkv)"
      )

  def set_video_path(self, path):
    self.input_path.set(path)
    if not self.output_dir.get():
      self.output_dir.set(os.path.dirname(path))
    self.load_video(path)

  def browse_output_folder(self):
    folder = filedialog.askdirectory(title="Select Save Destination Folder")
    if folder:
      self.output_dir.set(folder)

  def load_video(self, path):
    if self.cap is not None:
      self.cap.release()

    self.cap = cv2.VideoCapture(path)
    if self.cap.isOpened():
      self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
      if self.fps == 0:
        self.fps = 30
      self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

      self.seek_slider.config(to=max(1, self.total_frames - 1))
      self.current_frame_idx = 0
      self.selected_click_points = []
      self.show_frame(0)

  def show_frame(self, frame_idx):
    if self.cap is None or not self.cap.isOpened():
      return
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = self.cap.read()
    if ret:
      self.current_frame_img = frame
      self.render_frame_with_aspect_ratio(frame)
      self.lbl_video_info.config(
          text=(
              f"Frame: {frame_idx}/{self.total_frames} | Targets Selected:"
              f" {len(self.selected_click_points)}"
          )
      )

  def render_frame_with_aspect_ratio(self, frame):
    canvas_w, canvas_h = 640, 520
    h, w, _ = frame.shape
    frame_copy = frame.copy()

    for pt in self.selected_click_points:
      cv2.circle(frame_copy, pt, 6, (255, 0, 127), -1)

    scale = min(canvas_w / w, canvas_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame_copy, (new_w, new_h))

    canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2
    canvas_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
        resized
    )

    self.render_scale = scale
    self.render_offset_x = x_offset
    self.render_offset_y = y_offset

    rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
    self.photo_img = tkinter.PhotoImage(
        data=cv2.imencode(".png", rgb)[1].tobytes()
    )
    self.canvas.create_image(0, 0, anchor="nw", image=self.photo_img)

  def on_seek_slider(self, val):
    if self.cap is not None and self.cap.isOpened():
      idx = int(float(val))
      self.current_frame_idx = idx
      self.show_frame(idx)

  def toggle_play_pause(self):
    self.is_playing = not self.is_playing
    if self.is_playing:
      threading.Thread(target=self.play_video_loop, daemon=True).start()

  def play_video_loop(self):
    while self.is_playing and self.cap is not None and self.cap.isOpened():
      if self.current_frame_idx >= self.total_frames - 1:
        self.current_frame_idx = 0
      else:
        self.current_frame_idx += 1

      self.root.after(
          0, lambda idx=self.current_frame_idx: self.show_frame(idx)
      )
      self.root.after(
          0, lambda idx=self.current_frame_idx: self.seek_slider.set(idx)
      )
      import time

      time.sleep(1 / self.fps)

  def on_canvas_click(self, event):
    if self.current_frame_img is not None:
      click_x = event.x - self.render_offset_x
      click_y = event.y - self.render_offset_y

      if 0 <= click_x < int(
          self.orig_width * self.render_scale
      ) and 0 <= click_y < int(self.orig_height * self.render_scale):
        orig_x = int(click_x / self.render_scale)
        orig_y = int(click_y / self.render_scale)
        self.selected_click_points.append((orig_x, orig_y))
        print(
            f"Selected subject at original resolution: ({orig_x}, {orig_y})"
        )
        self.show_frame(self.current_frame_idx)

  def set_start_frame(self):
    self.start_frame_val.set(self.current_frame_idx)
    messagebox.showinfo(
        "Start Frame Locked",
        f"Masking will start at frame {self.current_frame_idx}.",
    )

  def clear_selections(self):
    self.selected_click_points = []
    self.show_frame(self.current_frame_idx)

  def start_processing_thread(self):
    video_path = self.input_path.get()
    out_folder = self.output_dir.get()

    if not video_path or not os.path.exists(video_path):
      messagebox.showerror(
          "Error", "Please select a valid input reference video file first!"
      )
      return
    if not out_folder or not os.path.exists(out_folder):
      messagebox.showerror(
          "Error", "Please select a valid save destination folder!"
      )
      return

    self.btn_run.config(state="disabled", bg="#555555")
    self.lbl_status.config(text="Status: Processing masked video export...")

    threading.Thread(
        target=self.process_video, args=(video_path, out_folder), daemon=True
    ).start()

  def process_video(self, input_path, out_folder):
    self.is_playing = False
    if self.cap is not None:
      self.cap.release()

    try:
      cap = cv2.VideoCapture(input_path)
      fps = int(cap.get(cv2.CAP_PROP_FPS))
      if fps == 0:
        fps = 30
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

      base_name = os.path.splitext(os.path.basename(input_path))[0]
      temp_output_path = os.path.join(out_folder, f"temp_{base_name}_masked.mp4")
      final_output_path = os.path.join(out_folder, f"{base_name}_masked.mp4")

      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

      model_weights = resource_path("yolov8n-seg.pt")
      model = YOLO(model_weights)

      target = self.mask_target.get()
      m_style = self.mask_style.get()
      audio_choice = self.audio_option.get()
      dilation_amount = self.dilation_val.get()
      start_mask_frame = self.start_frame_val.get()
      frame_count = 0

      target_track_ids = set()
      last_valid_masks = {}

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_count += 1
        h, w, _ = frame.shape

        if frame_count >= start_mask_frame:
          results = model.track(
              frame,
              persist=True,
              verbose=False,
              classes=[0],
              conf=0.3,
              tracker="bytetrack.yaml",
          )
          active_masks_for_frame = []

          if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = (
                results[0].boxes.id.cpu().numpy()
                if results[0].boxes.id is not None
                else None
            )

            if frame_count == start_mask_frame and len(self.selected_click_points) > 0:
              for pt in self.selected_click_points:
                cx, cy = pt
                closest_idx = None
                min_dist = float("inf")
                for i, box in enumerate(boxes):
                  dist = (cx - (box[0] + box[2]) / 2) ** 2 + (
                      cy - (box[1] + box[3]) / 2
                  ) ** 2
                  if (
                      box[0] - 50 <= cx <= box[2] + 50
                      and box[1] - 50 <= cy <= box[3] + 50
                  ):
                    if dist < min_dist:
                      min_dist = dist
                      closest_idx = i

                if closest_idx is not None:
                  if track_ids is not None and track_ids[closest_idx] is not None:
                    target_track_ids.add(int(track_ids[closest_idx]))
                  else:
                    target_track_ids.add(closest_idx)

            if len(target_track_ids) == 0 and len(boxes) > 0:
              if track_ids is not None and track_ids[0] is not None:
                target_track_ids.add(int(track_ids[0]))
              else:
                target_track_ids.add(0)

            for i, mask in enumerate(masks):
              tid = (
                  int(track_ids[i])
                  if (track_ids is not None and track_ids[i] is not None)
                  else i
              )
              is_targeted = False

              if (tid in target_track_ids or i in target_track_ids) or (
                  len(self.selected_click_points) == 0 and i == 0
              ):
                is_targeted = True

              if is_targeted:
                resized_mask = cv2.resize(mask, (w, h))
                binary_mask = (resized_mask > 0.35).astype(np.uint8) * 255

                box = boxes[i]
                y1_box, y2_box = int(box[1]), int(box[3])
                box_height = y2_box - y1_box

                if target == "Face & Hair Only":
                  y_cutoff = y1_box + int(box_height * 0.42)
                  upper_mask = np.zeros_like(binary_mask)
                  upper_mask[:y_cutoff, :] = 255
                  binary_mask = cv2.bitwise_and(binary_mask, upper_mask)

                elif target == "Body Only":
                  y_cutoff = y1_box + int(box_height * 0.35)
                  lower_mask = np.zeros_like(binary_mask)
                  lower_mask[y_cutoff:, :] = 255
                  binary_mask = cv2.bitwise_and(binary_mask, lower_mask)

                active_masks_for_frame.append(binary_mask)
                last_valid_masks[tid] = binary_mask

          for b_mask in active_masks_for_frame:
            if dilation_amount > 0:
              kernel_size = dilation_amount * 2 + 1
              kernel = cv2.getStructuringElement(
                  cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
              )
              b_mask = cv2.dilate(b_mask, kernel, iterations=1)

            mask_idx = b_mask > 127
            if m_style == "black":
              frame[mask_idx] = [0, 0, 0]
            elif m_style == "blur":
              blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)
              frame[mask_idx] = blurred_frame[mask_idx]

        out.write(frame)

        if frame_count % 30 == 0:
          progress_text = f"Status: Processing frame {frame_count}/{total_frames}"
          self.root.after(
              0, lambda pt=progress_text: self.lbl_status.config(text=pt)
          )

      cap.release()
      out.release()

      self.root.after(
          0, lambda: self.lbl_status.config(text="Status: Muxing audio track...")
      )

      video_clip = VideoFileClip(temp_output_path)
      if audio_choice == "Keep Audio":
        source_clip = VideoFileClip(input_path)
        if source_clip.audio is not None:
          video_clip = video_clip.set_audio(source_clip.audio)

      video_clip.write_videofile(
          final_output_path,
          codec="libx264",
          audio_codec="aac" if audio_choice == "Keep Audio" else None,
          logger=None,
      )
      video_clip.close()
      if audio_choice == "Keep Audio":
        source_clip.close()

      if os.path.exists(temp_output_path):
        try:
          os.remove(temp_output_path)
        except OSError:
          pass

      self.root.after(0, lambda: self.on_process_success(final_output_path))

    except Exception as e:
      err_str = str(e)
      print(f"Processing Error: {err_str}")
      self.root.after(
          0,
          lambda: self.on_process_error(f"Error during export:\n{err_str}"),
      )

  def on_process_error(self, err_msg):
    messagebox.showerror("Error", err_msg)
    self.btn_run.config(state="normal", bg=self.accent_color)
    self.lbl_status.config(text="Status: Failed")

  def on_process_success(self, out_path):
    self.btn_run.config(state="normal", bg=self.accent_color)
    self.lbl_status.config(text="Status: Complete!")
    messagebox.showinfo(
        "Success", f"Masked video successfully saved to:\n{out_path}"
    )


if __name__ == "__main__":
  if HAS_DND:
    root = TkinterDnD.Tk()
  else:
    root = tkinter.Tk()
  app = FaceMaskApp(root)
  root.mainloop()