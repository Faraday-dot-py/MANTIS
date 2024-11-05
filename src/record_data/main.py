import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog  # Import filedialog
import cv2 
from PIL import Image, ImageTk
from scrollable import VerticalScrolledFrame
from src.main.tmapmaker import TmapMaker
import os
from datetime import datetime


width, height = 900, 600

print("Loading Camera")
cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera loaded")

class App:
    def __init__(self, master):
        self.tmapMaker = TmapMaker()
        self.do_annotations = False
        self.camera_enabled = False
        self.output_directory = None  # Store selected output directory
        self.master = master
        master.title("Image Classification App")

        

        # Camera view frame
        self.camera_frame = tk.Frame(master, width=frame_width, height=frame_height)
        self.camera_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.camera_view = tk.Label(self.camera_frame)
        self.camera_view.pack()

        # Class labels frame with scrollbar
        self.class_labels_frame = VerticalScrolledFrame(master, height=height, width=width)
        self.class_labels_frame.grid(row=0, column=2, rowspan=6, padx=10, pady=10)

        # Add class labels
        self.class_labels = []
        for i in range(20):
            label_frame = tk.Frame(self.class_labels_frame.scrollable_frame)
            label_frame.pack(pady=5)
            class_name_label = tk.Label(label_frame, text=f"Class Name {i+1}", font=("Arial", 12))
            class_name_label.grid(row=0, column=0)
            sample_count_label = tk.Label(label_frame, text="# of samples", font=("Arial", 12))
            sample_count_label.grid(row=0, column=1, padx=10)
            self.class_labels.append((class_name_label, sample_count_label))

        # Buttons frame
        self.buttons_frame = tk.Frame(master)
        self.buttons_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=10)

        # Directory selection button
        self.output_directory_button = ttk.Button(
            self.buttons_frame, 
            text="Select Output Directory", 
            command=self.select_output_directory
        )
        self.output_directory_button.pack(side=tk.LEFT, padx=5)

        self.enable_camera_button = ttk.Button(self.buttons_frame, text="Enable Camera", command=self.toggle_camera)
        self.enable_camera_button.pack(side=tk.LEFT, padx=5)

        self.annotate_button = ttk.Button(self.buttons_frame, text="Annotate", command=lambda: setattr(self, 'do_annotations', not self.do_annotations))
        self.annotate_button.pack(side=tk.LEFT, padx=5)

        self.exit_button = ttk.Button(self.buttons_frame, text="Exit", command=master.quit)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_bar = tk.Label(master, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=7, column=0, columnspan=3, sticky=tk.W + tk.E)

    def select_output_directory(self):
        # Open directory selection dialog
        self.output_directory = filedialog.askdirectory()
        if self.output_directory:
            self.update_status(f"Output directory selected: {self.output_directory}")
        else:
            self.update_status("No output directory selected")

    def start_recording(self):
        self.output_directory = os.path.join(self.output_directory, "MANTIS_Output_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        self.update_status(f"Created output directory")

    def save_sample(self):
        pass

    def create_new_class(self):
        pass

    def update_status(self, message):
        self.status_bar.config(text=f"Status: {message}")

    def open_camera(self):
        _, frame = cap.read()
        if self.do_annotations:
            landmarkerResults = self.tmapMaker.detectHands(frame)
            frame = self.tmapMaker.annotateImage(frame, landmarkerResults)
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        self.camera_view.photo_image = photo_image
        self.camera_view.configure(image=photo_image)
        if self.camera_enabled:
            self.camera_view.after(10, self.open_camera)

    def closed_camera(self):
        captured_image = Image.new('RGB', (frame_width, frame_height))
        photo_image = ImageTk.PhotoImage(image=captured_image)
        self.camera_view.photo_image = photo_image
        self.camera_view.configure(image=photo_image)
        if not self.camera_enabled:
            self.camera_view.after(10, self.closed_camera)

    def toggle_camera(self):
        self.camera_enabled = not self.camera_enabled
        self.open_camera() if self.camera_enabled else self.closed_camera()
    
    def cam_init(self):
        self.closed_camera()

    def init(self):
        self.cam_init()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f'{width}x{height}')
    app = App(root)
    app.init()
    root.mainloop()
