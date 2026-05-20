import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt

# Import our core brains
from core.ai_segmenter import BackgroundSegmenter
from core.dsp_engine import DSPEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP App - Interactive Control Panel")
        self.resize(1000, 600) # Made the window a bit wider for the buttons

        # --- Initialize AI and DSP ---
        self.segmenter = BackgroundSegmenter()
        self.dsp = DSPEngine()

        # --- State Variables ---
        # These tell our video loop what math to apply right now
        self.current_fg_mode = "Normal"  
        self.current_bg_mode = "Normal"

        # --- Build the UI Layout ---
        self.setup_ui()

        # --- Initialize Webcam ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return 

        # --- Start Video Loop ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_and_display_frame)
        self.timer.start(30) 

    def setup_ui(self):
        """Creates the grid, labels, and buttons for the GUI"""
        # Create a central widget and a horizontal layout (Video Left | Controls Right)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. The Video Viewport (Left Side)
        self.viewport = QLabel()
        self.viewport.setScaledContents(True)
        self.viewport.setMinimumSize(640, 480)
        main_layout.addWidget(self.viewport, stretch=3) # Takes up 3/4 of the screen

        # 2. The Control Panel (Right Side)
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=1) # Takes up 1/4 of the screen

        # --- Foreground Controls ---
        lbl_fg = QLabel("Subject (Foreground)")
        lbl_fg.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        control_panel.addWidget(lbl_fg)

        btn_fg_normal = QPushButton("Normal (Bypass DSP)")
        btn_fg_normal.clicked.connect(lambda: self.set_fg_mode("Normal"))
        control_panel.addWidget(btn_fg_normal)

        btn_fg_edge = QPushButton("Edge Detect")
        btn_fg_edge.clicked.connect(lambda: self.set_fg_mode("Edge Detect"))
        control_panel.addWidget(btn_fg_edge)

        btn_fg_sharpen = QPushButton("Sharpen")
        btn_fg_sharpen.clicked.connect(lambda: self.set_fg_mode("Sharpen"))
        control_panel.addWidget(btn_fg_sharpen)

        btn_fg_emboss = QPushButton("Emboss")
        btn_fg_emboss.clicked.connect(lambda: self.set_fg_mode("Emboss"))
        control_panel.addWidget(btn_fg_emboss)

        control_panel.addSpacing(20) # Empty space between sections

        # --- Background Controls ---
        lbl_bg = QLabel("Background")
        lbl_bg.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        control_panel.addWidget(lbl_bg)

        btn_bg_normal = QPushButton("Normal (Bypass DSP)")
        btn_bg_normal.clicked.connect(lambda: self.set_bg_mode("Normal"))
        control_panel.addWidget(btn_bg_normal)

        btn_bg_blur = QPushButton("Heavy Blur")
        btn_bg_blur.clicked.connect(lambda: self.set_bg_mode("Blur"))
        control_panel.addWidget(btn_bg_blur)

        btn_bg_black = QPushButton("Solid Black")
        btn_bg_black.clicked.connect(lambda: self.set_bg_mode("Black"))
        control_panel.addWidget(btn_bg_black)

        control_panel.addStretch() # Pushes all buttons to the top

    # --- Button Click Handlers ---
    def set_fg_mode(self, mode):
        self.current_fg_mode = mode
        print(f"Foreground set to: {mode}")

    def set_bg_mode(self, mode):
        self.current_bg_mode = mode
        print(f"Background set to: {mode}")

    # --- The Core Loop ---
    def process_and_display_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        # 1. AI SEGMENTATION
        mask = self.segmenter.get_mask(frame)
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 2. PROCESS FOREGROUND (Implementing Option 2 Optimization)
        if self.current_fg_mode == "Normal":
            processed_fg = frame # Bypass math entirely!
        else:
            kernel = self.dsp.kernels[self.current_fg_mode]
            processed_fg = self.dsp.apply_convolution(frame, kernel)

        # 3. PROCESS BACKGROUND
        if self.current_bg_mode == "Normal":
            processed_bg = frame # Bypass math
        elif self.current_bg_mode == "Blur":
            kernel = self.dsp.get_blur_kernel(size=25)
            processed_bg = self.dsp.apply_convolution(frame, kernel)
        elif self.current_bg_mode == "Black":
            # Just create an array of pure zeros (black) the same size as the frame
            processed_bg = np.zeros_like(frame)

        # 4. COMPOSITE & DISPLAY
        final_output = self.dsp.composite_layers(processed_fg, processed_bg, mask_3d)

        rgb_frame = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, width, height, channels * width, QImage.Format.Format_RGB888)
        self.viewport.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'cap'):
            self.cap.release()
        print("Webcam released. Shutting down.")