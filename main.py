import os
import cv2
import sys
import time
import glob
import json
import numpy as np
import tensorflow as tf
from supabase import create_client, Client
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit,
                             QDialog, QDoubleSpinBox, QSpinBox, QCheckBox, QVBoxLayout as QVBoxLayout2,
                             QHBoxLayout as QHBoxLayout2, QGroupBox, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, QSize, Qt, pyqtSignal

# Supabase configuration
SUPABASE_URL = "https://dhjtafvgharqjvwifsbt.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_H_pI_4ZPVlrPSUWYRZWMCQ_EW2TgHuz"

# Display settings for Linux (commented out for Windows)
# os.environ['QT_QPA_PLATFORM'] = 'wayland'


class CameraUI(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.scan_mode = False  # Flag for scan mode
        self.trayDetected = False
        self.lastScanTime = 0
        self.SCAN_COOLDOWN_MS = 1000

        # Enhancement settings 
        self.clahe_clip_value = 1.5
        self.clahe_tile_value = 8
        self.sharpen_strength_value = 4.8
        self.dark_threshold_value = 0.7
        self.bright_threshold_value = 1.3
        self.tray_threshold_value = 0.06
        self.enable_gamma_value = True
        self.enable_clahe_value = True
        self.enable_sharpen_value = True
        self.model_threshold_value = 0.5        # AI decision threshold (0.5 = default)
        self.scan_cooldown_value = 1000         # Scan cooldown in milliseconds (1000 = 1 second)

        # Model and resolution settings
        self.model_file_value = "model_int8.tflite"  # Current model file (always reset to default)
        self.feed_width_value = 2560                # Feed resolution width (default: 2560×1536)
        self.feed_height_value = 1536               # Feed resolution height (default: 2560×1536)
        self.region_width_value = 960               # Detection box how width
        self.region_height_value = 720              # Detection box height

        self.init_ui()

        # Load saved settings (model path always resets)
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle('Pemindai Nampan Python')
        self.setGeometry(100, 100, 800, 700)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        controls_layout = QHBoxLayout()

        facility_label = QLabel('Fasilitas:')
        self.facility_combo = QComboBox()

        camera_label = QLabel('Kamera:')
        self.camera_combo = QComboBox()
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)

        self.scan_btn = QPushButton('Mulai Pemindaian')
        self.scan_btn.clicked.connect(self.start_scan)

        self.settings_btn = QPushButton('⚙')
        self.settings_btn.clicked.connect(lambda: SettingsDialog(self).exec_())

        controls_layout.addWidget(facility_label)
        controls_layout.addWidget(self.facility_combo)
        controls_layout.addWidget(camera_label)
        controls_layout.addWidget(self.camera_combo)
        controls_layout.addWidget(self.scan_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.settings_btn)

        layout.addLayout(controls_layout)

        self.preview_label = QLabel()
        self.preview_label.setMinimumHeight(600)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Terminal console
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(150)
        terminal_font = QFont("DejaVu Sans Mono", 12)
        self.terminal.setFont(terminal_font)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: #00FF00;
                border: 2px solid #00FF00;
                font-family: 'DejaVu Sans Mono';
                padding: 5px;
            }
        """)
        layout.addWidget(self.terminal)

        # Initialize terminal
        self.log("Terminal Pemindai Nampan")
        self.log("========================")
        self.log("Siap beroperasi")

        self.setLayout(layout)

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            self.log("Klien Supabase diinisialisasi")
        except Exception as e:
            self.log(f"Gagal menginisialisasi Supabase: {str(e)}")
            self.supabase = None

        # Populate facility and camera lists
        self.populate_facilities()
        self.enumerate_cameras()

        # Load TensorFlow Lite model
        self.modelLoaded = self.load_model(self.model_file_value)

    def log(self, message):
        """Add message to terminal console"""
        from datetime import datetime
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.terminal.append(f"{timestamp} {message}")

    def load_model(self, model_path):
        """Load TensorFlow Lite model and return success status"""
        try:
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Detect number of classes from output shape
            output_shape = self.output_details[0]['shape']
            self.num_classes = output_shape[1] 

            # Set class names based on number of classes
            if self.num_classes == 2:
                self.class_names = ["CLEAN", "DIRTY"]
            elif self.num_classes == 3:
                self.class_names = ["CLEAN", "DIRTY", "TERBALIK"]
            else:
                self.log(f"Unsupported number of classes: {self.num_classes}")
                return False

            # Check quantization
            self.is_quantized = self.input_details[0]['dtype'] != np.float32

            self.log(f"Model {model_path} loaded successfully - {self.num_classes} classes: {self.class_names}")
            return True
        except Exception as e:
            self.log(f"Failed to load model {model_path}: {str(e)}")
            return False

    def populate_facilities(self):
        """Populate facility dropdown with data from Supabase dapurs table"""
        self.facility_combo.clear()

        if not self.supabase:
            self.facility_combo.addItem("No Supabase connection")
            return

        try:
            # Query dapur table
            response = self.supabase.table("dapurs").select("id,name").execute()
            data = response.data

            if data:
                for dapur in data:
                    self.facility_combo.addItem(dapur["name"])
                    self.facility_combo.setItemData(self.facility_combo.count() - 1, dapur["id"])

                self.log(f"Loaded {len(data)} facilities from database")
            else:
                self.facility_combo.addItem("No facilities found")
                self.log("No facilities found in database")

        except Exception as e:
            self.log(f"Failed to load facilities: {str(e)}")
            self.facility_combo.addItem("Failed to load facilities")

    def enumerate_cameras(self):
        """Find available cameras"""
        self.camera_combo.clear()
        available_cameras = []

        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(str(i))
                cap.release()

        if not available_cameras:
            available_cameras = ['No cameras found']

        self.camera_combo.addItems(available_cameras)

    def on_camera_changed(self, camera_index_str):
        """Handle camera selection change"""
        if camera_index_str == 'No cameras found':
            if self.cap:
                self.cap.release()
                self.cap = None
            self.timer.stop()
            return

        try:
            camera_index = int(camera_index_str)
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                # Set camera to selected resolution from settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.feed_width_value)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.feed_height_value)
                self.timer.start(33)  # ~30 FPS
                self.log(f"Camera {camera_index} opened successfully at {self.feed_width_value}x{self.feed_height_value} resolution")
            else:
                self.log(f"Failed to open camera {camera_index}")
        except ValueError:
            pass

    def update_frame(self):
        """Update camera preview and detect tray if in scan mode"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Tray detection
                tray_detected = self.detect_tray(frame)

                if self.scan_mode:
                    currentMs = int(time.time() * 1000)  # milliseconds since epoch

                    # Detect tray entry
                    if tray_detected and not self.trayDetected and (currentMs - self.lastScanTime) > self.scan_cooldown_value:
                        self.trayDetected = True
                        self.lastScanTime = currentMs

                        # Run model inference
                        status = self.run_inference(frame)
                        self.log(f"TRAY STATUS: {status}")

                        # Send scan data to Supabase
                        self.send_scan_data(status)

                    # Detect tray exit (exactly like C++)
                    if not tray_detected and self.trayDetected:
                        self.trayDetected = False

                # Draw detection region
                regionWidth = self.region_width_value
                regionHeight = self.region_height_value
                startX = max(0, min((frame.shape[1] - regionWidth) // 2, frame.shape[1] - regionWidth))
                startY = max(0, min((frame.shape[0] - regionHeight) // 2, frame.shape[0] - regionHeight))

                # Indicator Colour
                current_time_ms = int(time.time() * 1000)
                in_cooldown = self.scan_mode and self.trayDetected and (current_time_ms - self.lastScanTime) < self.scan_cooldown_value

                if not self.scan_mode:
                    color = (255, 0, 0)  # Blue - Not scanning
                elif in_cooldown:
                    color = (255, 0, 255)  # Purple - Cooldown period
                elif tray_detected and self.scan_mode:
                    color = (0, 255, 0)  # Green - Tray detected, ready to scan
                else:
                    color = (0, 0, 255)  # Red - Scan mode active, waiting for tray
                cv2.rectangle(frame, (startX, startY), (startX + regionWidth, startY + regionHeight), color, 3)

                label = "Zona Deteksi Tray" + (" (Aktif)" if self.scan_mode else "")
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Convert to RGB
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create QImage
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(qt_image)
                pixmap = pixmap.scaled(QSize(800, 600), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(pixmap)
            else:
                self.preview_label.setText('Failed to capture frame')

    def detect_tray(self, frame):
        """Detect tray presence - exact copy of C++ logic"""
        regionWidth = self.region_width_value
        regionHeight = self.region_height_value
        startX = (frame.shape[1] - regionWidth) // 2  # cols
        startY = (frame.shape[0] - regionHeight) // 2  # rows

        startX = max(0, min(startX, frame.shape[1] - regionWidth))
        startY = max(0, min(startY, frame.shape[0] - regionHeight))

        # Crop frame
        detectionRegion = (startX, startY, regionWidth, regionHeight)
        roi = frame[startY:startY+regionHeight, startX:startX+regionWidth]

        # Edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Edge Pixel
        edgePixels = cv2.countNonZero(edges)
        totalPixels = edges.shape[0] * edges.shape[1]
        edgeRatio = edgePixels / totalPixels if totalPixels > 0 else 0

        return edgeRatio > self.tray_threshold_value  

    def send_scan_data(self, tray_status):
        """Send scan result to Supabase"""
        if not self.supabase:
            self.log("Cannot send data: No Supabase connection")
            return

        try:
            # Get selected facility ID
            facility_id = self.facility_combo.currentData()
            if not facility_id:
                self.log("Cannot send data: No facility selected")
                return

            # Map TERBALIK to DIRTY for database compatibility
            db_status = "DIRTY" if tray_status == "TERBALIK" else tray_status

            # Prepare payload
            from datetime import datetime
            payload = {
                'dapur_id': facility_id,
                'tray_status': db_status,
                'scanned_at': datetime.now().isoformat()
            }

            # Send to Supabase
            response = self.supabase.table('scans').insert(payload).execute()
            self.log(f"Scan data sent to Supabase: {db_status}")

        except Exception as e:
            self.log(f"Failed to send scan data: {str(e)}")

    def run_inference(self, frame):
        """Run TensorFlow Lite model inference on frame"""
        if not self.modelLoaded:
            return "ERROR: Model not loaded"

        try:
            # Crop detection region
            regionWidth = self.region_width_value
            regionHeight = self.region_height_value
            startX = (frame.shape[1] - regionWidth) // 2
            startY = (frame.shape[0] - regionHeight) // 2
            startX = max(0, min(startX, frame.shape[1] - regionWidth))
            startY = max(0, min(startY, frame.shape[0] - regionHeight))

            roi = frame[startY:startY+regionHeight, startX:startX+regionWidth]

            # Preprocess image for MobileNetV2
            img = cv2.resize(roi, (224, 224))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))

            if self.is_quantized:
                # Quantize for INT8 model
                scale, zero_point = self.input_details[0]['quantization']
                quantized = np.round(img / scale) + zero_point
                quantized = np.clip(quantized, -128, 127).astype(np.int8)
                input_data = np.expand_dims(quantized, axis=0)
            else:
                # Quantize Float32 model
                input_data = np.expand_dims(img, axis=0)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get prediction
            prediction = np.argmax(output_data[0])
            status = self.class_names[prediction]

            return status

        except Exception as e:
            return f"ERROR: {str(e)}"

    def start_scan(self):
        """Toggle scan mode"""
        self.scan_mode = not self.scan_mode
        if self.scan_mode:
            # Reset tray detection state
            self.trayDetected = False
            self.lastScanTime = 0

            self.scan_btn.setText('Hentikan Pemindaian')
            self.log("Mode pemindaian diaktifkan")
        else:
            self.scan_btn.setText('Mulai Pemindaian')
            self.log("Mode pemindaian dihentikan")

    def load_settings(self):
        """Load settings from JSON file (except model which always resets)"""
        settings_file = "settings.json"
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)

                # Load saved settings
                self.feed_width_value = settings.get('feed_width', 2560)
                self.feed_height_value = settings.get('feed_height', 1536)
                self.region_width_value = settings.get('region_width', 960)
                self.region_height_value = settings.get('region_height', 720)
                self.clahe_clip_value = settings.get('clahe_clip', 1.5)
                self.clahe_tile_value = settings.get('clahe_tile', 8)
                self.sharpen_strength_value = settings.get('sharpen_strength', 4.8)
                self.dark_threshold_value = settings.get('dark_threshold', 0.7)
                self.bright_threshold_value = settings.get('bright_threshold', 1.3)
                self.tray_threshold_value = settings.get('tray_threshold', 0.06)
                self.enable_gamma_value = settings.get('enable_gamma', True)
                self.enable_clahe_value = settings.get('enable_clahe', True)
                self.enable_sharpen_value = settings.get('enable_sharpen', True)
                self.model_threshold_value = settings.get('model_threshold', 0.5)
                self.scan_cooldown_value = settings.get('scan_cooldown', 1000)

                self.log("Settings loaded from file")
            else:
                self.log("No settings file found, using defaults")
        except Exception as e:
            self.log(f"Failed to load settings: {str(e)}, using defaults")

    def save_settings(self):
        """Save current settings to JSON file"""
        settings_file = "settings.json"
        try:
            settings = {
                'feed_width': self.feed_width_value,
                'feed_height': self.feed_height_value,
                'region_width': self.region_width_value,
                'region_height': self.region_height_value,
                'clahe_clip': self.clahe_clip_value,
                'clahe_tile': self.clahe_tile_value,
                'sharpen_strength': self.sharpen_strength_value,
                'dark_threshold': self.dark_threshold_value,
                'bright_threshold': self.bright_threshold_value,
                'tray_threshold': self.tray_threshold_value,
                'enable_gamma': self.enable_gamma_value,
                'enable_clahe': self.enable_clahe_value,
                'enable_sharpen': self.enable_sharpen_value,
                'model_threshold': self.model_threshold_value,
                'scan_cooldown': self.scan_cooldown_value
            }

            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)

            self.log("Settings saved to file")
        except Exception as e:
            self.log(f"Failed to save settings: {str(e)}")

    def closeEvent(self, event):
        """Cleanup on close and save settings"""
        # Save settings before closing
        self.save_settings()

        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()


class SettingsDialog(QDialog):
    """Settings dialog for adjusting tray scanner enhancement parameters"""

    # Preset definitions
    RESOLUTION_PRESETS = {
        "1280×720 (720p HD)": (1280, 720),
        "1920×1080 (1080p Full HD)": (1920, 1080),
        "2048×1536 (QSXGA)": (2048, 1536),
        "2560×1536 (WQXGA)": (2560, 1536),
        "3840×2160 (4K UHD)": (3840, 2160)
    }

    BOX_SIZE_PRESETS = {
        "640×480 (0.5× for HD)": (640, 480),
        "960×720 (0.5× for FHD)": (960, 720),
        "1024×768 (0.5× for QSXGA)": (1024, 768),
        "1280×960 (0.5× for QHD)": (1280, 960),
        "1920×1440 (0.5× for 4K)": (1920, 1440)
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.selected_model_path = parent.model_file_value
        self.setWindowTitle("Tray Scanner Settings")
        self.setModal(True)
        self.resize(600, 700) 

        # Log current settings
        self.parent.log("Opening settings dialog with current values...")

        layout = QVBoxLayout2(self)

        # Model selection
        model_group = QGroupBox("AI Model Selection")
        model_layout = QVBoxLayout2()
        self.model_label = QLabel(f"Current: {os.path.basename(parent.model_file_value)}")
        model_layout.addWidget(self.model_label)
        button_layout = QHBoxLayout2()
        self.browse_model_btn = QPushButton("Browse Files...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        self.reset_model_btn = QPushButton("Reset to Default")
        self.reset_model_btn.clicked.connect(self.reset_to_default_model)
        button_layout.addWidget(self.browse_model_btn)
        button_layout.addWidget(self.reset_model_btn)
        model_layout.addLayout(button_layout)
        model_group.setLayout(model_layout)

        # Feed resolution settings
        feed_group = QGroupBox("Feed Resolution (Applied to camera)")
        feed_layout = QVBoxLayout2()
        feed_layout.addWidget(QLabel("Preset:"))
        self.feed_preset = QComboBox()
        # Add preset options
        for preset_name in self.RESOLUTION_PRESETS.keys():
            self.feed_preset.addItem(preset_name)
        # Set current resolution as selected
        current_preset = next((name for name, (w, h) in self.RESOLUTION_PRESETS.items()
                              if w == parent.feed_width_value and h == parent.feed_height_value),
                             self.feed_preset.itemText(0))
        current_index = self.feed_preset.findText(current_preset)
        if current_index >= 0:
            self.feed_preset.setCurrentIndex(current_index)
        feed_layout.addWidget(self.feed_preset)
        # Add individual display for info
        self.feed_info_label = QLabel(f"Current: {parent.feed_width_value}×{parent.feed_height_value}")
        feed_layout.addWidget(self.feed_info_label)
        # Connect change signal
        self.feed_preset.currentTextChanged.connect(self.on_feed_preset_changed)
        feed_group.setLayout(feed_layout)

        # Detection box size settings
        box_group = QGroupBox("Detection Box Size")
        box_layout = QVBoxLayout2()
        box_layout.addWidget(QLabel("Preset:"))
        self.box_preset = QComboBox()
        # Add preset options
        for preset_name in self.BOX_SIZE_PRESETS.keys():
            self.box_preset.addItem(preset_name)
        # Set current box size as selected
        current_box_preset = next((name for name, (w, h) in self.BOX_SIZE_PRESETS.items()
                                  if w == parent.region_width_value and h == parent.region_height_value),
                                 self.box_preset.itemText(0))
        current_box_index = self.box_preset.findText(current_box_preset)
        if current_box_index >= 0:
            self.box_preset.setCurrentIndex(current_box_index)
        box_layout.addWidget(self.box_preset)
        # Add individual display for info
        self.box_info_label = QLabel(f"Current: {parent.region_width_value}×{parent.region_height_value}")
        box_layout.addWidget(self.box_info_label)
        # Connect change signal
        self.box_preset.currentTextChanged.connect(self.on_box_preset_changed)
        box_group.setLayout(box_layout)

        # CLAHE settings
        clahe_group = QGroupBox("CLAHE Enhancement")
        clahe_layout = QVBoxLayout2()
        clahe_layout.addWidget(QLabel("Clip Limit:"))
        self.clahe_clip = QDoubleSpinBox()
        self.clahe_clip.setRange(1.0, 4.0)
        self.clahe_clip.setSingleStep(0.1)
        self.clahe_clip.setValue(parent.clahe_clip_value)
        clahe_layout.addWidget(self.clahe_clip)
        clahe_layout.addWidget(QLabel("Tile Grid Size:"))
        self.clahe_tile = QSpinBox()
        self.clahe_tile.setRange(2, 16)
        self.clahe_tile.setValue(parent.clahe_tile_value)
        clahe_layout.addWidget(self.clahe_tile)
        clahe_group.setLayout(clahe_layout)

        # Sharpening settings
        sharpen_group = QGroupBox("Sharpening")
        sharpen_layout = QVBoxLayout2()
        sharpen_layout.addWidget(QLabel("Strength:"))
        self.sharpen_strength = QDoubleSpinBox()
        self.sharpen_strength.setRange(1.0, 10.0)
        self.sharpen_strength.setSingleStep(0.1)
        self.sharpen_strength.setValue(parent.sharpen_strength_value)
        sharpen_layout.addWidget(self.sharpen_strength)
        sharpen_group.setLayout(sharpen_layout)

        # Brightness settings
        brightness_group = QGroupBox("Brightness Thresholds")
        brightness_layout = QVBoxLayout2()
        brightness_layout.addWidget(QLabel("Dark Threshold:"))
        self.dark_threshold = QDoubleSpinBox()
        self.dark_threshold.setRange(0.5, 1.0)
        self.dark_threshold.setSingleStep(0.1)
        self.dark_threshold.setValue(parent.dark_threshold_value)
        brightness_layout.addWidget(self.dark_threshold)
        brightness_layout.addWidget(QLabel("Bright Threshold:"))
        self.bright_threshold = QDoubleSpinBox()
        self.bright_threshold.setRange(1.0, 2.0)
        self.bright_threshold.setSingleStep(0.1)
        self.bright_threshold.setValue(parent.bright_threshold_value)
        brightness_layout.addWidget(self.bright_threshold)
        brightness_group.setLayout(brightness_layout)

        # Tray detection threshold
        tray_group = QGroupBox("Tray Detection")
        tray_layout = QVBoxLayout2()
        tray_layout.addWidget(QLabel("Edge Ratio Threshold:"))
        self.tray_threshold = QDoubleSpinBox()
        self.tray_threshold.setRange(0.01, 0.2)
        self.tray_threshold.setSingleStep(0.01)
        self.tray_threshold.setValue(parent.tray_threshold_value)
        tray_layout.addWidget(self.tray_threshold)
        tray_group.setLayout(tray_layout)

        # Scan cooldown setting
        scan_group = QGroupBox("Scan Settings")
        scan_layout = QVBoxLayout2()
        scan_layout.addWidget(QLabel("Cooldown Period (ms):"))
        self.scan_cooldown = QSpinBox()
        self.scan_cooldown.setRange(500, 5000)
        self.scan_cooldown.setSingleStep(250)
        self.scan_cooldown.setValue(parent.scan_cooldown_value)
        scan_layout.addWidget(self.scan_cooldown)
        scan_group.setLayout(scan_layout)

        # AI Model threshold
        ai_group = QGroupBox("AI Decision Threshold")
        ai_layout = QVBoxLayout2()
        ai_layout.addWidget(QLabel("Dirty Classification Score:"))
        self.model_threshold = QDoubleSpinBox()
        self.model_threshold.setRange(0.3, 0.8)
        self.model_threshold.setSingleStep(0.05)
        self.model_threshold.setValue(parent.model_threshold_value)
        ai_layout.addWidget(self.model_threshold)
        ai_group.setLayout(ai_layout)

        # Checkboxes for enabling features
        feature_group = QGroupBox("Features")
        feature_layout = QVBoxLayout2()
        self.enable_gamma = QCheckBox("Enable Gamma Correction")
        self.enable_gamma.setChecked(parent.enable_gamma_value)
        feature_layout.addWidget(self.enable_gamma)
        self.enable_clahe = QCheckBox("Enable CLAHE Enhancement")
        self.enable_clahe.setChecked(parent.enable_clahe_value)
        feature_layout.addWidget(self.enable_clahe)
        self.enable_sharpen = QCheckBox("Enable Sharpening")
        self.enable_sharpen.setChecked(parent.enable_sharpen_value)
        feature_layout.addWidget(self.enable_sharpen)
        feature_group.setLayout(feature_layout)

        # Two-column layout arrangement
        # Row 1: Model + Feed resolution
        row1_layout = QHBoxLayout2()
        row1_layout.addWidget(model_group)
        row1_layout.addWidget(feed_group)
        layout.addLayout(row1_layout)

        # Row 2: Detection box + CLAHE
        row2_layout = QHBoxLayout2()
        row2_layout.addWidget(box_group)
        row2_layout.addWidget(clahe_group)
        layout.addLayout(row2_layout)

        # Row 3: Sharpening + Brightness
        row3_layout = QHBoxLayout2()
        row3_layout.addWidget(sharpen_group)
        row3_layout.addWidget(brightness_group)
        layout.addLayout(row3_layout)

        # Row 4: Tray detection + Scan settings
        row4_layout = QHBoxLayout2()
        row4_layout.addWidget(tray_group)
        row4_layout.addWidget(scan_group)
        layout.addLayout(row4_layout)

        # Row 5: AI threshold + Features
        row5_layout = QHBoxLayout2()
        row5_layout.addWidget(ai_group)
        row5_layout.addWidget(feature_group)
        layout.addLayout(row5_layout)

        # Buttons
        button_layout = QHBoxLayout2()
        reset_defaults_btn = QPushButton("Reset to Defaults")
        reset_defaults_btn.clicked.connect(self.reset_to_defaults)
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(reset_defaults_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browse_model(self):
        """Open file dialog to select a TFLite model file"""
        file_dialog = QFileDialog(self, "Select TFLite Model File", "", "TensorFlow Lite Files (*.tflite);;All Files (*)")
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            if selected_file:
                # Update the label to show selected file
                self.parent.log(f"Selected model file: {selected_file}")
                # Store temporarily, will be applied on save
                self.selected_model_path = selected_file
                self.model_label.setText(f"Selected: {os.path.basename(selected_file)}")

    def reset_to_default_model(self):
        """Reset to default model (model_int8.tflite)"""
        self.selected_model_path = "model_int8.tflite"
        self.model_label.setText(f"Selected: model_int8.tflite")
        self.parent.log("Reset to default model: model_int8.tflite")

    def on_feed_preset_changed(self, preset_name):
        """Update feed resolution info label when preset changes"""
        if preset_name in self.RESOLUTION_PRESETS:
            w, h = self.RESOLUTION_PRESETS[preset_name]
            self.feed_info_label.setText(f"Selected: {w}×{h}")

    def on_box_preset_changed(self, preset_name):
        """Update detection box size info label when preset changes"""
        if preset_name in self.BOX_SIZE_PRESETS:
            w, h = self.BOX_SIZE_PRESETS[preset_name]
            self.box_info_label.setText(f"Selected: {w}×{h}")

    def reset_to_defaults(self):
        """Reset all settings to their default values"""
        # Reset model path
        self.selected_model_path = "model_int8.tflite"
        self.model_label.setText("Selected: model_int8.tflite")

        # Reset feed resolution preset
        default_feed_preset = "2560×1536 (WQXGA)"
        if default_feed_preset in self.RESOLUTION_PRESETS:
            default_index = self.feed_preset.findText(default_feed_preset)
            if default_index >= 0:
                self.feed_preset.setCurrentIndex(default_index)
                self.on_feed_preset_changed(default_feed_preset)

        # Reset box size preset
        default_box_preset = "960×720 (0.5× for FHD)"
        if default_box_preset in self.BOX_SIZE_PRESETS:
            default_index = self.box_preset.findText(default_box_preset)
            if default_index >= 0:
                self.box_preset.setCurrentIndex(default_index)
                self.on_box_preset_changed(default_box_preset)

        # Reset all spin box values
        self.clahe_clip.setValue(1.5)
        self.clahe_tile.setValue(8)
        self.sharpen_strength.setValue(4.8)
        self.dark_threshold.setValue(0.7)
        self.bright_threshold.setValue(1.3)
        self.tray_threshold.setValue(0.06)
        self.scan_cooldown.setValue(1000)
        self.model_threshold.setValue(0.5)

        # Reset checkboxes
        self.enable_gamma.setChecked(True)
        self.enable_clahe.setChecked(True)
        self.enable_sharpen.setChecked(True)

        self.parent.log("All settings reset to defaults")

    def save_settings(self):
        """Save settings to parent and close dialog"""
        # Check if model needs to be reloaded
        model_changed = (self.selected_model_path != self.parent.model_file_value)

        if model_changed:
            if self.parent.load_model(self.selected_model_path):
                self.parent.model_file_value = self.selected_model_path
                self.parent.modelLoaded = True
            else:
                self.parent.modelLoaded = False
                self.parent.log("Failed to load new model, keeping old one")

        # Get values from preset dropdowns
        feed_preset = self.feed_preset.currentText()
        if feed_preset in self.RESOLUTION_PRESETS:
            self.parent.feed_width_value, self.parent.feed_height_value = self.RESOLUTION_PRESETS[feed_preset]

        box_preset = self.box_preset.currentText()
        if box_preset in self.BOX_SIZE_PRESETS:
            self.parent.region_width_value, self.parent.region_height_value = self.BOX_SIZE_PRESETS[box_preset]

        # Save all other settings to parent
        self.parent.clahe_clip_value = self.clahe_clip.value()
        self.parent.clahe_tile_value = self.clahe_tile.value()
        self.parent.sharpen_strength_value = self.sharpen_strength.value()
        self.parent.dark_threshold_value = self.dark_threshold.value()
        self.parent.bright_threshold_value = self.bright_threshold.value()
        self.parent.tray_threshold_value = self.tray_threshold.value()
        self.parent.scan_cooldown_value = self.scan_cooldown.value()
        self.parent.model_threshold_value = self.model_threshold.value()
        self.parent.enable_gamma_value = self.enable_gamma.isChecked()
        self.parent.enable_clahe_value = self.enable_clahe.isChecked()
        self.parent.enable_sharpen_value = self.enable_sharpen.isChecked()

        self.parent.log(f"Settings applied: Model={os.path.basename(self.parent.model_file_value)}, "
                       f"Box={self.parent.region_width_value}x{self.parent.region_height_value}, "
                       f"FeedRes={self.parent.feed_width_value}x{self.parent.feed_height_value}")
        self.parent.log(f"Features: Gamma={self.enable_gamma.isChecked()}, CLAHE={self.enable_clahe.isChecked()}, "
                       f"Sharpen={self.enable_sharpen.isChecked()}")
        self.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraUI()
    window.show()
    sys.exit(app.exec_())
