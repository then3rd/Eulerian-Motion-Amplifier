"""GUI class"""

import logging
import os

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .ProcessingThread import ProcessingThread

logger = logging.getLogger(__name__)


class VideoMagnificationGUI(QWidget):
    """PyQt6 GUI for Eulerian Video Magnification"""

    def __init__(self):
        super().__init__()
        self.processing_thread = None
        self.init_ui()

    def init_ui(self):  # noqa: PLR0915
        """Layout the QT UI"""
        self.setWindowTitle("Eulerian Video Magnification")
        self.setMinimumWidth(600)

        layout = QVBoxLayout()

        # File selection group
        file_group = QGroupBox("Files")
        file_layout = QVBoxLayout()

        # Input file
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Video:"))
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select input video file...")
        input_layout.addWidget(self.input_path)
        self.input_btn = QPushButton("Browse")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_btn)
        file_layout.addLayout(input_layout)

        # Output file
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Video:"))
        self.output_path = QLineEdit("output.mp4")
        output_layout.addWidget(self.output_path)
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_btn)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["motion", "color"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        params_layout.addLayout(mode_layout)

        # Alpha (amplification factor)
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Amplification Factor (alpha):"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(1.0, 200.0)
        self.alpha_spin.setValue(10.0)
        self.alpha_spin.setSingleStep(1.0)
        alpha_layout.addWidget(self.alpha_spin)
        alpha_layout.addStretch()
        params_layout.addLayout(alpha_layout)

        # Pyramid levels
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Pyramid Levels:"))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 10)
        self.level_spin.setValue(4)
        level_layout.addWidget(self.level_spin)
        level_layout.addStretch()
        params_layout.addLayout(level_layout)

        # Frequency min
        freq_min_layout = QHBoxLayout()
        freq_min_layout.addWidget(QLabel("Min Frequency (Hz):"))
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.0, 100.0)
        self.freq_min_spin.setValue(0.4)
        self.freq_min_spin.setSingleStep(0.1)
        self.freq_min_spin.setDecimals(2)
        freq_min_layout.addWidget(self.freq_min_spin)
        freq_min_layout.addStretch()
        params_layout.addLayout(freq_min_layout)

        # Frequency max
        freq_max_layout = QHBoxLayout()
        freq_max_layout.addWidget(QLabel("Max Frequency (Hz):"))
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.0, 100.0)
        self.freq_max_spin.setValue(3.0)
        self.freq_max_spin.setSingleStep(0.1)
        self.freq_max_spin.setDecimals(2)
        freq_max_layout.addWidget(self.freq_max_spin)
        freq_max_layout.addStretch()
        params_layout.addLayout(freq_max_layout)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setStyleSheet("QPushButton { padding: 10px; font-weight: bold; }")
        layout.addWidget(self.process_btn)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

    def on_mode_changed(self, mode):
        """Update default alpha value when mode changes"""
        if mode == "color":
            self.alpha_spin.setValue(50.0)
        else:
            self.alpha_spin.setValue(10.0)

    def browse_input(self):
        """Browse for input video file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if filename:
            self.input_path.setText(filename)
            # Auto-set output path based on input
            if not self.output_path.text() or self.output_path.text() == "output.mp4":
                base, _ext = os.path.splitext(filename)
                self.output_path.setText(f"{base}_magnified.mp4")

    def browse_output(self):
        """Browse for output video file"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Video",
            self.output_path.text(),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if filename:
            self.output_path.setText(filename)

    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)

    def process_video(self):
        """Start video processing in separate thread"""
        input_path = self.input_path.text()
        output_path = self.output_path.text()

        # Validation
        if not input_path:
            self.log("Error: Please select an input video file")
            return

        if not os.path.exists(input_path):
            self.log(f"Error: Input file '{input_path}' not found")
            return

        if not output_path:
            self.log("Error: Please specify an output file path")
            return

        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("Starting video processing...")

        # Create and start processing thread
        self.processing_thread = ProcessingThread(
            input_path,
            output_path,
            self.alpha_spin.value(),
            self.level_spin.value(),
            self.freq_min_spin.value(),
            self.freq_max_spin.value(),
            self.mode_combo.currentText(),
        )
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def processing_finished(self, success, message):
        """Handle processing completion"""
        self.log(message)
        self.process_btn.setEnabled(True)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)

        if success:
            self.progress_bar.setValue(100)
        else:
            self.progress_bar.setValue(0)
