"""Motion and Color Magnification"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
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
from scipy.fftpack import fft, ifft

logger = logging.getLogger(__name__)


class EulerianVideoMagnification:  # noqa: D101
    def __init__(self, video_path, output_path="output.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def build_gaussian_pyramid(self, frame, levels=4):
        """Build Gaussian pyramid for spatial filtering"""
        pyramid = [frame]
        for _i in range(levels - 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstruct_from_pyramid(self, pyramid, levels):
        """Reconstruct frame from Gaussian pyramid"""
        current = pyramid[-1]
        for i in range(levels - 1, 0, -1):
            current = cv2.pyrUp(current)
            # Ensure size matches
            if current.shape != pyramid[i - 1].shape:
                current = cv2.resize(current, (pyramid[i - 1].shape[1], pyramid[i - 1].shape[0]))
            current += pyramid[i - 1]
        return current

    def temporal_bandpass_filter(self, data, fps, freq_min=0.4, freq_max=3.0):
        """Apply temporal bandpass filter"""
        fft_data = fft(data, axis=0)
        frequencies = np.fft.fftfreq(data.shape[0], d=1.0 / fps)

        # Create bandpass mask
        mask = np.logical_and(np.abs(frequencies) >= freq_min, np.abs(frequencies) <= freq_max)

        # Apply mask
        fft_data[~mask] = 0
        # filter data
        return np.real(ifft(fft_data, axis=0))

    def magnify_motion(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        alpha=10,
        level=4,
        freq_min=0.4,
        freq_max=3.0,
        *,
        attenuate=True,
        progress_callback=None,
    ):
        """Magnify subtle motions in video

        Args:
            alpha: amplification factor (higher = more magnification)
            level: pyramid levels for spatial filtering
            freq_min: lower frequency bound (Hz)
            freq_max: upper frequency bound (Hz)
            attenuate: if True, apply attenuation based on pyramid level
            progress_callback: progress callback function

        """
        logger.info("Processing video: %s", self.video_path)
        logger.info("Total frames: %d", self.frame_count)
        logger.info("FPS: %d", self.fps)
        logger.info("Resolution: %dx%d", self.width, self.height)
        logger.info("Amplification factor: %s", alpha)
        logger.info("Frequency range: %s-%s Hz", freq_min, freq_max)

        # Read all frames and build pyramids
        logger.info("Building Gaussian pyramids...")
        pyramid_videos = []

        for frame_idx in range(self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert to float
            frame = frame.astype(np.float32)

            # Build pyramid
            pyramid = self.build_gaussian_pyramid(frame, levels=level)

            if frame_idx == 0:
                # Initialize pyramid video storage
                for pyr_level in range(level):
                    h, w = pyramid[pyr_level].shape[:2]
                    pyramid_videos.append(np.zeros((self.frame_count, h, w, 3), dtype=np.float32))

            # Store each pyramid level
            for pyr_level in range(level):
                pyramid_videos[pyr_level][frame_idx] = pyramid[pyr_level]

            if (frame_idx + 1) % 50 == 0:
                logger.info("Processed %d/%d frames", frame_idx + 1, self.frame_count)

            if progress_callback:
                progress = int((frame_idx + 1) / self.frame_count * 33)
                progress_callback(progress)

        # Apply temporal filtering to each pyramid level
        logger.info("Applying temporal filtering...")
        filtered_pyramid_videos = []

        for pyr_level in range(level):
            logger.info("Filtering pyramid level %d/%d", pyr_level + 1, level)
            filtered = self.temporal_bandpass_filter(
                pyramid_videos[pyr_level], self.fps, freq_min, freq_max
            )
            filtered_pyramid_videos.append(filtered)

        # Magnify and reconstruct
        logger.info("Magnifying and reconstructing frames...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Changed from VideoWriter_fourcc
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        for frame_idx in range(self.frame_count):
            ret, original_frame = self.cap.read()
            if not ret:
                break

            original_frame = original_frame.astype(np.float32)

            # Reconstruct filtered pyramid for this frame
            filtered_pyramid = []
            for pyr_level in range(level):
                filtered_level = filtered_pyramid_videos[pyr_level][frame_idx]

                # Apply amplification with optional attenuation
                if attenuate:
                    # Attenuate higher frequency components more
                    exaggeration = alpha * (1 - (pyr_level / level))
                    filtered_level *= exaggeration
                else:
                    filtered_level *= alpha

                filtered_pyramid.append(filtered_level)

            # Add original lowest level
            filtered_pyramid.append(self.build_gaussian_pyramid(original_frame, levels=level)[-1])

            # Reconstruct
            reconstructed = self.reconstruct_from_pyramid(filtered_pyramid, level)

            # Add to original
            output_frame = original_frame + reconstructed

            # Clip and convert
            output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)

            out.write(output_frame)

            if (frame_idx + 1) % 50 == 0:
                logger.info("Reconstructed %d/%d frames", frame_idx + 1, self.frame_count)

            if progress_callback:
                progress = 66 + int((frame_idx + 1) / self.frame_count * 34)
                progress_callback(progress)

        out.release()
        self.cap.release()
        logger.info("Done! Output saved to: %s", self.output_path)

        if progress_callback:
            progress_callback(100)

    def magnify_color(self, alpha=50, level=4, freq_min=0.4, freq_max=3.0):
        """Convenience method for color magnification (e.g., visualizing pulse)"""  # noqa: D401
        self.magnify_motion(
            alpha=alpha,
            level=level,
            freq_min=freq_min,
            freq_max=freq_max,
            attenuate=False,
        )


class ProcessingThread(QThread):
    """Thread for processing video to keep GUI responsive"""

    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, video_path, output_path, alpha, level, freq_min, freq_max, mode):  # noqa: PLR0913
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.alpha = alpha
        self.level = level
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.mode = mode

    def run(self):
        """Starting point for the thread."""  # noqa: D401
        try:
            evm = EulerianVideoMagnification(self.video_path, self.output_path)

            if self.mode == "motion":
                evm.magnify_motion(
                    alpha=self.alpha,
                    level=self.level,
                    freq_min=self.freq_min,
                    freq_max=self.freq_max,
                    progress_callback=self.progress_update.emit,
                )
            else:
                evm.magnify_color(
                    alpha=self.alpha,
                    level=self.level,
                    freq_min=self.freq_min,
                    freq_max=self.freq_max,
                )

            self.finished_signal.emit(True, f"Success! Output saved to: {self.output_path}")
        except Exception as e:  # noqa: BLE001
            self.finished_signal.emit(False, f"Error: {e!s}")


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


def main():
    """Primary routine."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Check if GUI should be launched (no command-line arguments)
    if len(sys.argv) == 1:
        # Launch GUI
        app = QApplication(sys.argv)  # pyright: ignore[reportPossiblyUnboundVariable]
        gui = VideoMagnificationGUI()
        gui.show()
        sys.exit(app.exec())

    # Command-line mode
    parser = argparse.ArgumentParser(description="Eulerian Video Magnification")
    parser.add_argument("input", help="Input video file (MP4)")
    parser.add_argument(
        "-o", "--output", default="output.mp4", help="Output video file (default: output.mp4)"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=10, help="Amplification factor (default: 10)"
    )
    parser.add_argument("-l", "--level", type=int, default=4, help="Pyramid levels (default: 4)")
    parser.add_argument(
        "--freq-min", type=float, default=0.4, help="Minimum frequency in Hz (default: 0.4)"
    )
    parser.add_argument(
        "--freq-max", type=float, default=3.0, help="Maximum frequency in Hz (default: 3.0)"
    )
    parser.add_argument(
        "--mode",
        choices=["motion", "color"],
        default="motion",
        help="Magnification mode (default: motion)",
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch GUI mode (default if no arguments provided)"
    )

    args = parser.parse_args()

    # Launch GUI if --gui flag is provided
    if args.gui:
        app = QApplication(sys.argv)
        gui = VideoMagnificationGUI()
        gui.show()
        sys.exit(app.exec())

    # CLI processing
    if not os.path.exists(args.input):
        logger.error("Input file '%s' not found", args.input)
        return

    evm = EulerianVideoMagnification(args.input, args.output)

    if args.mode == "motion":
        evm.magnify_motion(
            alpha=args.alpha, level=args.level, freq_min=args.freq_min, freq_max=args.freq_max
        )
    else:
        evm.magnify_color(
            alpha=args.alpha, level=args.level, freq_min=args.freq_min, freq_max=args.freq_max
        )


if __name__ == "__main__":
    main()
