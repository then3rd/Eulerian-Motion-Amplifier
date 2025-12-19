"""GUI threading"""

import logging

from PyQt6.QtCore import QThread, pyqtSignal

from .EulerianVideoMagnification import EulerianVideoMagnification

logger = logging.getLogger(__name__)


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
                    progress_callback=self.progress_update.emit,
                )

            self.finished_signal.emit(True, f"Success! Output saved to: {self.output_path}")  # noqa: FBT003
        except Exception as e:  # noqa: BLE001
            self.finished_signal.emit(False, f"Error: {e!s}")  # noqa: FBT003
