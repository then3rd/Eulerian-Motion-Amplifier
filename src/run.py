"""Motion and Color Magnification"""

import argparse
import logging
import os
import sys

from PyQt6.QtWidgets import (
    QApplication,
)

from .EulerianVideoMagnification import EulerianVideoMagnification
from .VideoMagnificationGUI import VideoMagnificationGUI

logger = logging.getLogger(__name__)


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
    parser.add_argument("-i", "--input", required=False, help="Input video file (MP4)")
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
