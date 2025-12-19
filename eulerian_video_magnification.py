"""Motion and Color Magnification"""

import argparse
import os

import cv2
import numpy as np
from scipy.fftpack import fft, ifft


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
        for i in range(levels - 1):
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
        filtered_data = np.real(ifft(fft_data, axis=0))

        return filtered_data

    def magnify_motion(
        self, alpha=10, level=4, freq_min=0.4, freq_max=3.0, amplify_color=False, attenuate=True
    ):
        """Magnify subtle motions in video

        Parameters
        ----------
        - alpha: amplification factor (higher = more magnification)
        - level: pyramid levels for spatial filtering
        - freq_min: lower frequency bound (Hz)
        - freq_max: upper frequency bound (Hz)
        - amplify_color: if True, amplify color changes; if False, amplify motion
        - attenuate: if True, apply attenuation based on pyramid level

        """
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.frame_count}")
        print(f"FPS: {self.fps}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Amplification factor: {alpha}")
        print(f"Frequency range: {freq_min}-{freq_max} Hz")

        # Read all frames and build pyramids
        print("\nBuilding Gaussian pyramids...")
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
                print(f"Processed {frame_idx + 1}/{self.frame_count} frames")

        # Apply temporal filtering to each pyramid level
        print("\nApplying temporal filtering...")
        filtered_pyramid_videos = []

        for pyr_level in range(level):
            print(f"Filtering pyramid level {pyr_level + 1}/{level}")
            filtered = self.temporal_bandpass_filter(
                pyramid_videos[pyr_level], self.fps, freq_min, freq_max
            )
            filtered_pyramid_videos.append(filtered)

        # Magnify and reconstruct
        print("\nMagnifying and reconstructing frames...")
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
                    lambda_c = (self.width**2 + self.height**2) ** 0.5
                    delta = lambda_c / 8 / (1 + alpha)
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
                print(f"Reconstructed {frame_idx + 1}/{self.frame_count} frames")

        out.release()
        self.cap.release()
        print(f"\nDone! Output saved to: {self.output_path}")

    def magnify_color(self, alpha=50, level=4, freq_min=0.4, freq_max=3.0):
        """Convenience method for color magnification (e.g., visualizing pulse)"""  # noqa: D401
        self.magnify_motion(
            alpha=alpha,
            level=level,
            freq_min=freq_min,
            freq_max=freq_max,
            amplify_color=True,
            attenuate=False,
        )


def main():
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

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
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
