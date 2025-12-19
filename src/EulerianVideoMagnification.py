"""Video Magnification Logic"""

import logging

import cv2
import numpy as np
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

    def build_gaussian_pyramid(self, frame: np.ndarray, levels: int = 4) -> list[np.ndarray]:
        """Build Gaussian pyramid for spatial filtering.

        Args:
            frame: Input frame as numpy array
            levels: Number of pyramid levels to create (default: 4)

        Returns:
            List of downsampled frames, from original size to smallest

        """
        pyramid = [frame.copy()]
        current_level = frame
        for _ in range(levels - 1):
            current_level = cv2.pyrDown(current_level)
            pyramid.append(current_level)
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

    def magnify_color(self, alpha=50, level=4, freq_min=0.4, freq_max=3.0, progress_callback=None):
        """Convenience method for color magnification (e.g., visualizing pulse)"""  # noqa: D401
        self.magnify_motion(
            alpha=alpha,
            level=level,
            freq_min=freq_min,
            freq_max=freq_max,
            attenuate=False,
            progress_callback=progress_callback,
        )
