#!/usr/bin/env python3
"""
Smart Video Re-encoder - Intelligently Re-encode videos based on estimated size reduction
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import signal
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Codec name mappings - maps common names to canonical ffprobe names
CODEC_ALIASES = {
    # H.264 / AVC
    "h264": "h264",
    "h.264": "h264",
    "x264": "h264",
    "libx264": "h264",
    "avc": "h264",
    "avc1": "h264",
    # H.265 / HEVC
    "h265": "hevc",
    "h.265": "hevc",
    "x265": "hevc",
    "libx265": "hevc",
    "hevc": "hevc",
    "hev1": "hevc",
    "hvc1": "hevc",
    # VP9
    "vp9": "vp9",
    "vp09": "vp9",
    "libvpx-vp9": "vp9",
    "libvpx": "vp9",
    # AV1
    "av1": "av1",
    "av01": "av1",
    "libaom-av1": "av1",
    "libsvtav1": "av1",
    "svt-av1": "av1",
}


def normalize_codec_name(codec: str) -> str:
    """Normalize codec name to canonical ffprobe name"""
    return CODEC_ALIASES.get(codec.lower(), codec.lower())


class VideoTranscoder:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        source_codecs: List[str],
        target_codec: str,
        crf: int,
        threshold: float,
        sample_duration: int,
        dry_run: bool,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.source_codecs = [normalize_codec_name(codec) for codec in source_codecs]
        self.target_codec = target_codec
        self.crf = crf
        self.threshold = threshold
        self.sample_duration = sample_duration
        self.dry_run = dry_run

        self.stats = {
            "total_files": 0,
            "skipped_codec": 0,
            "skipped_size": 0,
            "transcoded": 0,
            "errors": 0,
            "total_saved": 0,
        }

    def handle_signal(self, signum: int):
        logger.info(f"\nSignal {signum} ({signal.Signals(signum).name}) received.")
        self.print_summary()
        sys.exit(0)

    def get_video_info(self, video_path: Path) -> Optional[Dict]:
        """Get video information using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed for {video_path}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output for {video_path}: {e}")
            return None

    def get_video_codec(self, info: Dict) -> Optional[str]:
        """Extract video codec from ffprobe info"""
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                codec = stream.get("codec_name", "").lower()
                return codec
        return None

    def get_duration(self, info: Dict) -> Optional[float]:
        """Extract duration from ffprobe info"""
        duration = info.get("format", {}).get("duration")
        if duration:
            return float(duration)
        return None

    def print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Skipped (codec): {self.stats['skipped_codec']}")
        logger.info(f"Skipped (size): {self.stats['skipped_size']}")
        logger.info(f"Transcoded: {self.stats['transcoded']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(
            f"Total saved: {self.stats['total_saved'] / 1024 / 1024 / 1024:.2f} GB"
        )
        if self.dry_run:
            logger.info("\n[DRY RUN MODE - No files were actually transcoded]")

    def transcode_sample(
        self, video_path: Path, start_time: float, duration: int, output_path: Path
    ) -> Optional[int]:
        """Transcode a sample segment and return its size in bytes"""
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-c:v:0",  # only first video stream
                self.target_codec,
                "-crf",
                str(self.crf),
                "-v",
                "quiet",
                "-stats",
                str(output_path),
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            if output_path.exists():
                return output_path.stat().st_size
            return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Sample transcoding failed: {e}")
            return None

    def estimate_transcoded_size(
        self, video_path: Path, duration: float
    ) -> Optional[float]:
        """Estimate the size of the transcoded video by sampling"""
        # Calculate sample positions: 25%, 50%, 75% through the video
        sample_positions = [duration * 0.25, duration * 0.50, duration * 0.75]

        sample_sizes = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for idx, start_time in enumerate(sample_positions):
                # Ensure we don't go past the end
                if start_time + self.sample_duration > duration:
                    start_time = max(0, duration - self.sample_duration)

                sample_path = (
                    tmpdir_path / f"sample_{idx}.{video_path.name.split(".")[-1]}"
                )

                logger.info(f"  Transcoding sample {idx + 1}/3 at {start_time:.1f}s...")
                sample_size = self.transcode_sample(
                    video_path, start_time, self.sample_duration, sample_path
                )

                if sample_size is None:
                    logger.warning(f"  Failed to transcode sample {idx + 1}")
                    return None

                sample_sizes.append(sample_size)
                logger.info(
                    f"  Sample {idx + 1} size: {sample_size / 1024 / 1024:.2f} MB"
                )

        # Calculate average sample size and extrapolate
        avg_sample_size = sum(sample_sizes) / len(sample_sizes)
        estimated_size = (avg_sample_size / self.sample_duration) * duration

        logger.info(f"  Average sample size: {avg_sample_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Estimated full size: {estimated_size / 1024 / 1024:.2f} MB")

        return estimated_size

    def transcode_video(self, input_path: Path, output_path: Path) -> bool:
        """Transcode the full video"""
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-map",
                "0",
                "-c",
                "copy",
                "-c:v:0",  # only first video stream
                self.target_codec,
                "-crf",
                str(self.crf),
                str(output_path),
            ]

            logger.info(f"  Starting full transcode...")
            subprocess.run(cmd, check=True)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Transcoding failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise
        except SystemExit:
            logger.info("Removing Unfinished Video")
            if output_path.exists():
                output_path.unlink()
            raise

    def check_disk_space(self, required_bytes: int) -> bool:
        """Check if there's enough disk space"""
        stat = shutil.disk_usage(self.output_dir)
        available = stat.free

        # Add 10% buffer
        required_with_buffer = required_bytes * 1.1

        if available < required_with_buffer:
            logger.error(
                f"Insufficient disk space. Required: {required_with_buffer / 1024 / 1024 / 1024:.2f} GB, "
                f"Available: {available / 1024 / 1024 / 1024:.2f} GB"
            )
            return False
        return True

    def process_video(self, video_path: Path) -> None:
        """Process a single video file"""
        self.stats["total_files"] += 1

        logger.info(f"\nProcessing: {video_path}")

        # Get video info
        info = self.get_video_info(video_path)
        if info is None:
            logger.error(f"Could not read video info, skipping")
            self.stats["errors"] += 1
            return

        # Check codec
        codec = self.get_video_codec(info)
        if codec is None:
            logger.error(f"Could not determine video codec, skipping")
            self.stats["errors"] += 1
            return

        logger.info(f"  Current codec: {codec}")

        if codec not in self.source_codecs:
            logger.info(
                f"  Codec {codec} not in target list {self.source_codecs}, skipping"
            )
            self.stats["skipped_codec"] += 1
            return

        # Get duration
        duration = self.get_duration(info)
        if duration is None:
            logger.error(f"Could not determine video duration, skipping")
            self.stats["errors"] += 1
            return

        logger.info(f"  Duration: {duration:.1f}s ({duration / 60:.1f} minutes)")

        # Check if video is long enough for sampling
        if duration < self.sample_duration * 3:
            logger.warning(f"  Video too short for accurate sampling, skipping")
            self.stats["errors"] += 1
            return

        # Get current size
        current_size = video_path.stat().st_size
        logger.info(f"  Current size: {current_size / 1024 / 1024:.2f} MB")

        # Estimate transcoded size
        estimated_size = self.estimate_transcoded_size(video_path, duration)
        if estimated_size is None:
            logger.error(f"Could not estimate transcoded size, skipping")
            self.stats["errors"] += 1
            return

        # Calculate reduction percentage
        reduction = (current_size - estimated_size) / current_size
        logger.info(f"  Estimated reduction: {reduction * 100:.1f}%")

        if reduction < self.threshold:
            logger.info(
                f"  Reduction below threshold ({self.threshold * 100:.1f}%), skipping"
            )
            self.stats["skipped_size"] += 1
            return

        # Calculate relative path and output path
        rel_path = video_path.relative_to(self.input_dir)
        output_path = self.output_dir / rel_path

        logger.info(
            f"  Would save: {(current_size - estimated_size) / 1024 / 1024:.2f} MB"
        )

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would transcode to: {output_path}")
            self.stats["transcoded"] += 1
            self.stats["total_saved"] += current_size - estimated_size
            return

        # Check disk space
        if not self.check_disk_space(int(estimated_size)):
            logger.error(f"Not enough disk space, stopping")
            sys.exit(1)

        # Transcode the full video
        if self.transcode_video(video_path, output_path):
            actual_size = output_path.stat().st_size
            actual_reduction = (current_size - actual_size) / current_size
            logger.info(f"  ✓ Transcoded successfully!")
            logger.info(f"  Actual size: {actual_size / 1024 / 1024:.2f} MB")
            logger.info(f"  Actual reduction: {actual_reduction * 100:.1f}%")
            logger.info(f"  Saved: {(current_size - actual_size) / 1024 / 1024:.2f} MB")

            # remove the video if size is bigger than original
            if actual_size <= current_size:
                logger.info("Video is smaller then original. Removing Video")
                output_path.unlink()
            else:
                self.stats["transcoded"] += 1
                self.stats["total_saved"] += current_size - actual_size
        else:
            self.stats["errors"] += 1

    def process_directory(self) -> None:
        """Recursively process all video files in the input directory"""
        video_extensions = {
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".m4v",
            ".flv",
            ".wmv",
            ".webm",
        }

        video_files: list[Path] = []
        for ext in video_extensions:
            video_files.extend(self.input_dir.rglob(f"*{ext}"))
            video_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))

        video_files = sorted(set(video_files))

        logger.info(f"Found {len(video_files)} video files")

        for video_path in video_files:
            try:
                self.process_video(video_path)
            except Exception as e:
                logger.error(f"Unexpected error processing {video_path}: {e}")
                self.stats["errors"] += 1

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Smart video transcoder that estimates size reduction before transcoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dir", type=Path, help="Input directory containing videos"
    )

    parser.add_argument(
        "output_dir", type=Path, help="Output directory for transcoded videos"
    )

    parser.add_argument(
        "--source-codecs",
        nargs="+",
        default=["h264", "hevc"],
        help="Source codecs to transcode from (accepts aliases such as avc, h.264, libx264, etc.)",
    )

    parser.add_argument(
        "--target-codec",
        default="libx265",
        choices=["libx264", "libx265", "libvpx-vp9", "libaom-av1"],
        help="Target codec to transcode to",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="CRF value for encoding (lower = better quality, larger file)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Minimum size reduction threshold (0.15 = 15%%)",
    )

    parser.add_argument(
        "--sample-duration",
        type=int,
        default=240,
        help="Duration of each sample in seconds (default: 240 = 4 minutes)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually transcoding",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file or directory (default: transcode.log in current directory)",
    )

    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file
    if log_file is None:
        log_file = Path("transcode.log")
    elif log_file.is_dir():
        log_file = log_file / "transcode.log"

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging now that we know the log file location
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if ffmpeg and ffprobe are available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg and ffprobe must be installed and in PATH")
        sys.exit(1)

    # Create transcoder and process
    transcoder = VideoTranscoder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_codecs=args.source_codecs,
        target_codec=args.target_codec,
        crf=args.crf,
        threshold=args.threshold,
        sample_duration=args.sample_duration,
        dry_run=args.dry_run,
    )

    def signal_handler(signum, frame):
        transcoder.handle_signal(signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting video transcoding process")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Source codecs: {args.source_codecs}")
    logger.info(f"Target codec: {args.target_codec}")
    logger.info(f"CRF: {args.crf}")
    logger.info(f"Threshold: {args.threshold * 100}%")
    logger.info(f"Sample duration: {args.sample_duration}s")
    logger.info(f"Dry run: {args.dry_run}")

    transcoder.process_directory()


if __name__ == "__main__":
    main()
