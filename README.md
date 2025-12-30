# smart-video-reencoder

A simple Python CLI tool that intelligently decides whether a video should be re-encoded by estimating potential size reduction before doing a full encode. This avoids wasting time and CPU on videos that would not benefit meaningfully from re-encoding.

## Features

- Recursive directory scanning
- Sample-based size estimation
- Threshold-based decision making (e.g. only re-encode if savings ≥ 15%)
- Supports multiple source codecs
- Configurable target codec and CRF
- Dry-run mode
- Logging to file
- Single-file script with no Python dependencies (uses ffmpeg)

## Usage

```bash
./smart_video_reencode.py input_dir output_dir [options]
```

### Example

```bash
./smart_video_reencode.py input/dir output/dir \
  --source-codecs h264 hevc \
  --target-codec libx265 \
  --crf 23 \
  --threshold 0.15 \
  --sample-duration 240 \
  --dry-run \
  --log-file /path/to/transcode.log
```

### Options

```bash
-h, --help          show help message and exit
--source-codecs     Source codecs to re-encode from
--target-codec      Target codec (default: libx265)
--crf               CRF value (lower = higher quality)
--threshold         Minimum size reduction required (0.15 = 15%)
--sample-duration   Length of each sample in seconds
--dry-run           Estimate only, do not re-encode
--log-file          Log file path
```

### Requirements

- Python 3
- ffmpeg available in PATH

No third-party Python libraries are required.

### Limitations

- CPU-only encoding (no GPU acceleration)
