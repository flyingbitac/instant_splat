"""Cut a video into frames at fixed time intervals."""

import argparse
import cv2
import os
from pathlib import Path


def cut_video_to_frames(video_path: str, output_dir: str, interval_sec: float = 1.0):
    """Extract frames from video at fixed time intervals.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval_sec: Time interval between frames in seconds
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Cannot determine video FPS")

    frame_interval = int(fps * interval_sec)
    frame_count = 0
    saved_count = 0

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Interval: {interval_sec}s ({frame_interval} frames)")
    print(f"Output: {output_dir}")
    print("-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="Cut video into frames at fixed intervals")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("-o", "--output", default="frames", help="Output directory (default: frames)")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Time interval in seconds (default: 1.0)")

    args = parser.parse_args()
    cut_video_to_frames(args.video, args.output, args.interval)


if __name__ == "__main__":
    main()