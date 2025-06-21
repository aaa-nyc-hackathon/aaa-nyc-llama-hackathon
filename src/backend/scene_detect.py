from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import os
import cv2
import subprocess
import shutil  # Import shutil for directory removal


def trim_clips_except_first(input_dir, output_dir=None, trim_percent=0.05):
    """
    Trim off the first trim_percent (e.g., 5%) of each video clip in the input directory,
    except for the first one.

    Args:
        input_dir: Directory containing the video clips
        output_dir: Directory to save the trimmed clips (defaults to input_dir if None)
        trim_percent: Percentage of the clip to trim from the beginning (0.05 = 5%)
    """
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    # List clips in the directory
    clips = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi", ".mkv"))]
    clips.sort()  # Sort to ensure first clip is processed first

    print(f"Found {len(clips)} clips to process")

    # Process each clip
    for i, clip_name in enumerate(clips):
        input_path = os.path.join(input_dir, clip_name)
        output_path = os.path.join(output_dir, f"trimmed_{clip_name}")
        # Skip trimming the first clip
        if i == 0:
            print(f"Skipping trim for first clip: {clip_name}")
            # If output dir is different from input dir, just copy the first clip
            if output_dir != input_dir:
                print(f"Copying first clip to output directory: {output_path}")
                try:
                    # Use ffmpeg to copy the file instead of direct file copying
                    # This ensures the output is valid and properly finalized
                    ffmpeg_copy_cmd = [
                        "ffmpeg",
                        "-i",
                        input_path,
                        "-c",
                        "copy",
                        "-y",
                        output_path,
                    ]
                    subprocess.run(
                        ffmpeg_copy_cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f"  Successfully copied to: {output_path}")
                except Exception as e:
                    print(f"  Error copying clip: {e}")
            continue

        # Get clip duration
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening clip: {input_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # Calculate time to trim
        trim_seconds = duration * trim_percent
        start_time = trim_seconds

        print(f"Processing clip {i + 1}/{len(clips)}: {clip_name}")
        print(f"  Duration: {duration:.2f}s, Trimming: {trim_seconds:.2f}s")
        # Use FFmpeg to trim the clip - using re-encoding instead of stream copy
        # The -c copy method can cause corruption when cutting at non-keyframes
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-c:v",
            "libx264",  # Re-encode video
            "-c:a",
            "aac",  # Re-encode audio
            "-preset",
            "fast",  # Speed up encoding
            "-y",  # Overwrite output file if it exists
            output_path,
        ]

        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # Verify the output file exists and has a valid size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"  Successfully trimmed and saved to: {output_path}")
            else:
                print(
                    "  Warning: Output file may be corrupted (small size or doesn't exist)."
                )
                # Try again with different parameters
                ffmpeg_cmd_retry = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-ss",
                    str(start_time),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    "-y",
                    output_path,
                ]
                print("  Retrying with different encoding parameters...")
                subprocess.run(
                    ffmpeg_cmd_retry,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except subprocess.CalledProcessError as e:
            print(f"  Error trimming clip: {e}")
            error_output = e.stderr.decode("utf-8") if e.stderr else "No error output"
            print(f"  FFmpeg error: {error_output}")


def split_camera_moves(input_video_path, output_dir="video_segments", threshold=1.5):
    """
    Splits a video into segments based on camera movements and scene changes.
    Args:
        input_video_path: Path to the input video file
        output_folder: Folder to save the output segments
        threshold: Threshold for camera movement detection
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file does not exist: {input_video_path}")

    temp_output_dir = os.path.join(os.getcwd(), "temp_scenes")
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(os.getcwd(), output_dir))
    try:
        scene_list = detect(
            input_video_path, AdaptiveDetector(threshold), show_progress=True
        )
        split_video_ffmpeg(
            input_video_path, scene_list, output_dir=temp_output_dir, show_progress=True
        )
        trim_clips_except_first(
            input_dir=temp_output_dir, output_dir=output_dir, trim_percent=0.08
        )
    except Exception as e:
        print(f"Error processing video: {e}")

    # Use shutil.rmtree to remove directory and all its contents
    try:
        shutil.rmtree(temp_output_dir)
        print(f"Successfully removed temporary directory: {temp_output_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temporary directory: {e}")

    # list of video paths
    video_paths = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith((".mp4"))
    ]
    return video_paths
