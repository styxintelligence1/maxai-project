import os
import shutil
from datetime import datetime
from pathlib import Path

# === CONFIG ===
# Where to put all organised runs on your laptop
RUNS_ROOT = Path("C:/cow_runs")

# Video extensions we care about
VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".360",
    ".MP4", ".MOV", ".M4V", ".AVI", ".MKV"
}


def find_sd_card_videos() -> list[Path]:
    """Scan all drives D: through Z: for /DCIM and collect all video files."""
    sd_roots = [Path(f"{drive}:/DCIM") for drive in "DEFGHIJKLMNOPQRSTUVWXYZ"]

    videos: list[Path] = []
    found_roots: list[Path] = []

    for sd_root in sd_roots:
        if not sd_root.exists():
            continue

        found_roots.append(sd_root)

        for path in sd_root.rglob("*"):
            if path.is_file() and path.suffix in VIDEO_EXTS:
                videos.append(path)

    if not found_roots:
        print("No SD card found (no DCIM folder on any drive).")
        return []

    print("Scanning these DCIM folders:")
    for r in found_roots:
        print(f"  - {r}")

    return videos


def create_run_folder() -> Path:
    """Create a new timestamped run folder on C:."""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = RUNS_ROOT / ts
    run_folder.mkdir()
    return run_folder


def copy_videos_to_run(videos: list[Path], run_folder: Path):
    """Copy all videos into the new run folder."""
    for src in videos:
        dest = run_folder / src.name
        print(f"Copying {src} -> {dest}")
        shutil.copy2(src, dest)
    print("All videos copied.")


def main():
    print("=== Cow Run Importer v1 ===")
    print("Looking for video files on any SD card (D:–Z: /DCIM)...")

    videos = find_sd_card_videos()

    if not videos:
        print("No video files found on any SD card.")
        return

    print(f"\nFound {len(videos)} video(s) on SD card(s).")

    run_folder = create_run_folder()
    print(f"Created run folder: {run_folder}")

    copy_videos_to_run(videos, run_folder)

    print("\n✅ Done. Your videos are now in:")
    print(run_folder)
    print("\nNext step will be: run YOLOv8x TensorRT over this folder automatically.")


if __name__ == "__main__":
    main()
