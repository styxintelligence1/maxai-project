"""
Stitch GoPro 360 clips into a single equirectangular video per milking/session.
"""
import pathlib

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    print("[INFO] Stitch script stub, root:", root)

if __name__ == "__main__":
    main()
