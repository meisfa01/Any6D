import os
import re
import shutil
import argparse

def extract_timestamp(filename):
    """
    Extracts timestamp (float) from filenames like:
    _Depth_1762774446980.69213867187500.png
    """
    match = re.search(r"_(\d+\.\d+)\.png$", filename)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Sort RGB and Depth images chronologically and rename them.")
    parser.add_argument("input_folder", help="Input folder containing images.")
    parser.add_argument("output_folder", help="Output folder for sorted images.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    depth_folder = os.path.join(output_folder, "depth")
    rgb_folder = os.path.join(output_folder, "rgb")
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)

    depth_files = []
    rgb_files = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".png"):
            continue
        if "metadata" in filename.lower():
            continue

        ts = extract_timestamp(filename)
        if ts is None:
            continue

        path = os.path.join(input_folder, filename)
        if "depth" in filename.lower():
            depth_files.append((ts, path))
        elif "color" in filename.lower():
            rgb_files.append((ts, path))

    depth_files.sort(key=lambda x: x[0])
    rgb_files.sort(key=lambda x: x[0])

    for i, (_, src) in enumerate(depth_files):
        dst = os.path.join(depth_folder, f"{i:04d}.png")
        shutil.copy2(src, dst)

    for i, (_, src) in enumerate(rgb_files):
        dst = os.path.join(rgb_folder, f"{i:04d}.png")
        shutil.copy2(src, dst)

    print("Done")
    print(f"  Depth images: {len(depth_files)}")
    print(f"  RGB images:   {len(rgb_files)}")
    print(f"Output saved in: {output_folder}")

if __name__ == "__main__":
    main()
