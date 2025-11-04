#!/usr/bin/env python3
"""
Split EPIC-KITCHENS processed data by instruction index.

This script takes the processed data from data_1/ directory and reorganizes it
into individual directories per instruction (indexed by the data.json entries).

Input structure:
- data_1/data.json: List of instruction metadata
- data_1/rgb_frames/{video_id}/frame_*.jpg: RGB frames
- data_1/masks/{video_id}/{class_name}/frame_*.png: Segmentation masks

Output structure:
- data/images/{IDX}/N.png: RGB frames for instruction IDX (N = 1, 2, 3, ...)
- data/masks/{IDX}/N.png: Masks for instruction IDX (N = 1, 2, 3, ...)
- data/{IDX}/data.json: Metadata for instruction IDX
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2


def extract_frame_number(frame_name):
    """
    Extract frame number from filename.

    Args:
        frame_name: Frame filename (e.g., 'P01_01_frame_0000000140.jpg')

    Returns:
        Frame number as integer
    """
    # Extract frame number from pattern: {video_id}_frame_{frame_num}.jpg
    parts = frame_name.split("_frame_")
    if len(parts) == 2:
        frame_num_str = parts[1].split(".")[0]
        return int(frame_num_str)
    return None


def process_instruction(
    instruction_data,
    idx,
    source_rgb_dir,
    source_masks_dir,
    output_base_dir,
    mask_classes=None,
):
    """
    Process a single instruction entry and copy relevant files.

    Args:
        instruction_data: Dictionary containing instruction metadata
        idx: Index of this instruction in the data.json
        source_rgb_dir: Source directory for RGB frames
        source_masks_dir: Source directory for masks
        output_base_dir: Base output directory
        mask_classes: List of mask class names to include (None = all classes)

    Returns:
        Tuple of (copied_frames, copied_masks, instruction_data_or_none)
        instruction_data_or_none is the instruction_data if successful, None otherwise
    """
    video_id = instruction_data["video_id"]
    start_frame = instruction_data["start_frame"]
    stop_frame = instruction_data["stop_frame"]

    # Collect frames in order
    frame_data = []  # List of (frame_num, rgb_path, mask_paths)

    # Process RGB frames
    video_rgb_dir = os.path.join(source_rgb_dir, video_id)

    if os.path.exists(video_rgb_dir):
        rgb_files = sorted(os.listdir(video_rgb_dir))

        for rgb_file in rgb_files:
            if not rgb_file.endswith(".jpg"):
                continue

            frame_num = extract_frame_number(rgb_file)
            if frame_num is None:
                continue

            # Check if frame is within instruction range
            if start_frame <= frame_num <= stop_frame:
                src_path = os.path.join(video_rgb_dir, rgb_file)
                frame_data.append((frame_num, src_path, []))

    # Process masks and match with frames
    video_masks_dir = os.path.join(source_masks_dir, video_id)
    if os.path.exists(video_masks_dir):
        # Get all class directories
        class_dirs = [
            d
            for d in os.listdir(video_masks_dir)
            if os.path.isdir(os.path.join(video_masks_dir, d))
        ]

        for class_name in class_dirs:
            # Filter by mask_classes if specified
            if mask_classes is not None and class_name not in mask_classes:
                continue

            class_masks_src = os.path.join(video_masks_dir, class_name)
            mask_files = sorted(os.listdir(class_masks_src))

            for mask_file in mask_files:
                if not mask_file.endswith(".png"):
                    continue

                frame_num = extract_frame_number(mask_file)
                if frame_num is None:
                    continue

                # Check if frame is within instruction range
                if start_frame <= frame_num <= stop_frame:
                    src_path = os.path.join(class_masks_src, mask_file)

                    # Find matching frame in frame_data or add new entry
                    found = False
                    for i, (fnum, rgb_path, mask_paths) in enumerate(frame_data):
                        if fnum == frame_num:
                            mask_paths.append(src_path)
                            found = True
                            break

                    if not found:
                        # Add frame entry with no RGB (mask only)
                        frame_data.append((frame_num, None, [src_path]))

    # Sort by frame number
    frame_data.sort(key=lambda x: x[0])

    # Check if there's any data to save
    if not frame_data:
        return 0, 0, None

    # Prepare output directories (but don't create yet)
    output_images_dir = os.path.join(output_base_dir, "images", str(idx))
    output_masks_dir = os.path.join(output_base_dir, "masks", str(idx))

    # Save with sequential numbering (1-indexed)
    copied_frames = 0
    copied_masks = 0

    for n, (frame_num, rgb_path, mask_paths) in enumerate(frame_data, start=1):
        # Save RGB frame
        if rgb_path is not None:
            img = cv2.imread(rgb_path)
            if img is not None:
                os.makedirs(output_images_dir, exist_ok=True)
                dst_path = os.path.join(output_images_dir, f"{n}.png")
                cv2.imwrite(dst_path, img)
                copied_frames += 1

        # Save masks (merge all masks into single image)
        if mask_paths:
            # Read first mask to get dimensions
            first_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
            if first_mask is not None:
                os.makedirs(output_masks_dir, exist_ok=True)
                # If multiple masks, merge them
                merged_mask = first_mask.copy()
                for mask_path in mask_paths[1:]:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        merged_mask = cv2.bitwise_or(merged_mask, mask)

                dst_path = os.path.join(output_masks_dir, f"{n}.png")
                cv2.imwrite(dst_path, merged_mask)
                copied_masks += 1

    # Return instruction data only if we actually copied frames or masks
    if copied_frames > 0 or copied_masks > 0:
        return copied_frames, copied_masks, instruction_data
    else:
        return 0, 0, None


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Split EPIC-KITCHENS data by instruction index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data_1",
        help="Input directory containing processed data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_2",
        help="Output directory for split data",
    )

    parser.add_argument(
        "--mask-classes",
        type=str,
        nargs="+",
        default="right hand",
        help='Specific mask classes to include (e.g., "right_hand" "left_hand"). If not specified, all classes are included.',
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start processing from this instruction index",
    )

    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="End processing at this instruction index (exclusive). -1 means process all.",
    )

    args = parser.parse_args()

    # Load instruction data
    # data_json_path = os.path.join(args.input_dir, "data.json")
    data_json_path = "data.json"
    with open(data_json_path, "r") as f:
        instructions = json.load(f)

    # Determine range to process
    start_idx = args.start_idx
    end_idx = len(instructions) if args.end_idx == -1 else args.end_idx
    end_idx = min(end_idx, len(instructions))

    # Setup directories
    source_rgb_dir = os.path.join(args.input_dir, "rgb_frames")
    source_masks_dir = os.path.join(args.input_dir, "masks")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("EPIC-KITCHENS Instruction Data Splitter")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total instructions: {len(instructions)}")
    print(f"Processing range: {start_idx} to {end_idx}")
    if args.mask_classes:
        print(f"Mask classes filter: {', '.join(args.mask_classes)}")
    else:
        print("Mask classes filter: All classes")
    print("=" * 60 + "\n")

    # Process each instruction
    total_frames = 0
    total_masks = 0
    skipped_count = 0
    successful_annotations = []

    for idx in tqdm(range(start_idx, end_idx), desc="Processing instructions"):
        instruction_data = instructions[idx]

        try:
            frames, masks, annotation = process_instruction(
                instruction_data,
                idx,
                source_rgb_dir,
                source_masks_dir,
                args.output_dir,
                mask_classes=args.mask_classes,
            )
            total_frames += frames
            total_masks += masks

            if annotation is not None:
                # Add index to annotation
                annotation["index"] = idx
                successful_annotations.append(annotation)
            else:
                skipped_count += 1

        except Exception as e:
            print(f"\nError processing instruction {idx}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Save annotations.json with all successful instructions
    if successful_annotations:
        annotations_path = os.path.join(args.output_dir, "annotations.json")
        with open(annotations_path, "w") as f:
            json.dump(successful_annotations, f, indent=2)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"Data saved to: {args.output_dir}")
    print(f"Total RGB frames copied: {total_frames}")
    print(f"Total mask images copied: {total_masks}")
    print(f"Successful instructions: {len(successful_annotations)}")
    print(f"Instructions with no data: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
