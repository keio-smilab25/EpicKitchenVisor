#!/usr/bin/env python3
"""
EPIC-KITCHENS Dataset Processor

This script processes EPIC-KITCHENS sparse annotation data and extracts:
- RGB frames as JPG images
- Instance segmentation masks as PNG images (one file per instance)

Each class is stored in a separate directory, and multiple instances of the
same class in a frame are saved as separate mask files.
"""

import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def generate_masks_by_class(masks_info, input_resolution=(1920, 1080)):
    """
    Generate mask images grouped by class name.

    Args:
        masks_info: List of annotation dictionaries containing segments and class names
        input_resolution: Tuple of (width, height) for the output mask

    Returns:
        Dictionary mapping class_name to list of mask images (one per instance)
        Each mask has pixel value 255 for the object and 0 for background
    """
    class_masks = defaultdict(list)

    # Group annotations by class name
    for entity in masks_info:
        class_name = entity.get("name", "unknown")
        class_masks[class_name].append(entity)

    # Generate separate mask for each instance
    masks = {}
    for class_name, entities in class_masks.items():
        instance_masks = []

        for entity in entities:
            # Create a new mask for this instance
            img = np.zeros([input_resolution[1], input_resolution[0]], dtype=np.uint8)

            object_annotations = entity["segments"]
            polygons = [object_annotations]

            ps = []
            for polygon in polygons:
                for poly in polygon:
                    if poly == []:
                        poly = [[0.0, 0.0]]
                    ps.append(np.array(poly, dtype=np.int32))

            # Fill polygon with 255 (white) for visibility
            cv2.fillPoly(img, ps, 255)

            instance_masks.append(img)

        masks[class_name] = instance_masks

    return masks


def process_video(
    annotation_file_path,
    video_id,
    source_rgb_frames_dir,
    output_rgb_frames_dir,
    masks_dir,
    temp_extract_dir,
):
    """
    Process a single video: extract RGB frames and generate masks.

    Args:
        annotation_file_path: Path to the JSON annotation file
        video_id: Video identifier (e.g., 'P01_01')
        source_rgb_frames_dir: Directory containing source RGB frame zip files
        output_rgb_frames_dir: Output directory for RGB frames
        masks_dir: Output directory for masks
        temp_extract_dir: Temporary directory for extraction
    """
    # Load annotation JSON
    with open(annotation_file_path, "r") as f:
        annotation_data = json.load(f)

    video_annotations = annotation_data["video_annotations"]

    # Determine input resolution (sparse annotations are 1920x1080)
    input_resolution = (1920, 1080)

    # Get RGB frames zip path
    participant_id = video_id.split("_")[0]
    rgb_frames_zip_path = os.path.join(
        source_rgb_frames_dir, participant_id, f"{video_id}.zip"
    )

    if not os.path.exists(rgb_frames_zip_path):
        print(f"Warning: RGB frames not found for {video_id}: {rgb_frames_zip_path}")
        return

    # Create output directories for this video
    video_rgb_frames_dir = os.path.join(output_rgb_frames_dir, video_id)
    video_masks_dir = os.path.join(masks_dir, video_id)
    os.makedirs(video_rgb_frames_dir, exist_ok=True)
    os.makedirs(video_masks_dir, exist_ok=True)

    # Extract RGB frames to temporary directory
    video_temp_dir = os.path.join(temp_extract_dir, video_id)
    if os.path.exists(video_temp_dir):
        shutil.rmtree(video_temp_dir)

    print(f"Extracting RGB frames for {video_id}...")
    with zipfile.ZipFile(rgb_frames_zip_path, "r") as zip_ref:
        zip_ref.extractall(video_temp_dir)

    # Process each annotated frame
    print(f"Processing {len(video_annotations)} frames for {video_id}...")
    for datapoint in tqdm(video_annotations, desc=f"Processing {video_id}"):
        image_name = datapoint["image"]["name"]
        masks_info = datapoint["annotations"]

        # Find the RGB frame file
        rgb_frame_path = os.path.join(video_temp_dir, image_name)

        # Some zip files might have a nested structure, try to find the file
        if not os.path.exists(rgb_frame_path):
            for root, dirs, files in os.walk(video_temp_dir):
                if image_name in files:
                    rgb_frame_path = os.path.join(root, image_name)
                    break

        if not os.path.exists(rgb_frame_path):
            print(f"Warning: RGB frame not found: {rgb_frame_path}")
            continue

        # Copy RGB image to output directory
        output_image_path = os.path.join(video_rgb_frames_dir, image_name)
        shutil.copy(rgb_frame_path, output_image_path)

        # Generate masks grouped by class
        class_masks = generate_masks_by_class(masks_info, input_resolution)

        # Save each class mask in its respective directory
        mask_name_base = image_name.replace(".jpg", "")
        for class_name, instance_masks in class_masks.items():
            # Create class directory if it doesn't exist
            class_mask_dir = os.path.join(video_masks_dir, class_name)
            os.makedirs(class_mask_dir, exist_ok=True)

            # Save each instance as a separate file
            for instance_idx, mask in enumerate(instance_masks):
                # Only save mask if it contains annotations
                if not np.all(mask == 0):
                    # Generate filename with instance index
                    if len(instance_masks) == 1:
                        # Single instance: no index suffix
                        mask_name = f"{mask_name_base}.png"
                    else:
                        # Multiple instances: add index suffix
                        mask_name = f"{mask_name_base}_instance_{instance_idx}.png"

                    output_mask_path = os.path.join(class_mask_dir, mask_name)
                    cv2.imwrite(output_mask_path, mask)

    # Clean up temporary extraction
    shutil.rmtree(video_temp_dir)
    print(f"Completed processing {video_id}")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Process EPIC-KITCHENS sparse annotations and extract RGB frames with masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--annotation-root",
        type=str,
        default="2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations",
        help="Root directory of sparse annotations",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset mode (train or test)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--n-videos",
        type=int,
        default=10,
        help="Number of videos to process (set to -1 for all videos)",
    )

    parser.add_argument(
        "--temp-dir",
        type=str,
        default="data/temp_extract",
        help="Temporary directory for extraction",
    )

    args = parser.parse_args()

    # Setup directories
    output_rgb_frames_dir = os.path.join(args.output_dir, "rgb_frames")
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(output_rgb_frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Setup annotation and RGB frame directories (source data)
    annotation_dir = os.path.join(args.annotation_root, "annotations", args.mode)
    source_rgb_frames_dir = os.path.join(args.annotation_root, "rgb_frames", args.mode)

    # Get list of annotation files
    annotation_file_list = sorted(os.listdir(annotation_dir))
    json_files = [f for f in annotation_file_list if f.endswith(".json")]

    # Determine files to process
    if args.n_videos == -1:
        files_to_process = json_files
    else:
        files_to_process = json_files[: args.n_videos]

    print("\n" + "=" * 60)
    print("EPIC-KITCHENS Dataset Processor")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Processing {len(files_to_process)} out of {len(json_files)} total videos")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60 + "\n")

    # Process each video
    for annotation_file in files_to_process:
        video_id = annotation_file.replace(".json", "")
        annotation_file_path = os.path.join(annotation_dir, annotation_file)

        print(f"\n{'='*60}")
        print(f"Processing video: {video_id}")
        print(f"{'='*60}")

        try:
            process_video(
                annotation_file_path,
                video_id,
                source_rgb_frames_dir,
                output_rgb_frames_dir,
                masks_dir,
                args.temp_dir,
            )
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Clean up temporary directory
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"RGB frames saved to: {output_rgb_frames_dir}")
    print(f"Masks saved to: {masks_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
