import json
import os
import zipfile
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Setup directories
data_dir = "data"
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Setup annotation and RGB frame directories
annotation_dir_root = "2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations"
annotation_dir = os.path.join(annotation_dir_root, "annotations")
rgb_frames_dir = os.path.join(annotation_dir_root, "rgb_frames")
mode = "train"

annotation_dir = os.path.join(annotation_dir, mode)
rgb_frames_dir = os.path.join(rgb_frames_dir, mode)

# Temporary extraction directory
temp_extract_dir = "data/temp_extract"
os.makedirs(temp_extract_dir, exist_ok=True)


def generate_mask_from_annotations(masks_info, input_resolution=(1920, 1080)):
    """
    Generate a mask image from annotation polygons.
    Each object instance gets a unique pixel value (1, 2, 3, ...).
    """
    img = np.zeros([input_resolution[1], input_resolution[0]], dtype=np.uint8)

    for i, entity in enumerate(masks_info, start=1):
        object_annotations = entity["segments"]
        polygons = []
        polygons.append(object_annotations)

        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))

        # Fill polygon with unique instance ID
        cv2.fillPoly(img, ps, (i, i, i))

    return img


def process_video(annotation_file_path, video_id):
    """
    Process a single video: extract images and masks.
    """
    # Load annotation JSON
    with open(annotation_file_path, "r") as f:
        annotation_data = json.load(f)

    video_annotations = annotation_data["video_annotations"]

    # Determine input resolution (sparse annotations are 1920x1080)
    input_resolution = (1920, 1080)

    # Get RGB frames zip path
    participant_id = video_id.split("_")[0]
    rgb_frames_zip_path = os.path.join(rgb_frames_dir, participant_id, f"{video_id}.zip")

    if not os.path.exists(rgb_frames_zip_path):
        print(f"Warning: RGB frames not found for {video_id}: {rgb_frames_zip_path}")
        return

    # Create output directories for this video
    video_images_dir = os.path.join(images_dir, video_id)
    video_masks_dir = os.path.join(masks_dir, video_id)
    os.makedirs(video_images_dir, exist_ok=True)
    os.makedirs(video_masks_dir, exist_ok=True)

    # Extract RGB frames to temporary directory
    video_temp_dir = os.path.join(temp_extract_dir, video_id)
    if os.path.exists(video_temp_dir):
        shutil.rmtree(video_temp_dir)

    print(f"Extracting RGB frames for {video_id}...")
    with zipfile.ZipFile(rgb_frames_zip_path, 'r') as zip_ref:
        zip_ref.extractall(video_temp_dir)

    # Process each annotated frame
    print(f"Processing {len(video_annotations)} frames for {video_id}...")
    for datapoint in tqdm(video_annotations, desc=f"Processing {video_id}"):
        image_name = datapoint["image"]["name"]
        image_path = datapoint["image"]["image_path"]
        masks_info = datapoint["annotations"]

        # Find the RGB frame file
        # image_path format: "P01_01/frame_0000000000.jpg"
        rgb_frame_path = os.path.join(video_temp_dir, image_name)

        # Some zip files might have a nested structure, try to find the file
        if not os.path.exists(rgb_frame_path):
            # Search for the file in the extracted directory
            for root, dirs, files in os.walk(video_temp_dir):
                if image_name in files:
                    rgb_frame_path = os.path.join(root, image_name)
                    break

        if not os.path.exists(rgb_frame_path):
            print(f"Warning: RGB frame not found: {rgb_frame_path}")
            continue

        # Copy RGB image to output directory
        output_image_path = os.path.join(video_images_dir, image_name)
        shutil.copy(rgb_frame_path, output_image_path)

        # Generate and save mask
        mask = generate_mask_from_annotations(masks_info, input_resolution)

        # Only save mask if it contains annotations
        if not np.all(mask == 0):
            mask_name = image_name.replace(".jpg", ".png")
            output_mask_path = os.path.join(video_masks_dir, mask_name)

            # Save mask as PNG
            cv2.imwrite(output_mask_path, mask)

    # Clean up temporary extraction
    shutil.rmtree(video_temp_dir)
    print(f"Completed processing {video_id}")

    # Return statistics
    return {
        "video_id": video_id,
        "num_frames": len(video_annotations),
        "images_dir": video_images_dir,
        "masks_dir": video_masks_dir
    }


# Test with just one video - P01_01
print("Testing with one video: P01_01")
annotation_file = "P01_01.json"
video_id = "P01_01"
annotation_file_path = os.path.join(annotation_dir, annotation_file)

if os.path.exists(annotation_file_path):
    try:
        result = process_video(annotation_file_path, video_id)
        print("\n" + "="*60)
        print("Test completed successfully!")
        print(f"Video ID: {result['video_id']}")
        print(f"Number of frames: {result['num_frames']}")
        print(f"Images saved to: {result['images_dir']}")
        print(f"Masks saved to: {result['masks_dir']}")
        print("="*60)

        # Check what was created
        num_images = len(os.listdir(result['images_dir']))
        num_masks = len(os.listdir(result['masks_dir']))
        print(f"\nFiles created:")
        print(f"  Images: {num_images}")
        print(f"  Masks: {num_masks}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print(f"Annotation file not found: {annotation_file_path}")

# Clean up temporary directory
if os.path.exists(temp_extract_dir):
    shutil.rmtree(temp_extract_dir)
