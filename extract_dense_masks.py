import json
import os
import zipfile
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# Setup directories
data_dir = "data"
masks_dir = os.path.join(data_dir, "masks_dense")
os.makedirs(masks_dir, exist_ok=True)

# Setup annotation directories for Interpolations (Dense)
interpolations_dir = "2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations"
mode = "train"
interpolations_train_dir = os.path.join(interpolations_dir, mode)

# Temporary extraction directory
temp_extract_dir = "data/temp_interpolations"
os.makedirs(temp_extract_dir, exist_ok=True)


def generate_mask_from_annotations(masks_info, input_resolution=(854, 480)):
    """
    Generate a mask image from annotation polygons.
    Each object instance gets a unique pixel value (1, 2, 3, ...).
    Note: Interpolations use 854x480 resolution (not 1920x1080)
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


def process_video_interpolations(interpolation_zip_path, video_id):
    """
    Process interpolation annotations: extract and generate dense masks.
    """
    print(f"Extracting interpolation annotations for {video_id}...")

    # Extract JSON from zip
    video_temp_json = os.path.join(temp_extract_dir, f"{video_id}_interpolations.json")
    with zipfile.ZipFile(interpolation_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # Find the extracted JSON file
    json_file = os.path.join(temp_extract_dir, f"{video_id}_interpolations.json")
    if not os.path.exists(json_file):
        print(f"Warning: JSON not found at {json_file}")
        return None

    # Load annotation JSON
    print(f"Loading annotations for {video_id}...")
    with open(json_file, "r") as f:
        annotation_data = json.load(f)

    video_annotations = annotation_data["video_annotations"]
    print(f"Found {len(video_annotations)} annotated frames for {video_id}")

    # Determine input resolution (interpolations are 854x480)
    input_resolution = (854, 480)

    # Create output directory for this video's masks
    video_masks_dir = os.path.join(masks_dir, video_id)
    os.makedirs(video_masks_dir, exist_ok=True)

    # Process each annotated frame
    print(f"Generating masks for {len(video_annotations)} frames...")
    mask_count = 0

    for datapoint in tqdm(video_annotations, desc=f"Processing {video_id}"):
        image_name = datapoint["image"]["name"]
        masks_info = datapoint["annotations"]

        # Generate mask from annotations
        mask = generate_mask_from_annotations(masks_info, input_resolution)

        # Only save mask if it contains annotations
        if not np.all(mask == 0):
            # Convert image name to PNG if needed
            mask_name = image_name.replace(".jpg", ".png") if image_name.endswith(".jpg") else image_name
            if not mask_name.endswith(".png"):
                mask_name = mask_name.rsplit(".", 1)[0] + ".png"

            output_mask_path = os.path.join(video_masks_dir, mask_name)

            # Save mask as PNG
            cv2.imwrite(output_mask_path, mask)
            mask_count += 1

    # Clean up extracted JSON
    if os.path.exists(json_file):
        os.remove(json_file)

    print(f"Completed processing {video_id}: saved {mask_count} masks")

    return {
        "video_id": video_id,
        "num_frames": len(video_annotations),
        "num_masks": mask_count,
        "masks_dir": video_masks_dir
    }


# Main processing loop
interpolation_zip_files = sorted([f for f in os.listdir(interpolations_train_dir)
                                  if f.endswith("_interpolations.zip")])

print(f"Found {len(interpolation_zip_files)} videos to process")
print("="*60)

results = []
for zip_file in interpolation_zip_files:
    video_id = zip_file.replace("_interpolations.zip", "")
    interpolation_zip_path = os.path.join(interpolations_train_dir, zip_file)

    print(f"\n{'='*60}")
    print(f"Processing video: {video_id}")
    print(f"{'='*60}")

    try:
        result = process_video_interpolations(interpolation_zip_path, video_id)
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error processing {video_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Clean up temporary directory
if os.path.exists(temp_extract_dir):
    shutil.rmtree(temp_extract_dir)

# Summary
print("\n" + "="*60)
print("Processing Complete!")
print("="*60)
total_frames = sum(r["num_frames"] for r in results)
total_masks = sum(r["num_masks"] for r in results)
print(f"Processed {len(results)} videos")
print(f"Total frames: {total_frames:,}")
print(f"Total masks saved: {total_masks:,}")
print(f"Masks saved to: {masks_dir}")
print("="*60)
