import json
import os
import zipfile
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# Setup directories
data_dir = "data"
masks_dir = os.path.join(data_dir, "masks_dense_test")
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


# Test with one video
test_video = "P01_01"
zip_file = f"{test_video}_interpolations.zip"
interpolation_zip_path = os.path.join(interpolations_train_dir, zip_file)

print(f"Testing with video: {test_video}")
print("="*60)

if os.path.exists(interpolation_zip_path):
    try:
        result = process_video_interpolations(interpolation_zip_path, test_video)

        if result:
            print("\n" + "="*60)
            print("Test completed successfully!")
            print(f"Video ID: {result['video_id']}")
            print(f"Number of frames: {result['num_frames']:,}")
            print(f"Number of masks saved: {result['num_masks']:,}")
            print(f"Masks saved to: {result['masks_dir']}")
            print("="*60)

            # Check some mask properties
            mask_files = os.listdir(result['masks_dir'])
            if mask_files:
                sample_mask = cv2.imread(os.path.join(result['masks_dir'], mask_files[0]), cv2.IMREAD_GRAYSCALE)
                print(f"\nSample mask info:")
                print(f"  Shape: {sample_mask.shape}")
                print(f"  Unique values: {np.unique(sample_mask)}")
                print(f"  Data type: {sample_mask.dtype}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print(f"Interpolation zip not found: {interpolation_zip_path}")

# Clean up temporary directory
if os.path.exists(temp_extract_dir):
    shutil.rmtree(temp_extract_dir)
