import json
import numpy as np
import os
from tqdm import tqdm
import cv2


def find_instruction_for_frame(epic_annotations_by_video, video_id, frame_number):
    """
    Find the EPIC-KITCHENS instruction annotation that contains the given frame.

    Args:
        epic_annotations_by_video: Dictionary mapping video_id to list of annotations
        video_id: Video ID to search in
        frame_number: Frame number to find

    Returns:
        Annotation dictionary if found, None otherwise
    """
    annotations_for_video = epic_annotations_by_video.get(video_id, [])

    for annotation in annotations_for_video:
        start_frame = annotation["start_frame"]
        stop_frame = annotation["stop_frame"]

        if start_frame <= frame_number <= stop_frame:
            return annotation

    return None


def build_epic_annotations_by_video(epic_annotations_list):
    """
    Build a dictionary mapping video_id to list of annotations for faster lookup.

    Args:
        epic_annotations_list: List of EPIC-KITCHENS annotation dictionaries

    Returns:
        Dictionary with video_id as key and list of annotations as value
    """
    epic_annotations_by_video = {}

    for list_index, annotation in enumerate(
        tqdm(epic_annotations_list, desc="Building EPIC annotations dict")
    ):
        # Add original index for later reference
        annotation["original_index"] = list_index

        # Initialize sparse_mask_path field
        epic_annotations_list[list_index]["sparse_mask_path"] = []

        # Group annotations by video_id
        video_id = annotation["video_id"]
        if video_id in epic_annotations_by_video:
            epic_annotations_by_video[video_id].append(annotation)
        else:
            epic_annotations_by_video[video_id] = [annotation]

    return epic_annotations_by_video


def extract_frame_number_from_visor_annotation(visor_frame_annotation):
    """
    Extract frame number from VISOR frame annotation.

    Args:
        visor_frame_annotation: Single frame annotation from VISOR

    Returns:
        Frame number as integer
    """
    image_name = visor_frame_annotation["image"]["name"]
    # Parse "frame_0000001234.jpg" -> 1234
    frame_number_str = image_name.split("frame_")[1].split(".")[0]
    frame_number = int(frame_number_str)
    return frame_number


def find_target_object_polygons(visor_frame_annotation, target_object_name):
    """
    Find polygons for the target object in VISOR frame annotation.

    Args:
        visor_frame_annotation: Single frame annotation from VISOR
        target_object_name: Name of the object to extract (e.g., "knife", "left hand")

    Returns:
        List of polygon segments if found, None otherwise
    """
    annotations_in_frame = visor_frame_annotation["annotations"]

    for annotation_item in annotations_in_frame:
        object_name = annotation_item["name"]

        if object_name == target_object_name:
            # Return segments wrapped in a list for consistency
            polygon_segments = [annotation_item["segments"]]
            return polygon_segments

    return None


def convert_polygons_to_numpy_arrays(polygon_segments):
    """
    Convert polygon segments to list of numpy arrays for cv2.fillPoly.

    Args:
        polygon_segments: List of polygon segments from VISOR

    Returns:
        List of numpy arrays representing polygons
    """
    numpy_polygons = []

    for segment_group in polygon_segments:
        for polygon_points in segment_group:
            # Handle empty polygons
            if polygon_points == []:
                polygon_points = [[0.0, 0.0]]

            numpy_polygon = np.array(polygon_points, dtype=np.int32)
            numpy_polygons.append(numpy_polygon)

    return numpy_polygons


def create_mask_from_polygons(polygon_list, image_width, image_height):
    """
    Create binary mask from polygons.

    Args:
        polygon_list: List of numpy arrays representing polygons
        image_width: Width of output mask
        image_height: Height of output mask

    Returns:
        Binary mask as numpy array (0 for background, 255 for object)
    """
    mask = np.zeros([image_height, image_width], dtype=np.uint8)
    cv2.fillPoly(mask, polygon_list, 255)
    return mask


def calculate_output_frame_index(absolute_frame_number, instruction_start_frame):
    """
    Calculate the frame index within an instruction (for output filename).

    Args:
        absolute_frame_number: Absolute frame number in the video
        instruction_start_frame: Start frame of the instruction

    Returns:
        Frame index within the instruction (0-based)
    """
    frame_index = absolute_frame_number - instruction_start_frame
    return frame_index


def save_mask(mask, output_path):
    """
    Save mask to disk.

    Args:
        mask: Binary mask as numpy array
        output_path: Path to save the mask
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)


def process_visor_annotations(
    visor_annotations_dir,
    epic_annotations_list,
    epic_annotations_by_video,
    output_masks_dir,
    image_width,
    image_height,
):
    """
    Process VISOR annotations and generate masks for EPIC-KITCHENS instructions.

    Args:
        visor_annotations_dir: Directory containing VISOR annotation JSON files
        epic_annotations_list: List of EPIC-KITCHENS annotations (will be modified)
        epic_annotations_by_video: Dictionary mapping video_id to annotations
        output_masks_dir: Directory to save output masks
        image_width: Width of output masks
        image_height: Height of output masks
    """
    # Get list of VISOR annotation files
    visor_annotation_files = os.listdir(visor_annotations_dir)
    visor_annotation_files = [f for f in visor_annotation_files if f.endswith(".json")]

    # Process each VISOR video annotation file
    for visor_filename in tqdm(
        visor_annotation_files, desc="Processing VISOR annotations"
    ):
        # Load VISOR annotation file
        video_id = visor_filename.replace(".json", "")
        visor_file_path = os.path.join(visor_annotations_dir, visor_filename)
        visor_data = json.load(open(visor_file_path, "r"))

        # Process each frame in the video
        visor_frame_annotations = visor_data["video_annotations"]
        for visor_frame_annotation in visor_frame_annotations:
            # Extract frame number
            frame_number = extract_frame_number_from_visor_annotation(
                visor_frame_annotation
            )

            # Find corresponding EPIC-KITCHENS instruction
            epic_annotation = find_instruction_for_frame(
                epic_annotations_by_video, video_id, frame_number
            )

            if epic_annotation is None:
                continue

            # Extract instruction information
            instruction_id = epic_annotation["id"]
            instruction_start_frame = epic_annotation["start_frame"]
            # target_object_name = epic_annotation[
            #     "noun"
            # ]  # <-- Change this to extract different objects
            target_object_name = "right hand"
            original_list_index = epic_annotation["original_index"]

            # Find polygons for target object
            polygon_segments = find_target_object_polygons(
                visor_frame_annotation, target_object_name
            )

            if polygon_segments is None:
                continue

            # Convert polygons to numpy arrays
            numpy_polygons = convert_polygons_to_numpy_arrays(polygon_segments)

            # Create mask from polygons
            mask = create_mask_from_polygons(numpy_polygons, image_width, image_height)

            # Calculate output frame index and path
            frame_index = calculate_output_frame_index(
                frame_number, instruction_start_frame
            )
            output_mask_path = os.path.join(
                output_masks_dir, f"{instruction_id}", f"{frame_index}.png"
            )

            # Save mask
            save_mask(mask, output_mask_path)

            # Add mask path to annotation
            if "sparse_mask_path" not in epic_annotations_list[original_list_index]:
                epic_annotations_list[original_list_index]["sparse_mask_path"] = [
                    output_mask_path
                ]
            else:
                epic_annotations_list[original_list_index]["sparse_mask_path"].append(
                    output_mask_path
                )


if __name__ == "__main__":
    # ===== Configuration (modify these paths as needed) =====
    data_root_dir = "data"
    epic_annotations_json_path = os.path.join(
        data_root_dir, "epic_kitchens_100_train.json"
    )
    output_masks_dir = os.path.join(data_root_dir, "original_sparse_masks")
    visor_annotations_dir = "visor_data/GroundTruth-SparseAnnotations/annotations/train"

    # Mask resolution (EPIC-KITCHENS resolution)
    mask_width = 1920
    mask_height = 1080

    # Output JSON path (same as input)
    output_json_path = os.path.join(data_root_dir, "epic_kitchens_100_train.json")
    # ========================================================

    # Load EPIC-KITCHENS annotations
    epic_annotations_list = json.load(open(epic_annotations_json_path, "r"))

    # Build video-to-annotations mapping
    epic_annotations_by_video = build_epic_annotations_by_video(epic_annotations_list)

    # Process VISOR annotations and generate masks
    process_visor_annotations(
        visor_annotations_dir,
        epic_annotations_list,
        epic_annotations_by_video,
        output_masks_dir,
        mask_width,
        mask_height,
    )

    # Save updated annotations with mask paths
    with open(output_json_path, "w") as json_file:
        json.dump(epic_annotations_list, json_file, indent=1)

    print(f"Processed VISOR annotations")
    print(f"Saved updated annotations to: {output_json_path}")
