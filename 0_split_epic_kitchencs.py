import pandas as pd
import json
import os
from tqdm import tqdm
from PIL import Image
import shutil


def parse_all_nouns_field(all_nouns_str):
    """
    Parse the 'all_nouns' field from CSV (e.g., "['noun1', 'noun2']") into a list.

    Args:
        all_nouns_str: String representation of noun list from CSV

    Returns:
        List of noun strings
    """
    cleaned_str = all_nouns_str.replace("[", "").replace("]", "").replace("'", "")
    noun_list = cleaned_str.split(", ")
    return noun_list


def create_annotation_dict(row, instruction_id):
    """
    Create annotation dictionary from a DataFrame row.

    Args:
        row: pandas DataFrame row containing annotation data
        instruction_id: Unique identifier for this instruction

    Returns:
        Dictionary with annotation fields
    """
    participant_id = row["participant_id"]
    video_id = row["video_id"]
    start_frame = row["start_frame"]
    stop_frame = row["stop_frame"]
    instruction_text = row["narration"]
    verb = row["verb"]
    noun = row["noun"]
    all_nouns = parse_all_nouns_field(row["all_nouns"])

    annotation_dict = {
        "id": instruction_id,
        "participant_id": participant_id,
        "video_id": video_id,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "instruction": instruction_text,
        "verb": verb,
        "noun": noun,
        "all_nouns": all_nouns,
    }

    return annotation_dict


def get_source_frame_dir(epic_kitchens_root_dir, participant_id, video_id):
    """
    Get the directory path containing source RGB frames.

    Args:
        epic_kitchens_root_dir: Root directory of EPIC-KITCHENS dataset
        participant_id: Participant ID (e.g., 'P01')
        video_id: Video ID (e.g., 'P01_01')

    Returns:
        Path to the frame directory
    """
    frame_dir = os.path.join(
        epic_kitchens_root_dir,
        participant_id,
        "rgb_frames",
        video_id
    )
    return frame_dir


def generate_frame_filenames(start_frame, stop_frame):
    """
    Generate list of frame filenames for the given frame range.

    Args:
        start_frame: Starting frame number
        stop_frame: Ending frame number (inclusive)

    Returns:
        List of frame filenames (e.g., ['frame_0000000000.jpg', ...])
    """
    frame_filenames = [
        f"frame_{frame_num:010d}.jpg"
        for frame_num in range(start_frame, stop_frame + 1)
    ]
    return frame_filenames


def copy_instruction_frames(source_frame_dir, frame_filenames, output_frame_dir):
    """
    Copy frames from source directory to output directory with sequential naming.

    Args:
        source_frame_dir: Directory containing source frames
        frame_filenames: List of source frame filenames
        output_frame_dir: Directory to save output frames
    """
    os.makedirs(output_frame_dir, exist_ok=True)

    for frame_index, source_filename in enumerate(frame_filenames):
        source_path = os.path.join(source_frame_dir, source_filename)
        output_filename = f"{frame_index}.jpg"
        output_path = os.path.join(output_frame_dir, output_filename)
        shutil.copy(source_path, output_path)


def process_epic_kitchens_annotations(annotation_csv_path, epic_kitchens_root_dir, output_root_dir):
    """
    Process EPIC-KITCHENS annotations and extract frames for each instruction.

    Args:
        annotation_csv_path: Path to EPIC_100_train.csv
        epic_kitchens_root_dir: Root directory of EPIC-KITCHENS dataset
        output_root_dir: Root directory for output files

    Returns:
        List of annotation dictionaries
    """
    # Load annotations CSV
    annotations_df = pd.read_csv(annotation_csv_path)

    all_annotations = []

    # Process each instruction
    for instruction_id, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Processing instructions"):
        # Create annotation dictionary
        annotation_dict = create_annotation_dict(row, instruction_id)

        # Get source frame directory
        participant_id = annotation_dict["participant_id"]
        video_id = annotation_dict["video_id"]
        source_frame_dir = get_source_frame_dir(epic_kitchens_root_dir, participant_id, video_id)

        # Skip if source frames don't exist
        if not os.path.exists(source_frame_dir):
            continue

        # Generate frame filenames
        start_frame = annotation_dict["start_frame"]
        stop_frame = annotation_dict["stop_frame"]
        frame_filenames = generate_frame_filenames(start_frame, stop_frame)

        # Copy frames to output directory
        output_frame_dir = os.path.join(output_root_dir, "images", f"{instruction_id}")
        copy_instruction_frames(source_frame_dir, frame_filenames, output_frame_dir)

        # Add output directory to annotation
        annotation_dict["image_dir"] = output_frame_dir

        all_annotations.append(annotation_dict)

    return all_annotations


if __name__ == "__main__":
    # ===== Configuration (modify these paths as needed) =====
    annotation_csv_path = "epic-kitchens-100-annotations/EPIC_100_train.csv"
    epic_kitchens_root_dir = "EPIC-KITCHENS"
    output_root_dir = "data"
    output_json_filename = "epic_kitchens_100_train.json"
    # ========================================================

    # Create output directory
    os.makedirs(output_root_dir, exist_ok=True)

    # Process annotations and extract frames
    all_annotations = process_epic_kitchens_annotations(
        annotation_csv_path,
        epic_kitchens_root_dir,
        output_root_dir
    )

    # Save annotations to JSON
    output_json_path = os.path.join(output_root_dir, output_json_filename)
    with open(output_json_path, "w") as json_file:
        json.dump(all_annotations, json_file, indent=1)

    print(f"Processed {len(all_annotations)} instructions")
    print(f"Saved annotations to: {output_json_path}")
