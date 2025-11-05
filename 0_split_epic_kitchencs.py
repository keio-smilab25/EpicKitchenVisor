import pandas as pd
import json
import os
from tqdm import tqdm
from PIL import Image
import shutil

if __name__ == "__main__":
    annotation_path = "epic-kitchens-100-annotations/EPIC_100_train.csv"
    EPIC_KITCHENS_dir = "EPIC-KITCHENS"
    output_dir = "data"
    save_path = os.path.join(output_dir, "epic_kitchens_100_train.json")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(annotation_path)

    # for read df with tqdm
    all_annotations = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        participant_id = row["participant_id"]
        video_id = row["video_id"]
        start_frame = row["start_frame"]
        stop_frame = row["stop_frame"]
        instruction = row["narration"]
        verb = row["verb"]
        noun = row["noun"]
        all_nouns = (
            row["all_nouns"]
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .split(", ")
        )
        json_item = {
            "id": idx,
            "participant_id": participant_id,
            "video_id": video_id,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "instruction": instruction,
            "verb": verb,
            "noun": noun,
            "all_nouns": all_nouns,
        }

        image_dir_base = os.path.join(
            EPIC_KITCHENS_dir, participant_id, "rgb_frames", video_id
        )
        if not os.path.exists(image_dir_base):
            continue
        image_filename_list = [
            f"frame_{frame:010d}.jpg" for frame in range(start_frame, stop_frame + 1)
        ]
        image_save_dir = os.path.join(output_dir, "images", f"{idx}")
        os.makedirs(image_save_dir, exist_ok=True)
        for idy, image_filename in enumerate(image_filename_list):
            src_image_path = os.path.join(image_dir_base, image_filename)
            dst_image_path = os.path.join(image_save_dir, f"{idy}.jpg")
            shutil.copy(src_image_path, dst_image_path)
            # image = Image.open(src_image_path).convert("RGB")
            # image.save(dst_image_path)
        json_item["image_dir"] = image_save_dir

        all_annotations.append(json_item)

    json.dump(all_annotations, open(save_path, "w"), indent=1)
