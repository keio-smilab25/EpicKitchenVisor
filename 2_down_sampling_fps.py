import os
import shutil
import json
from tqdm import tqdm


def down_sample_sequence(
    image_seq_dir: str,
    mask_seq_dir: str,
    output_image_seq_dir: str,
    output_mask_seq_dir: str,
    original_fps: int,
    target_fps: int,
):
    os.makedirs(output_image_seq_dir, exist_ok=True)
    os.makedirs(output_mask_seq_dir, exist_ok=True)

    frame_indices = []
    step = original_fps / target_fps
    idx = 0.0
    while int(idx) < len(os.listdir(image_seq_dir)):
        frame_indices.append(int(idx))
        idx += step

    for idx, i in enumerate(frame_indices):
        img_filename = f"{i}.jpg"
        dst_img_filename = f"{idx}.png"
        src_img_path = os.path.join(image_seq_dir, img_filename)
        src_mask_path = os.path.join(mask_seq_dir, img_filename)

        dst_img_path = os.path.join(output_image_seq_dir, dst_img_filename)
        dst_mask_path = os.path.join(output_mask_seq_dir, dst_img_filename)

        # convert jpg to png
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_mask_path, dst_mask_path)


if __name__ == "__main__":
    target_dir = "data"
    annotation_file_name = "epic_kitchens_100_train.json"
    original_fps = 60
    target_fps = 15

    images_dir = os.path.join(target_dir, "images")
    masks_dir = os.path.join(target_dir, "masks")
    annotation_file_path = os.path.join(target_dir, annotation_file_name)

    original_images_dir = os.path.join(target_dir, "original_images")
    original_masks_dir = os.path.join(target_dir, "original_masks")

    os.rename(images_dir, original_images_dir)
    os.rename(masks_dir, original_masks_dir)

    annotation = json.load(open(annotation_file_path, "r"))
    for item in tqdm(annotation):
        id = item["id"]
        original_image_seq_dir = os.path.join(original_images_dir, str(id))
        original_mask_seq_dir = os.path.join(original_masks_dir, str(id))

        if not os.path.exists(original_image_seq_dir) or not os.path.exists(
            original_mask_seq_dir
        ):
            continue

        output_image_seq_dir = os.path.join(images_dir, str(id))
        output_mask_seq_dir = os.path.join(masks_dir, str(id))
        down_sample_sequence(
            original_image_seq_dir,
            original_mask_seq_dir,
            output_image_seq_dir,
            output_mask_seq_dir,
            original_fps,
            target_fps,
        )

    print("Down-sampling completed.")
