import json
import numpy as np
import os
from tqdm import tqdm
import cv2


def get_annotation_by_frame(epic_annotations_dict, visor_video_id, frame_num):
    epic_annotations = epic_annotations_dict.get(visor_video_id, [])
    for annotation in epic_annotations:
        if annotation["start_frame"] <= frame_num <= annotation["stop_frame"]:
            return annotation
    return None


if __name__ == "__main__":
    data_dir = "data"
    epic_annotations_path = os.path.join(data_dir, "epic_kitchens_100_train.json")
    sparse_masks_dir = os.path.join(data_dir, "sparse_masks")
    visor_annotations_dir = "visor_data/GroundTruth-SparseAnnotations/annotations/train"

    epic_annotations = json.load(open(epic_annotations_path, "r"))
    epic_annotations_dict = {}
    # for item in tqdm(epic_annotations, desc="Building EPIC annotations dict"):
    for idx, item in enumerate(
        tqdm(epic_annotations, desc="Building EPIC annotations dict")
    ):
        item["original_index"] = idx
        epic_annotations[idx]["sparse_mask_path"] = []
        video_id = item["video_id"]
        if video_id in epic_annotations_dict:
            epic_annotations_dict[video_id].append(item)
        else:
            epic_annotations_dict[video_id] = [item]

    visor_annotation_list = os.listdir(visor_annotations_dir)
    visor_annotation_list = [f for f in visor_annotation_list if f.endswith(".json")]

    for visor_annotation_file in tqdm(
        visor_annotation_list, desc="Processing Visor annotations"
    ):
        visor_video_id = visor_annotation_file.replace(".json", "")
        visor_annotation_data = json.load(
            open(os.path.join(visor_annotations_dir, visor_annotation_file), "r")
        )
        video_annotations = visor_annotation_data["video_annotations"]
        for video_annotation in video_annotations:
            frame_num = int(
                video_annotation["image"]["name"].split("frame_")[1].split(".")[0]
            )
            epic_annotation = get_annotation_by_frame(
                epic_annotations_dict, visor_video_id, frame_num
            )
            if epic_annotation is None:
                continue

            id = epic_annotation["id"]
            start_frame = epic_annotation["start_frame"]
            target_object = epic_annotation["noun"]
            original_index = epic_annotation["original_index"]
            polygons = None
            for annotation_item in video_annotation["annotations"]:
                name = annotation_item["name"]
                if name == target_object:
                    polygons = [annotation_item["segments"]]
                    break

            if polygons is None:
                continue

            save_file_name_num = frame_num - start_frame
            save_file_path = os.path.join(
                sparse_masks_dir,
                f"{id}",
                f"{save_file_name_num}.png",
            )

            ps = []
            for polygon in polygons:
                for poly in polygon:
                    if poly == []:
                        poly = [[0.0, 0.0]]
                    ps.append(np.array(poly, dtype=np.int32))

            input_resolution = (1920, 1080)
            img = np.zeros([input_resolution[1], input_resolution[0]], dtype=np.uint8)
            cv2.fillPoly(img, ps, 255)
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            cv2.imwrite(save_file_path, img)
            if "sparse_mask_path" not in epic_annotations[original_index]:
                epic_annotations[original_index]["sparse_mask_path"] = [save_file_path]
            else:
                epic_annotations[original_index]["sparse_mask_path"].append(
                    save_file_path
                )

    save_epic_annotations_path = os.path.join(data_dir, "epic_kitchens_100_train.json")
    json.dump(epic_annotations, open(save_epic_annotations_path, "w"), indent=1)
