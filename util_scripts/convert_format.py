import argparse
import os
import pickle
import shutil

from tqdm import tqdm
from util.box_ops import list_box_xyxy_to_xywh

parser = argparse.ArgumentParser()
parser.add_argument("--ag_path", required=True, type=str)
parser.add_argument("--dest_path", required=True, type=str)


def main(args):
    print("Start process")
    annotations = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}

    image_names = []
    with open(os.path.join(args.ag_path, "annotations/frame_list.txt"), "r") as file:
        lines = file.readlines()
        image_names = [line.rstrip() for line in lines]

    object_classes = {}
    with open(os.path.join(args.ag_path, "annotations/object_classes.txt"), "r") as file:
        lines = file.readlines()
        class_id = 0
        for line in lines:
            class_name = line.rstrip()
            object_classes[class_name] = class_id
            annotations["categories"].append(
                {
                    "supercategory": class_name,
                    "id": class_id,
                    "name": class_name,
                }
            )
            class_id += 1

    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join(args.ag_path, "annotations/object_bbox_and_relationship.pkl"),
            "rb",
        )
    )
    person_bbox = pickle.load(open(os.path.join(args.ag_path, "annotations/person_bbox.pkl", "rb")))

    image_names_with_subset = []
    for image_name in image_names:
        rels = object_bbox_and_relationship[image_name]
        image_names_with_subset.append(
            {
                "image_name": image_name,
                "subset": "train" if rels[0]["metadata"]["set"] == "train" else "val",
            }
        )

    if not os.path.exists(os.path.join(args.dest_path, "train")):
        os.mkdir(os.path.join(args.dest_path, "train"))

    if not os.path.exists(os.path.join(args.dest_path, "val")):
        os.mkdir(os.path.join(args.dest_path, "val"))

    if not os.path.exists(os.path.join(args.dest_path, "annotations")):
        os.mkdir(os.path.join(args.dest_path, "annotations"))

    image_id_map = {}

    print("Start copy frames")
    for i, image in tqdm(enumerate(image_names_with_subset)):
        frame_path = os.path.join(args.ag_path, "frames", image["image_name"])
        dest_path = os.path.join(args.dest_path, image["subset"], f"{i}.png")
        image_id_map[image["image_name"]] = i
        shutil.copy(frame_path, dest_path)

    print("Start convert annotations")
    count_id = 0
    for image_name in tqdm(image_names):
        rels = object_bbox_and_relationship[image_name]

        for rel in rels:
            bbox = rel["bbox"]
            class_id = object_classes[rel["class"].replace("/", "")]

            if bbox is None:
                continue

            annotations["annotations"].append(
                {
                    "segmentation": [],
                    "area": 0,
                    "iscrowd": 0,
                    "image_id": image_id_map[image_name],
                    "bbox": bbox,
                    "category_id": class_id,
                    "id": count_id,
                }
            )
            count_id += 1

        person = person_bbox[image_name]
        for p_bbox in person["bbox"]:
            bbox = list_box_xyxy_to_xywh(p_bbox)
            class_id = object_classes["person"]
            annotations["annotations"].append(
                {
                    "segmentation": [],
                    "area": 0,
                    "iscrowd": 0,
                    "image_id": image_id_map[image_name],
                    "bbox": bbox,
                    "category_id": class_id,
                    "id": count_id,
                }
            )
            count_id += 1

        annotations["images"].append(
            {
                "license": 1,
                "file_name": f"{image_id_map[image_name].png}",
                "coco_url": "",
                "height": person[0]["bbox_size"][1],
                "width": person[0]["bbox_size"][0],
                "date_captured": "",
                "flickr_url": "",
                "id": image_id_map[image_name],
            }
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
