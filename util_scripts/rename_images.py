import argparse
import os
import pickle
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--ag_annotations_path", required=True, type=str)
parser.add_argument("--ag_frames_path", required=True, type=str)
parser.add_argument("--dest_path", required=True, type=str)


def main(args):
    image_names = []
    with open(os.path.join(args.ag_frame_list_path, "frame_list.txt"), "r") as file:
        lines = file.readlines()
        image_names = [line.rstrip() for line in lines]

    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join(args.ag_annotations_path, "object_bbox_and_relationship.pkl"),
            "rb",
        )
    )

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

    for i, image in enumerate(image_names_with_subset):
        frame_path = os.path.join(args.ag_frames_path, image["image_name"])
        dest_path = os.path.join(args.dest_path, image["subset"], f"{i}.png")
        shutil.copy(frame_path, dest_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
