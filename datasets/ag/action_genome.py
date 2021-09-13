import json
import os
import pickle
from pathlib import Path

import datasets.transforms as T
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from util.box_ops import list_box_xyxy_to_xywh


class ActionGenome(Dataset):
    def __init__(self, image_set, img_folder, ann_folder, transforms=None):
        assert image_set in ["train", "val", "dummy"]

        self.image_set = image_set if image_set == "train" else "test"
        self.img_folder = img_folder
        self.transforms = transforms
        self.ids = []
        self.dataset = {}

        if image_set != "dummy":
            self._make_dataset(ann_folder)

    def _make_dataset(self, ann_folder):
        print("Making action genome dataset...")

        image_ids = []
        with open(os.path.join(ann_folder, "frame_list.txt"), "r") as file:
            lines = file.readlines()
            image_ids = [line.rstrip() for line in lines]

        """
        try:
            ann_file = pickle.load(
                open(os.path.join(ann_folder, f"action_genome_detection_{self.image_set}.pkl"), "rb")
            )
            self.ids = image_ids
            self.dataset = ann_file
            return
        except:
            pass
        """

        object_classes = {}
        with open(os.path.join(ann_folder, "object_classes.txt"), "r") as file:
            lines = file.readlines()
            class_id = 0
            for line in lines:
                object_classes[line.rstrip()] = class_id
                class_id += 1

        object_bbox_and_rel = pickle.load(
            open(os.path.join(ann_folder, "object_bbox_and_relationship.pkl"), "rb")
        )

        person_bbox = pickle.load(open(os.path.join(ann_folder, "person_bbox.pkl"), "rb"))

        for i, id in tqdm(enumerate(image_ids)):
            rels = object_bbox_and_rel[id]
            boxes, classes = [], []
            image_set = ""
            for rel in rels:
                if rel["metadata"]["set"] != self.image_set:
                    continue

                image_set = rel["metadata"]["set"]

                if rel["bbox"] is None:
                    continue

                boxes.append(rel["bbox"])
                classes.append(object_classes[rel["class"].replace("/", "")])

            # If image_set corresponds to the self.image_set, extract the person boxes.
            if image_set == self.image_set:
                person = person_bbox[id]
                bbox = person["bbox"]
                boxes.extend([list_box_xyxy_to_xywh(b) for b in bbox])
                classes.extend([object_classes["person"] for _ in range(len(bbox))])

            if len(boxes) == 0:
                continue

            image = Image.open(os.path.join(self.img_folder, id))
            w, h = image.size

            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            classes = torch.tensor(classes, dtype=torch.int64)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            self.ids.append(id)
            self.dataset[id] = {
                "boxes": boxes,
                "labels": classes,
                "image_id": torch.tensor(i),
                "size": torch.as_tensor([int(h), int(w)]),
                "orig_size": torch.as_tensor([int(h), int(w)]),
            }

        """
        pickle.dump(
            self.dataset,
            open(os.path.join(ann_folder, f"action_genome_detection_{self.image_set}.pkl"), "wb"),
        )
        """

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _load_image(self, id):
        return Image.open(os.path.join(self.img_folder, id)).convert("RGB")

    def _load_target(self, id):
        """
        Dataset item format:
        {
            "boxes": [[0.5426, 0.4170, 0.3599, 0.7220],
                      [0.2823, 0.4324, 0.4917, 0.2580],
                      [0.4459, 0.6099, 0.6642, 0.2646]],
            "labels": [18, 1, 1],
            "image_id": "xxx.mp4/yyy.png",
        }

        """
        return self.dataset[id]


def ag_transforms(image_set):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    if args.eval and image_set == "train":
        return ActionGenome("dummy", "", "")

    root = Path(args.ag_path)
    assert root.exists(), f"provided COCO path {root} does not exist"

    img_folder = root / "frames"
    ann_folder = root / "annotations"
    dataset = ActionGenome(
        image_set,
        img_folder,
        ann_folder,
        ag_transforms(image_set),
    )
    return dataset
