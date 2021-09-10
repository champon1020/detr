import os
import pickle
from pathlib import Path

import datasets.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class ActionGenome(Dataset):
    def __init__(self, image_set, img_folder, ann_folder, transforms):
        assert image_set in ["train", "test"]

        self.image_set = image_set
        self.img_folder = img_folder
        self._transforms = transforms
        self.ids = []
        self.dataset = {}
        self.object_classes = {}
        self._make_dataset(ann_folder)

    def _make_dataset(self, ann_folder):
        print("Making action genome dataset...")
        with open(os.path.join(ann_folder, "frame_list.txt"), "r") as file:
            lines = file.readlines()
            self.ids = [line.rstrip() for line in lines]

        with open(os.path.join(ann_folder, "object_classes.txt"), "r") as file:
            lines = file.readlines()
            class_id = 0
            for line in lines:
                self.object_classes[line.rstrip()] = class_id
                class_id += 1

        object_bbox_and_rel = pickle.load(
            open(os.path.join(ann_folder, "object_bbox_and_relationship.pkl"), "rb")
        )

        for id in tqdm(self.ids):
            rels = object_bbox_and_rel[id]
            boxes, classes = [], []
            for rel in rels:
                if rel["metadata"]["set"] != self.image_set:
                    continue

                boxes.append(rel["bbox"])
                classes.append(self.object_classes[rel["class"]])

            image = Image.open(os.path.join(self.img_folder, id))
            w, h = image.size

            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            classes = torch.tensor(classes, dtype=torch.int64)

            self.dataset[id] = {"boxes": boxes, "labels": classes, "image_id": id}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = self._laod_image(id)
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

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

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

    """
    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )
    """

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
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
