import copy

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import all_extend, all_gather


class ActionGenomeEvaluator(object):
    def __init__(self, device):
        self.device = device
        self.num_classes = 36  # include person.
        self.eval = {}
        self._reset()

    def update(self, predictions, targets):
        """
        Arguments:
            predictions (List[Dict]): ex. [{"scores": [...], "labels": [...], "boxes": [[...], [...], ...]}]
            targets (List[Dict]): same as dataset output.

        """
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for i, t in enumerate(targets):
            t["boxes"] = (box_cxcywh_to_xyxy(t["boxes"]) * scale_fct[i]).to(self.device)

        self.targets.append(targets)
        self.predictions.append(predictions)

    def synchronize_between_processes(self):
        self.targets = all_extend(self.targets)
        self.predictions = all_extend(self.predictions)

    def summarize(self):
        dist.barrier()
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        true_boxes = []
        true_labels = []

        for preds in self.predictions:
            pred_boxes.extend([p["boxes"].to(self.device) for p in preds])
            pred_labels.extend([p["labels"].to(self.device) for p in preds])
            pred_scores.extend([p["scores"].to(self.device) for p in preds])

        for tgts in self.targets:
            true_boxes.extend([t["boxes"].to(self.device) for t in tgts])
            true_labels.extend([t["labels"].to(self.device) for t in tgts])

        assert len(true_boxes) == len(true_labels) == len(pred_boxes) == len(pred_labels) == len(pred_scores)

        mAP = self._evaluate(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels)

        self.eval["mAP"] = mAP
        print(f"mAP: {mAP}")
        self._reset()

    def _reset(self):
        self.targets = []
        self.predictions = []

    def _evaluate(self, pred_boxes, pred_labels, pred_scores, true_boxes, true_labels):
        """
        Arguments:
            pred_boxes (List[Tensor[B, 4]]):
            pred_labels (List[Tensor[N, B, C]]):
            pred_scores (List[Tensor[N, B]]):
            true_boxes (List[Tensor[N, B, 4]]):
            true_labels (List[Tensor[N, B, C]]):

        """

        n_classes = self.num_classes

        # Store the image indices for tracking.
        # For example, there are two images that contain 2 boxes and 3 boxes.
        # Then, true_images will be [[0, 0], [1, 1, 1]], which indicates the same indices
        # to the objects in same images.
        true_images = list()
        for i in range(len(true_boxes)):
            true_images.extend([i] * true_boxes[i].size(0))

        true_images = torch.LongTensor(true_images).to(self.device)
        true_boxes = torch.cat(true_boxes, dim=0)
        true_labels = torch.cat(true_labels, dim=0)

        assert len(true_images) == len(true_boxes) == len(true_labels)

        # Store the image indices for tracking.
        # Details are the same as true_images.
        pred_images = list()
        for i in range(len(pred_boxes)):
            pred_images.extend([i] * pred_boxes[i].size(0))

        pred_images = torch.LongTensor(pred_images).to(self.device)
        pred_boxes = torch.cat(pred_boxes, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)

        assert len(pred_images) == len(pred_boxes) == len(pred_labels) == len(pred_scores)

        # Calculate average precisions for each class.
        average_precisions = torch.zeros((n_classes), dtype=torch.float)

        # Loop for the classes.
        for c in range(n_classes):
            print(f"Start class {c}")

            # Extract only objects with this class.
            class_true_images = true_images[true_labels == c]
            class_true_boxes = true_boxes[true_labels == c]
            class_pred_images = pred_images[pred_labels == c]
            class_pred_boxes = pred_boxes[pred_labels == c]
            class_pred_scores = pred_scores[pred_labels == c]

            # Each box is whether detected or not.
            class_true_boxes_detected = torch.zeros((class_true_boxes.size(0)), dtype=torch.uint8).to(self.device)

            # Number of detected objects in this class.
            class_n_pred_detections = class_pred_boxes.shape[0]
            if class_n_pred_detections == 0:
                continue

            # Sort the detected objects in descending-order of detection scores.
            class_pred_scores, sort_ind = torch.sort(class_pred_scores, dim=0, descending=True)
            class_pred_boxes = class_pred_boxes[sort_ind]

            # True positives and false positives.
            tp = torch.zeros((class_n_pred_detections), dtype=torch.float).to(self.device)
            fp = torch.zeros((class_n_pred_detections), dtype=torch.float).to(self.device)

            # Loop for detected objects.
            for d in tqdm(range(class_n_pred_detections)):
                this_pred_image = class_pred_images[d]
                this_pred_box = class_pred_boxes[d].unsqueeze(0)

                # Find objects in the same image with this class.
                # If it was not found, false positive.
                target_object_boxes = class_true_boxes[class_true_images == this_pred_image]
                if target_object_boxes.size(0) == 0:
                    fp[d] = 1
                    continue

                # Calculate each overlap with target object boxes.
                overlaps = self._find_overlap(this_pred_box, target_object_boxes)

                # Find maximum overlaps.
                max_overlaps, ind = torch.max(overlaps.squeeze(0), dim=0)

                # Reconstruct the boxes indices.
                box_ind = torch.tensor(range(class_true_boxes.size(0)))[class_true_images == this_pred_image][
                    ind
                ]

                if max_overlaps.item() > 0.5:
                    if class_true_boxes_detected[box_ind] == 0:
                        tp[d] = 1
                        class_true_boxes_detected[box_ind] = 1
                    else:
                        fp[d] = 1
                else:
                    fp[d] = 1

            csum_tp = torch.cumsum(tp, dim=0)
            csum_fp = torch.cumsum(fp, dim=0)
            csum_precision = csum_tp / (csum_tp + csum_fp + 1e-10)
            csum_recall = csum_tp / (class_true_boxes.size(0) + 1e-10)

            recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(self.device)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = csum_recall >= t
                if recalls_above_t.any():
                    precisions[i] = csum_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.0
            average_precisions[c] = precisions.mean()
            print(f"Class {c} AP: {average_precisions[c]}")

        mean_average_precision = average_precisions.mean().item()

        return mean_average_precision

    def _find_overlap(self, boxes1, boxes2):
        """
        Arguments:
            box1 (Tensor[B1, 4]): Format is (xyxy).
            box2 (Tensor[B2, 4]): Format is (xyxy).

        Returns:
            Tensor[B1, B2]:

        """
        # Calculate intersections. Tensor[B1, B2].
        intersection = self._find_intersection(boxes1, boxes2)

        # Calculate arease of each box. Tensor[B1], Tensor[B2].
        area_boxex1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area_boxex2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Calculate union. Tensor[B1, B2].
        # Use torch auto-broadcasts singleton dimensions.
        union = area_boxex1.unsqueeze(1) + area_boxex2.unsqueeze(0) - intersection

        # Tensor[B1, B2].
        return intersection / union

    def _find_intersection(self, boxes1, boxes2):
        """
        Arguments:
            box1 (Tensor[B1, 4]): Format is (xyxy).
            box2 (Tensor[B2, 4]): Format is (xyxy).

        Returns:
            Tensor[B1, B2]:

        """
        # Use torch auto-broadcasts singleton dimensions.
        # Tensor[B1, B2, 2].
        lower_bounds = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0))

        # Tensor[B1, B2, 2].
        upper_bounds = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0))

        # Tensor[B1, B2, 2].
        intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)

        # Tensor[B1, B2].
        return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
