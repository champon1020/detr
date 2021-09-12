.PHONY: train_ag
train_ag:
	python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
		--num_workers=0 \
		--coco_path $(COCO_PATH) \
		--dataset_file $(DATASET_FILE) \
		--ag_path $(AG_PATH) \
		--output_dir $(OUTPUT_DIR)
