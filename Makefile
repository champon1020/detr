.PHONY: train_ag
train_ag:
	python main.py \
		--distributed \
		--num_workers 0 \
		--coco_path $(COCO_PATH) \
		--dataset_file $(DATASET_FILE) \
		--ag_path $(AG_PATH) \
		--output_dir $(OUTPUT_DIR)
