.PHONY: train_ag
train_ag:
	python main.py \
		--distributed \
		--num_workers 0 \
		--batch_size 4 \
		--dataset_file $(DATASET_FILE) \
		--ag_path $(AG_PATH) \
		--output_dir $(OUTPUT_DIR)
