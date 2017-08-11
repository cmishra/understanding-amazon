RAW_DATA:= /u/cmishra1/understanding_amazon/data_raw
PROCESSED_DATA:= /u/cmishra1/understanding_amazon/data_processed
PYTHON_EXE:= /opt/share/Python-2.7.12/x86_64/bin/python

EXTRACTION_TARGETS:= $(PROCESSED_DATA)/test-jpg $(PROCESSED_DATA)/targets.json $(PROCESSED_DATA)/train-jpg

.DEFAULT_GOAL:= all

$(PROCESSED_DATA)/targets.json:
	unzip -u $(RAW_DATA)/train_v2.csv.zip -d $(PROCESSED_DATA)
	rm -rf $(PROCESSED_DATA)/__MACOSX
	python csv_to_json.py --train_csv $(PROCESSED_DATA)/train_v2.csv --output $(PROCESSED_DATA)/targets.json
	rm $(PROCESSED_DATA)/train_v2.csv

$(PROCESSED_DATA)/test-jpg:
	jbsub -interactive -queue x86_1h -mem 50G tar -xf $(RAW_DATA)/test-jpg.tar -C $(PROCESSED_DATA)

$(PROCESSED_DATA)/train-jpg:
	jbsub -interactive -queue x86_1h -mem 50G tar -xf $(RAW_DATA)/train-jpg.tar -C $(PROCESSED_DATA)

### data split params
SPLIT_PARAMS:= --train_proportion 0.6
SPLIT_PARAMS+= --val_proportion 0.2
SPLIT_PARAMS+= --source_dir $(PROCESSED_DATA)/train-jpg
SPLIT_PARAMS+= --destination_dir $(PROCESSED_DATA)
$(PROCESSED_DATA)/test: $(EXTRACTION_TARGETS)
	$(PYTHON_EXE) create_splits.py $(SPLIT_PARAMS)
	rm -r $(PROCESSED_DATA)/train-jpg
	mv $(PROCESSED_DATA)/test-jpg $(PROCESSED_DATA)/test-competitorboard

all: $(PROCESSED_DATA)/test

.PHONY: clean all

clean:
	jbsub -interactive -queue x86_1h rm -rf $(PROCESSED_DATA)/*

