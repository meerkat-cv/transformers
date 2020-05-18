#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.:examples/ner/

echo $PYTHONPATH

MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
SEED=1

DATA_DIRS="/home/jpsmartinez/Datasets/NER/meerkat-NER_inspected"
LABEL_PATH=/home/jpsmartinez/Datasets/NER/meerkat-NER_inspected/labels.txt
OUTPUT_DIR=portuguese-bert-meerkat_inspected+lener_end

LOG_PATH=$OUTPUT_DIR/test.log

CMD="python3 -u test_ner.py \
    --data_dir $DATA_DIRS \
    --model_type bert \
    --labels $LABEL_PATH \
    --model_name_or_path $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_LENGTH --seed $SEED \
    --overwrite_cache"

tmux new -s ner-test -d "$CMD &> $LOG_PATH" && sleep 3 && tail -f $LOG_PATH
