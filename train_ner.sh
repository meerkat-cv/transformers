#!/bin/bash

echo $PYTHONPATH

# Configure training parameters
MAX_LENGTH=128
BERT_MODEL=neuralmind/bert-base-portuguese-cased
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
SEED=1

DATA_DIRS="/home/jpsmartinez/Datasets/meerkat-NER" # /home/jpsmartinez/Datasets/leNER-Br"
LABEL_PATH=/home/jpsmartinez/Datasets/meerkat-NER/labels.txt
OUTPUT_DIR=portuguese-bert-meerkat_only

# Configure script behavior
full () {
    ACTIONS="--do_train --do_eval --do_predict"
    OVERWRITES="--overwrite_output_dir --overwrite_cache"
}
just_test () {
    ACTIONS="--do_predict"
    OVERWRITES=""
}
just_train () {
    ACTIONS="--do_train"
    OVERWRITES="--overwrite_output_dir --overwrite_cache"
}

# Launch training
run() {
    CMD="python3 -u examples/ner/run_ner.py --data_dir $DATA_DIRS \
        --model_type bert --labels $LABEL_PATH \
        --model_name_or_path $BERT_MODEL --output_dir $OUTPUT_DIR --max_seq_length  $MAX_LENGTH --num_train_epochs $NUM_EPOCHS --per_gpu_train_batch_size $BATCH_SIZE --save_steps $SAVE_STEPS --seed $SEED \
        $ACTIONS $OVERWRITES" 

    tmux new -s ner-train -d "$CMD &> training.log"
}

#just_train
full

run