import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from examples.ner.run_ner import set_seed, get_labels, evaluate
from seqeval.metrics import f1_score, precision_score, recall_score

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForTokenClassification,
    AlbertTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForTokenClassification, AlbertTokenizer),
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        nargs="+",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    setattr(args, "proportion", [1, 1])
    setattr(args, "no_cuda", False)
    setattr(args, "local_rank", -1)
    setattr(args, "fp16", False)
    setattr(args, "per_gpu_eval_batch_size", 8)

    return args

def main():
    args = get_args()

    # device allocation
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    result, predictions, gt, examples_list = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test_examples")
    examples_list = np.array(examples_list)

    # flatten examples_list
    examples_ids = []
    for l in examples_list:
        assert all(map(lambda x: x == l[0], l[1:]))
        examples_ids.append(l[0])
    examples_ids = np.array(examples_ids)

    results_per_example = {}
    for e_idx in examples_ids:
        indexes = np.where(examples_ids == e_idx)[0]
        results_per_example[e_idx] = {
            "precision": precision_score(np.take(gt, indexes), np.take(predictions, indexes)),
            "recall": recall_score(np.take(gt, indexes), np.take(predictions, indexes)),
            "f1": f1_score(np.take(gt, indexes), np.take(predictions, indexes)),
        }
    print(results_per_example)

    examples_filter_criteria = lambda r: r["f1"] < 0.6
    bad_examples = list(filter(lambda k: examples_filter_criteria(results_per_example[k]), results_per_example.keys()))
    print(bad_examples)
    print("%f%% of examples are bad, given provided criteria" % (100 *  len(bad_examples) / len(results_per_example.keys())))

    # Save results
    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")

    # predictions for test set
    result, predictions, gt, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
    with open(output_test_predictions_file, "w") as writer:
        test_data_dir = args.data_dir[0]                    # first data dir is used for testing
        with open(os.path.join(test_data_dir, "test.txt"), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split()[0] + " " + predictions[example_id].pop(0) + " " + gt[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

main() if __name__ == '__main__' else True