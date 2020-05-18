import os
import argparse
import logging
import sys
sys.path.append('src/')
sys.path.append('examples/ner/')

import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from examples.ner.run_ner import set_seed, get_labels, evaluate
from seqeval.metrics import f1_score, precision_score, recall_score
from examples.ner.utils_ner import store_predictions

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
    setattr(args, "decode_batch", False)

    return args

def main():
    args = get_args()
    print(args)

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

    # predictions per example
    if os.path.isdir(os.path.join(args.data_dir[0], "test_samples")):
        result, predictions, gt, examples_list, examples_paths = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test_samples")
        assert len(predictions) == len(examples_list)
        examples_list = np.array(examples_list)

        # flatten examples_list
        examples_ids = []
        for l in examples_list:
            assert all(map(lambda x: x == l[0], l[1:]))
            examples_ids.append(l[0])
        examples_ids = np.array(examples_ids)
        print("Number of examples: %d" % (len(np.unique(examples_ids))))
        try:
            assert len(list(filter(lambda x: x>=0, np.unique(examples_ids)))) == len(set(examples_paths))
        except AssertionError:
            print(list(filter(lambda x: x>=0, np.unique(examples_ids))), set(examples_paths))
            raise Exception("Number of unique examples ids do not match number of example paths")

        results_per_example = {}
        predictions_per_sample_dir = "sample_predictions"
        os.mkdir(predictions_per_sample_dir)
        for e_idx in np.unique(examples_ids):
            indexes = np.where(examples_ids == e_idx)[0]
            e_gt = np.take(gt, indexes)
            e_pred = np.take(predictions, indexes)
            e_gt = [list(x) for x in e_gt]
            e_pred = [list(x) for x in e_pred]

            results_per_example[e_idx] = {
                "precision": precision_score(e_gt, e_pred),
                "recall": recall_score(e_gt, e_pred),
                "f1": f1_score(e_gt, e_pred),
            }
            e_path = examples_paths[e_idx]
            store_predictions(e_pred, e_gt, os.path.join(predictions_per_sample_dir, os.path.basename(e_path)), e_path)
        print(results_per_example)

        examples_filter_criteria = lambda r: r["f1"] < 0.6
        bad_examples = list(filter(lambda k: examples_filter_criteria(results_per_example[k]), results_per_example.keys()))
        print(bad_examples)
        print("%f%% of examples are bad, given provided criteria" % (100 *  len(bad_examples) / len(results_per_example.keys())))

    # predictions for test set
    do_test_set = True
    if do_test_set:
        result, predictions, gt, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")

        # computes metrics per class
        results_per_class = {}
        for label in labels:
            l_gt = np.copy(gt)
            l_preds = np.copy(predictions)
        
            # could be replaced by np.isin
            indexes = []
            for idx, g in enumerate(gt):
                if 'O' in g:
                    indexes.append(idx)
                    l_gt[idx] = list(map(lambda x: x if x == label else 'O', gt[idx]))
                    l_preds[idx] = list(map(lambda x: x if x == label else 'O', predictions[idx]))
        
            results_per_class[label] = {
                "precision": precision_score(np.take(l_gt, indexes), np.take(l_preds, indexes)),
                "recall": recall_score(np.take(l_gt, indexes), np.take(l_preds, indexes)),
                "f1": f1_score(np.take(l_gt, indexes), np.take(l_preds, indexes)),
            }
        for cl, metrics in sorted(list(results_per_class.items())):
            print("%s: %s" % (cl, metrics))

        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")

        store_predictions(predictions, gt, output_test_predictions_file, os.path.join(args.data_dir[0], "test.txt")) # first data dir is used for testing

main() if __name__ == '__main__' else True
