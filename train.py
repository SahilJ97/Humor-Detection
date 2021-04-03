'''
Training script for Humor Detection
'''
import logging
import os
import argparse
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from transformers import AdamW, Trainer, TrainingArguments, BertTokenizer

from model import HumorDetectionModel
from dataset import HumorDetectionDataset

logger = logging.getLogger(__name__)

# Sets the seed for reproducability
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    return


# Parses the arguments from an argument file
def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Learning rate for full train if provided.")
    parser.add_argument("--eval_per_epoch", action='store_true',
                        help="Run evaluation at each epoch during training.")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float,
                        help="Linear warmup over ratio% of train.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=4.0, type=float,
                        help="Maximum number of training epochs to perform.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=100,
                        help="random seed for initialization")

    parser.add_argument('--use_ambiguity', action='store_true', default=True,
                        help='use the ambiguity scoring during training.')
    parser.add_argument('--rnn_size', type=int, default=5,
                        help='hidden dimension of the RNN.')

    args = parser.parse_args()

    return args


def train(args, dataset, eval_dataset, model):
    # Trains the model

    # Loaders
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    # Optimizer and scheduler
    optim = AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)

    # Holders for loss and acc
    global_step = 0
    tr_loss = 0.0
    train_iterator = trange(int(args.epochs), desc='Epoch')

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc='Iteration')
        model.train()
        for step, batch in enumerate(epoch_iterator):
            optim.zero_grad()
            batch = tuple(t.to(args.device) for _,t in batch.items())

            inputs = {'token_indices': batch[0],
                      'ambiguity_scores': batch[1],
                      'attention_mask': batch[2],
                      'labels': batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

            tr_loss += loss.item()
            global_step += 1

            ## TODO: add loss & accuracy tqdm logger

        model.eval()
        end_of_train = args.epochs-1 == epoch
        results = evaluate(args, eval_dataset, model, save=end_of_train)

    return global_step, tr_loss / global_step, results

def train_trainer(args, train_dataset, eval_dataset, model):
    ## TODO: warmup ratio to steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epoch=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    return


'''
Runs evaluation for a model on the given dataset.
Returns a dictionary of computed metrics (acc, loss, f1, etc).
'''
def evaluate(args, dataset, model, save=False):
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)

    # Eval
    logger.info("***** Running Evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    # Holders for loss and acc
    eval_step = 0
    eval_loss = 0.0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'token_indices': batch[0],
                      'ambiguity_scores': batch[1],
                      'attention_mask': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()

        eval_step += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / eval_step
    preds = np.argmax(preds, axis=1)
    results = compute_metrics(preds, out_label_ids)
    results['eval_loss'] = eval_loss

    if save:
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    return results


'''
Used to calculate the different evaluation metrics for our task(s)
'''
def compute_metrics(preds, labels):
    results = {}
    results['acc'] = accuracy_score(preds, labels)
    return results


def main():
    args = parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    # TODO: train
    if args.do_train:
        # build the model
        model = HumorDetectionModel(rnn_size=args.rnn_size, use_ambiguity=args.use_ambiguity)
        model.to(args.device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Build dataset and train
        train_dataset = HumorDetectionDataset(os.path.join(args.data_dir, 'train_with_amb.tsv'), args.max_seq_length)
        eval_dataset = HumorDetectionDataset(os.path.join(args.data_dir, 'dev_with_amb.tsv'), args.max_seq_length)

        logger.info('Training: learning_rate = %s, batch_size = %s', args.learning_rate, args.batch_size)
        global_step, tr_loss, results = train(args, train_dataset, eval_dataset, model)

        # create output directory
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # Save - model, tokenizer, args
        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    ## TODO: evaluation - with no training
    if args.do_eval:
        pass

    return


if __name__ == '__main__':
    main()













