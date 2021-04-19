'''
Training script for Humor Detection
'''
import logging
import os
import json
import argparse
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from utils import convert_dataset_to_features

from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
#from tqdm.notebook import tqdm

from transformers import AdamW, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification

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
    parser.add_argument("--json", metavar='JSON', type=str, required=True,
                        help='json of arguments listed below')

    parser.add_argument("--data_dir", default=None, type=str,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Training parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
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
    parser.add_argument("--epochs", default=10.0, type=float,
                        help="Maximum number of training epochs to perform.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=100,
                        help="random seed for initialization")
    parser.add_argument('--ambiguity_fn', action='store_true', default="none",
                        help='Ambiguity function. none, wn (for WordNet), or csi (for course sense inventory)')
    # Model parameters
    parser.add_argument('--bert_base', action='store_true', default=False,
                        help='loads in bert-base instead of our custom model.')
    parser.add_argument('--rnn_size', type=int, default=768,
                        help='Hidden dimension of each direction of the bi-LSTM.')
    parser.add_argument('--grad_steps', type=int, default=1)

    args = parser.parse_args()

    return args

def train(args, dataset, eval_dataset, model):
    # Trains the model

    # Loaders
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    # Optimizer and scheduler
    optim = AdamW(model.parameters(), lr=args.learning_rate)

    total_batch_size = args.batch_size * args.grad_steps
    steps_per_ep = len(dataset) // total_batch_size

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", total_batch_size)

    # Holders for loss and acc
    global_step = 0
    tr_loss = 0.0
    train_iterator = trange(int(args.epochs), desc='Epoch')

    all_acc = []
    all_train_loss = []
    all_eval_loss = []

    for epoch in train_iterator:
        model.train()

        ep_loss = 0.0
        ep_step = 0
        optim.zero_grad()

        for step, batch in enumerate(train_loader):

            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[2],
                      'attention_mask': batch[3],
                      'labels': batch[4]}

            if not args.bert_base:
                inputs['ambiguity_scores'] = batch[1]

            outputs = model(**inputs)
            loss = outputs[0]

            if args.grad_steps > 1:
                loss = loss / args.grad_steps

            loss.backward()

            tr_loss += loss.item()
            ep_loss += loss.item()

            if (step+1) % args.grad_steps == 0:
                optim.step()
                optim.zero_grad()
                global_step += 1
                ep_step += 1

                if global_step % 10 == 0:
                    logger.info('epoch {}, {}/{}, Loss: {}'.format(epoch+1, ep_step, steps_per_ep,
                                                                   round(ep_loss / ep_step, 5)))

        model.eval()
        results = evaluate(args, eval_dataset, model, save=False)

        all_train_loss.append(round(ep_loss / ep_step,5))
        all_eval_loss.append(round(results['eval_loss'], 5))
        all_acc.append(results['acc'])

        if all_acc[-1] == max(all_acc):
            logger.info("Saving model checkpoint to %s", args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "state_dict.pt"))

        for i in range(len(all_train_loss)):
            logger.info('Epoch {}, Train Loss: {}, Eval Loss: {}, Eval Acc: {}'.format(i+1,
                                                               all_train_loss[i],
                                                               all_eval_loss[i],
                                                               all_acc[i]))

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for i in range(len(all_train_loss)):
            out = 'Epoch {}, Train Loss: {}, Eval Loss: {}, Eval Acc: {}'.format(i+1,
                                                               all_train_loss[i],
                                                               all_eval_loss[i],
                                                               all_acc[i])
            print(out, file=writer)

    return global_step, tr_loss / global_step, results


def train_trainer(args, train_dataset, eval_dataset, model):
    ## TODO: warmup ratio to steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        no_cuda=args.no_cuda
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

    for step, batch in enumerate(eval_loader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[2],
                      'attention_mask': batch[3],
                      'labels': batch[4]}

            if not args.bert_base:
                inputs['ambiguity_scores'] = batch[1]

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

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            if save:
                writer.write("%s = %s\n" % (key, str(results[key])))

    return results


def compute_metrics(preds, labels):
    '''
    Used to calculate the different evaluation metrics for our task(s)
    '''
    results = {}
    results['acc'] = accuracy_score(preds, labels)
    return results


def main():
    # Json based args parsing
    args = parse_args()
    with open(args.json) as f:
        a = json.load(f)
        args.__dict__.update(a)

    if args.data_dir is None:
        raise ValueError('Error: data_dir (Data Directory) must be specified in args.json.')
    if args.output_dir is None:
        raise ValueError('Error: output_dir (Output Directory) must be specified in args.json.')

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

    use_ambiguity = args.ambiguity_fn != "none"
    if args.do_train:
        # build the model
        logger.info('Loading in the Humor Detection model')
        if not args.bert_base:
            logger.info('Using custom model')
            model = HumorDetectionModel(rnn_size=args.rnn_size, use_ambiguity=use_ambiguity)
        else:
            logger.info('Loading in standard bert-base-uncased -- baseline testing')
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        model.to(args.device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Build dataset and train
        train_dataset = HumorDetectionDataset(args.data_dir, args.max_seq_length, 'train', args.ambiguity_fn)
        eval_dataset = HumorDetectionDataset(args.data_dir, args.max_seq_length, 'dev', args.ambiguity_fn)

        #print('Trainer attempt')
        #train_trainer(args, train_dataset, eval_dataset, model)

        logger.info('Training: learning_rate = %s, batch_size = %s', args.learning_rate, args.batch_size)
        global_step, tr_loss, results = train(args, train_dataset, eval_dataset, model)

        # create output directory
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # Save - model, tokenizer, args
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(args.output_dir)

    ## TODO: evaluation - with no training
    if args.do_eval:
        pass

    return


if __name__ == '__main__':
    main()
