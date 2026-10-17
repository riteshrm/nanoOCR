import torch
import argparse
from src.data.dataset import RECDataset, rec_collate
from src.rec_model.model import CRNN2
from src.utils.utils import ctc_decoder, get_latest_checkpoint
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


def alignment_counts(pred, truth):
    """Return edit errors and matched characters from a minimum-edit alignment.

    Among equally short alignments, prefer the one with more exact matches.
    A substitution counts as both a false positive and a false negative.
    """
    row = [(j, 0) for j in range(len(truth) + 1)]

    for i, p in enumerate(pred, 1):
        new_row = [(i, 0)]
        for j, g in enumerate(truth, 1):
            candidates = [
                (row[j][0] + 1, row[j][1]),
                (new_row[j - 1][0] + 1, new_row[j - 1][1]),
                (row[j - 1][0] + int(p != g),
                row[j - 1][1] + int(p == g)),
            ]
            new_row.append(min(candidates, key=lambda counts: (counts[0], -counts[1])))
        row = new_row

    return row[-1]


def evaluation_counts(predictions, targets, lengths):
    truths = torch.split(targets, lengths.tolist())
    if len(predictions) != len(truths):
        raise ValueError("Expected one prediction per ground-truth word")

    counts = {
        "errors": 0,
        "matches": 0,
        "predicted_chars": 0,
        "target_chars": 0,
        "exact_words": 0,
        "words": len(truths),
    }
    for pred, truth in zip(predictions, truths):
        pred, truth = pred.tolist(), truth.tolist()
        errors, matches = alignment_counts(pred, truth)
        counts["errors"] += errors
        counts["matches"] += matches
        counts["predicted_chars"] += len(pred)
        counts["target_chars"] += len(truth)
        counts["exact_words"] += int(pred == truth)
    return counts

def run(args):
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        path, _ = get_latest_checkpoint(args.ckpt_dir, "best-ckpt-")
        ckpt_path = f'{path}/pytorch_model.bin'
    if ckpt_path is None:
        print("No valid checkpoints found")
        return
    print("Loaded model from: ", ckpt_path)
    data = torch.load(f"{args.data_path}/vocab.pt", weights_only=False)
    
    vocab = data["vocab"]
    ctoi = data["ctoi"]
    itoc = data["itoc"]

    test_dataset = RECDataset(args.data_path, 'test.csv')


    device = "cuda" if torch.cuda.is_available() else "cpu"


    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=rec_collate)

    model = CRNN2(len(vocab)+1).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    totals = {
        "errors": 0,
        "matches": 0,
        "predicted_chars": 0,
        "target_chars": 0,
        "exact_words": 0,
        "words": 0,
    }
    for x, targets, target_lengths in tqdm(test_dataloader):
        x = x.to(device)
        with torch.no_grad():
            log_probs = model(x).log_softmax(-1)
        pred_texts, pred_labels = ctc_decoder(log_probs.permute(1, 0, 2), itoc)

        counts = evaluation_counts(
            pred_labels, targets, target_lengths
        )
        for name, count in counts.items():
            totals[name] += count

    cer = totals["errors"] / totals["target_chars"] if totals["target_chars"] else float("nan")
    # Precision is reported as zero when the model predicts no characters.
    precision = totals["matches"] / totals["predicted_chars"] if totals["predicted_chars"] else 0.0
    recall = totals["matches"] / totals["target_chars"] if totals["target_chars"] else 0.0
    word_accuracy = totals["exact_words"] / totals["words"] if totals["words"] else 0.0
    print(f"CER: {cer:.4f} ({cer:.2%})")
    print(f"Character precision: {precision:.2%}")
    print(f"Character recall: {recall:.2%}")
    print(f"Exact word-match accuracy: {word_accuracy:.2%} "
        f"({totals['exact_words']}/{totals['words']})")

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default='rec_data/processed')
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default='rec_checkpoints')

    args = parser.parse_args()
    
    run(args=args)

if __name__ == "__main__":
    main()