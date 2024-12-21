import re
import torch

def levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def compute_ser(y_true: list[list[str]], y_pred: list[list[str]]) -> float:
    ed_acc = 0
    length_acc = 0
    for t, h in zip(y_true, y_pred):
        ed_acc += levenshtein(t, h)
        length_acc += len(t)
    return 100.0 * ed_acc / length_acc


def compute_metrics(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> dict[str, float]:
    metrics = {"ser": compute_ser(y_true, y_pred)}
    return metrics
    
def filter_pred(x, vocab):
    return (x != vocab.c2i["<PAD>"]) and (x != vocab.c2i["<BLANK>"])
    
def split_sequence(sequence):
    return [re.split(r"\s+|:", " ".join(sample)) for sample in sequence]
    
def ctc_greedy_decoder(y_pred, vocab):
    """
    Greedy decoder for CTC batch predictions.

    Args:
        y_pred (torch.Tensor): Batch of predictions.
        vocab (Vocabulary): Vocabulary object.

    Returns:
        List[List[str]]: Decoded batch.
    """
    y_pred_decoded = [
        torch.argmax(i, dim=1) for i in y_pred
    ]  # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len)
    y_pred_decoded = [
        torch.unique_consecutive(sample, dim=0).tolist() for sample in y_pred_decoded
    ]

    y_pred_decoded = [
        [vocab.i2c[i] for i in sample if filter_pred(i, vocab)]
        for sample in y_pred_decoded
    ]

    return split_sequence(y_pred_decoded)