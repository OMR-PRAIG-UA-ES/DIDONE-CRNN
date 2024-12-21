import torch
import torch.nn.functional as F

def pad_batch_images(x: list[torch.Tensor]) -> torch.Tensor:
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x: list[torch.Tensor], pad_idx) -> torch.Tensor:
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack(
        [F.pad(i, pad=(0, max_length - i.shape[0]), value=pad_idx) for i in x], dim=0
    )
    x = x.type(torch.int32)
    return x
    
def ctc_batch_preparation(batch, padding_idx):
    x, xl, y, yl = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y, padding_idx)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl