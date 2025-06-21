import os

import torch as th
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedModel


@th.inference_mode()
def estimate_loss(
    model: PreTrainedModel, ctx, train_dataset: Dataset, val_dataset: Dataset, cfg
) -> dict:
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1

    train_dataloader, val_dataloader = (
        DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=not is_distributed,
            sampler=DistributedSampler(dataset) if is_distributed else None,
        )
        for dataset in [train_dataset, val_dataset]
    )

    eval_iters = len(val_dataset) // (cfg.batch_size * cfg.n_positions) + 1
    if is_distributed:
        eval_iters //= int(os.environ["WORLD_SIZE"])

    out = {}
    for split, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        losses = th.empty(eval_iters, device=model.device)
        for i, (X, Y) in zip(range(eval_iters), dataloader):
            with ctx:
                if isinstance(X, tuple):
                    output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                else:
                    output = model(input_ids=X, labels=Y)
                loss = output.loss
            losses[i] = loss.item()

        out[f"loss/{split}"] = losses.mean().item()
    return out
