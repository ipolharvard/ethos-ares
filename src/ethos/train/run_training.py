import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch as th
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, GPT2Config

from ethos.model import GPT2LMNoBiasModel

from ..constants import SpecialToken as ST
from ..datasets import TimelineDataset
from ..utils import load_model_checkpoint, setup_torch
from .metrics import estimate_loss
from .utils import ModelType, configure_optimizers, estimate_mfu, get_lr, make_infinite_loader


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig):
    """This training script can be run both on a single gpu in debug mode, and also in a larger
    training run with distributed data parallel (ddp).

    To run on a single GPU, example:
    $ ethos_train [args...]

    To run with DDP on 4 gpus on 1 node, example:

    $ torchrun --standalone --nproc_per_node=4 ethos_train [args...]

    To run with DDP on 4 gpus across 2 nodes, example:

    - Run on the first (master) node with example IP 123.456.123.456:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456
     --master_port=1234 ethos_train [args...]

    - Run on the worker node:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456
     --master_port=1234 ethos_train [args...]

    (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
    """
    model_type = ModelType(cfg.model_type)

    device = cfg.device
    out_dir = Path(cfg.out_dir)
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        th.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0

    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.n_positions

    if master_process:
        logger.info(f"Tokens per iteration per worker: {tokens_per_iter:,}")
        out_dir.mkdir(parents=True, exist_ok=True)
    ctx = setup_torch(device, cfg.dtype, 42 + seed_offset)

    train_dataset = TimelineDataset(
        cfg.data_fp,
        n_positions=cfg.n_positions,
        is_encoder_decoder=model_type == ModelType.ENC_DECODER,
    )
    vocab = train_dataset.vocab

    vocab_size = (len(vocab) // 64 + 1) * 64 if len(vocab) % 64 != 0 else len(vocab)
    tokens_of_interest = [ST.DEATH, ST.ADMISSION, ST.DISCHARGE]
    tokens_of_interest = {stoken: vocab.encode(stoken) for stoken in tokens_of_interest}

    # DATASETS
    train_dataset, val_dataset = train_dataset.train_test_split(cfg.val_size)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    train_dataloader = make_infinite_loader(train_dataloader)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = make_infinite_loader(val_dataloader)

    eval_iters = len(val_dataset) // (cfg.batch_size * cfg.n_positions) + 1
    if master_process:
        logger.info(
            "Train dataset size: {:,}, Val dataset size: {:,} (eval_iters={})".format(
                len(train_dataset), len(val_dataset), eval_iters
            )
        )

    def get_batch(split) -> tuple[th.Tensor | tuple, th.Tensor]:
        data = train_dataloader if split == "train" else val_dataloader
        x, y = next(data)
        y = y.to(device, non_blocking=True)
        if isinstance(x, list):
            return (x[0].to(device, non_blocking=True), x[1].to(device, non_blocking=True)), y
        return x.to(device, non_blocking=True), y

    iter_num, best_val_loss, best_metric_score, optimizer_state, wandb_path = 0, 1e9, 0, None, None
    if cfg.resume:
        model_fp = out_dir / "recent_model.pt"
        logger.info(f"Resuming from the most recent model: {model_fp}")
        raw_model, checkpoint = load_model_checkpoint(model_fp, map_location=device)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        best_metric_score = checkpoint["best_metric_score"]
        optimizer_state = checkpoint["optimizer"]
        wandb_path = checkpoint["wandb_path"]
    else:
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=cfg.n_positions,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_inner=None,
            activation_function=cfg.activation,
            resid_pdrop=cfg.dropout,
            embd_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
            bias=False,
        )
        if model_type == ModelType.ENC_DECODER:
            encoder_config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=cfg.n_embd,
                num_hidden_layers=1,
                num_attention_heads=cfg.n_head,
                intermediate_size=cfg.n_embd * 4,
                hidden_act=cfg.activation,
                hidden_dropout_prob=cfg.dropout,
                attention_probs_dropout_prob=cfg.dropout,
                max_position_embeddings=train_dataset.dataset.context_size,
                max_length=train_dataset.dataset.context_size,
                is_encoder_decoder=True,
                use_bfloat16=True,
            )
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, config)
            raw_model = EncoderDecoderModel(config=config)
        else:
            raw_model = GPT2LMNoBiasModel(config)

        if master_process:
            logger.info(f"Initializing a new model from scratch: {config}")
    logger.info(f"Model parameters: {raw_model.num_parameters() / 1e6:.2f}M")

    raw_model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = th.amp.GradScaler(enabled=(cfg.dtype == "float16"))
    # optimizer
    optimizer = configure_optimizers(
        raw_model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    num_params = raw_model.num_parameters()
    if master_process:
        logger.info(f"Number of parameters: {num_params / 1e6:.2f}M")
        logger.info(("Not c" if cfg.no_compile else "C") + "ompiling the model...")
    model = th.compile(raw_model, disable=cfg.no_compile)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    online_logger, wandb_run = None, None
    if cfg.wandb_log and master_process:
        import wandb

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        dataset_name = Path(cfg.data_fp).parts[-2]
        cfg_dict.update(
            {
                "dataset": dataset_name,
                "vocab_size": len(vocab),
                "vocab_size_train": vocab_size,
                "model_num_params": num_params,
                "model_num_params_total": raw_model.num_parameters(exclude_embeddings=False),
            }
        )
        run_id = wandb_path.split("/")[-1] if wandb_path is not None else None
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg_dict,
            tags=[dataset_name],
            resume_from=f"{run_id}?_step={iter_num}" if run_id is not None else None,
        )
        online_logger = wandb

    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, get_batch, eval_iters, tokens_of_interest)
            logger.info(
                "step {}: train loss {:.4f}, val loss {:.4f}".format(
                    iter_num,
                    losses["loss/train"],
                    losses["loss/val"],
                )
            )
            checkpoint = {
                "iter_num": iter_num,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": losses["loss/val"],
                "best_metric_score": best_metric_score,
                "model_config": raw_model.config,
                "vocab": vocab.stoi,
                "model_type": str(model_type),
                "wandb_path": wandb_run.path if wandb_run is not None else None,
            }
            th.save(checkpoint, out_dir / "recent_model.pt")
            logger.info("Saved the most recent model.")
            if losses["loss/val"] < best_val_loss:
                th.save(checkpoint, out_dir / "best_model.pt")
                logger.info(f"Saved the best model: {best_val_loss} => {losses['loss/val']}")
                best_val_loss = losses["loss/val"]

            metric_score = np.mean(
                [v for k, v in losses.items() if k.startswith("acc_top/") and "/all/" not in k]
            )
            if metric_score > best_metric_score:
                th.save(checkpoint, out_dir / "toi_metric_model.pt")
                logger.info(f"Saved the best metrics model: {best_metric_score} => {metric_score}")
                best_metric_score = metric_score

            if online_logger is not None:
                epochs = iter_num * tokens_per_iter / len(train_dataset)
                online_logger.log(
                    {
                        "other/iter": iter_num,
                        "other/lr": lr,
                        "other/mfu": running_mfu * 100,
                        "other/toi_avg_score": metric_score,
                        "other/epochs": epochs,
                        **losses,
                    }
                )

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(cfg.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == cfg.gradient_accumulation_steps - 1
            with ctx:
                if isinstance(X, tuple):
                    output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                else:
                    output = model(input_ids=X, labels=Y)
                loss = output.loss
                loss = (
                    loss / cfg.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above,
            # approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.gradient_accumulation_steps
            if local_iter_num >= 5 and model_type == ModelType.DECODER:
                mfu = estimate_mfu(
                    raw_model, num_params, cfg.batch_size * cfg.gradient_accumulation_steps, dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            logger.info(
                f"[{iter_num}]: loss={lossf:.4f}, time={dt * 1000:.0f}ms, mfu={running_mfu:.2%}"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > cfg.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
