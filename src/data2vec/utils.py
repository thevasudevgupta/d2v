import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import yaml
from flax import traverse_util
from flax.serialization import from_bytes, to_bytes

from .constants import CONFIG_PATH, IGNORE_INDEX, MODEL_PATH


def read_yaml(path):
    return yaml.safe_load(open(path, "r"))


def cross_entropy(logits, labels, ignore_index=IGNORE_INDEX):
    """
    Args:
        logits: bsz, seqlen, vocab_size
        labels: bsz, seqlen
    """
    loss_mask = labels != ignore_index

    vocab_size = logits.shape[-1]
    labels = (labels[..., None] == jnp.arange(vocab_size)[None]).astype("f4")
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(labels * logits, axis=-1)

    loss = jnp.where(loss_mask, loss, 0).sum()
    return loss / jnp.sum(loss_mask)


def custom_save_fn(
    save_dir,
    params,
    config_dict,
    tokenizer_save_fn,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    with open(save_dir / MODEL_PATH, "wb") as f:
        f.write(to_bytes(params))

    with open(save_dir / CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_dict, indent=2, sort_keys=True) + "\n")

    tokenizer_save_fn(save_dir)


def hf_save_fn(
    save_dir,
    params,
    model_save_fn,
    tokenizer_save_fn,
    push_to_hub=False,
):
    model_save_fn(save_dir, params=params, push_to_hub=push_to_hub)
    tokenizer_save_fn(save_dir, push_to_hub=push_to_hub)


def linear_scheduler_with_warmup(lr, init_lr, warmup_steps, num_train_steps):
    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=init_lr, end_value=lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=lr, end_value=1e-7, transition_steps=decay_steps
    )
    lr = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )
    return lr


def create_tx(lr, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {
            k: (k[-1] != "bias" and k[-2:] != ("LayerNorm", "scale"))
            for k in params.keys()
        }
        return traverse_util.unflatten_dict(mask)

    tx = optax.adamw(
        learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask
    )
    return tx
