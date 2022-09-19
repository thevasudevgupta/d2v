import optax
import yaml
from flax import traverse_util


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