import copy
import math
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax.training import train_state
from transformers import AutoTokenizer

from data2vec.data2vec_text import Data2VecTextModel, Data2VecTextModelConfig

from data2vec.constants import HF_TOKEN, IGNORE_INDEX
from data2vec.data2vec_text import ema_step
from data2vec.training import (BaseConfig, Trainer, TrainerConfig, TrainingStepOutput,
                       ValidationStepOutput)
from data2vec.utils import (create_tx, custom_save_fn, linear_scheduler_with_warmup,
                    read_yaml)

MASKED_INDEX = 0


def smooth_l1_loss(x, y, beta=4):
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    l1_loss = jnp.sum(jnp.abs(y - x), axis=-1)
    l2_loss = jnp.sum(jnp.square(y - x), axis=-1) / (2 * beta)

    loss = jnp.where(l1_loss < beta, l2_loss, l1_loss)
    return loss


def training_step(
    state: train_state.TrainState,
    dropout_rng: jnp.DeviceArray,
    batch: Dict[str, jnp.DeviceArray],
) -> TrainingStepOutput:
    new_drp_rng, drp_rng = jax.random.split(dropout_rng, num=2)

    def loss_fn(params):
        target_ids = batch.pop("target_ids")
        attention_mask = batch.pop("attention_mask")
        input_ids = batch.pop("input_ids")

        x = state.apply_fn(
            {"params": params},
            input_ids,
            attention_mask,
            deterministic=False,
            rngs={"dropout": drp_rng},
        )

        y = state.apply_fn(
            {"params": state.teacher_params},
            target_ids,
            attention_mask,
            deterministic=True,
            rngs=None,
            method=state.teacher_fn,
        )

        masked_indices = input_ids == MASKED_INDEX
        x = x.at[masked_indices].get()
        y = y.at(masked_indices).get()

        # taking mean is fine as long as batches are equally distributed
        # TODO: check if data2vec authors are doing mean/sum
        return state.loss_fn(x, y).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")

    teacher_params = state.ema_step(teacher_params, state.params)
    new_state = state.apply_gradients(grads=grads, teacher_params=teacher_params)

    return TrainingStepOutput(
        state=new_state,
        dropout_rng=new_drp_rng,
        loss=jax.lax.pmean(loss, axis_name="batch"),
        lr=state.lr_scheduler(state.step),
    )


def validation_step(
    state: train_state.TrainState, batch: Dict[str, jnp.DeviceArray]
) -> ValidationStepOutput:

    target_ids = batch.pop("target_ids")
    attention_mask = batch.pop("attention_mask")
    input_ids = batch.pop("input_ids")

    x = state.apply_fn(
        {"params": state.params},
        input_ids,
        attention_mask,
        deterministic=True,
        rngs=None,
    )

    y = state.apply_fn(
        {"params": state.teacher_params},
        target_ids,
        attention_mask,
        deterministic=True,
        rngs=None,
        method=state.teacher_fn,
    )

    masked_indices = input_ids == MASKED_INDEX
    x = x.at[masked_indices].get()
    y = y.at(masked_indices).get()

    loss = state.loss_fn(x, y).mean()

    return ValidationStepOutput(loss=loss)


class DataCollatorForMLMConfig(BaseConfig):
    max_length: int
    mlm_probability: float


class DataCollatorForMLM:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]):
        articles = [sample["text"] for sample in batch]
        inputs = self.tokenizer(
            articles,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
            return_special_tokens_mask=True,
        )

        special_tokens_mask = inputs.pop("special_tokens_mask")
        input_ids, labels = self.mask_tokens(inputs.pop("input_ids"), special_tokens_mask)

        return {
            "input_ids": input_ids,
            "target_ids": labels,
            "attention_mask": inputs.pop("attention_mask"),
        }

    def mask_tokens(
        self, input_ids: np.ndarray, special_tokens_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens input_ids/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = input_ids.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.config.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = IGNORE_INDEX  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool")
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(
            "bool"
        )
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(
            self.tokenizer.vocab_size, size=labels.shape, dtype="i4"
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels


class TrainState(train_state.TrainState):
    # data2vec specific extra state
    teacher_params: flax.core.FrozenDict[str, Any]
    ema_start_decay: float
    ema_end_decay: float
    total_steps: int
    teacher_fn: Callable = flax.struct.field(pytree_node=False)

    loss_fn: Callable = flax.struct.field(pytree_node=False)
    lr_scheduler: Callable = flax.struct.field(pytree_node=False)

    def ema_step(self, teacher_params, student_params):
        # TODO: try to understand how jit will handle floats & ints under the hood
        decay = self.get_decay()
        return ema_step(
            teacher_params, student_params, decay=decay, teacher_dtype=jnp.float32
        )

    def get_decay(self):
        r = self.ema_end_decay - self.ema_start_decay
        pct_remaining = 1 - self.step / self.total_steps
        return self.ema_end_decay - r * pct_remaining


configs_dict = read_yaml("config.yaml")
rngs = jax.random.PRNGKey(0)
print(configs_dict)
print(jax.devices())


model_config = configs_dict["model"]
dtype = jnp.float16 if model_config.pop("is_fp16") else jnp.float32
tokenizer_id = model_config.pop("tokenizer_id")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

if model_config.pop("vocab_size") is None:
    model_config["vocab_size"] = tokenizer.vocab_size

model_config = Data2VecTextModelConfig(**model_config)
model = Data2VecTextModel(model_config, dtype=dtype)

# initializing the model weights
init_rngs, rngs = jax.random.split(rngs, num=2)
input_ids, attn_mask = jnp.ones((2, 3), dtype="i4"), jnp.ones((2, 3), dtype="i4")
variables = model.init(init_rngs, input_ids, attn_mask)

# print(model.tabulate(init_rngs, input_ids, attn_mask))

datacollator_config = DataCollatorForMLMConfig.from_dict(configs_dict["data_collator"])
collate_fn = DataCollatorForMLM(datacollator_config, tokenizer)

save_fn = partial(
    custom_save_fn,
    config_dict=model.config.to_dict(),
    tokenizer_save_fn=tokenizer.save_pretrained,
)

trainer_config = TrainerConfig.from_dict(configs_dict["trainer"])
trainer = Trainer(
    trainer_config,
    training_step,
    validation_step,
    train_pmap_kwargs={"axis_name": "batch", "donate_argnums": (0, 1)},
    val_pmap_kwargs={"axis_name": "batch"},
    collate_fn=collate_fn,
    model_save_fn=save_fn,
)

# TODO: modify before starting main pretraining
dataset = load_dataset("wikipedia", "20220301.simple", split="train")
train_data, val_data = dataset, dataset
print(train_data, val_data)

# we are dropping the last batch for now
batch_size = trainer_config.batch_size_per_device * jax.device_count()
num_steps = math.ceil(len(train_data) // batch_size) * trainer_config.max_epochs

lr_scheduler = linear_scheduler_with_warmup(
    configs_dict["optax"]["lr"],
    configs_dict["optax"]["init_lr"],
    configs_dict["optax"]["warmup_steps"],
    num_steps,
)
tx = create_tx(lr_scheduler, configs_dict["optax"]["weight_decay"])

# we don't need to maintain separate state of teacher model
# as teacher model doesn't have trainable parameters
state = TrainState.create(
    apply_fn=model.apply,
    params=flax.core.unfreeze(variables["params"]),  # TODO: why we need to unfreeze here???
    tx=tx,
    loss_fn=smooth_l1_loss,
    lr_scheduler=lr_scheduler,
    # data2vec specifc arguments
    teacher_params=variables["params"],
    ema_start_decay=configs_dict["ema"]["ema_start_decay"],
    ema_end_decay=configs_dict["ema"]["ema_end_decay"],
    total_steps=num_steps,
    teacher_fn=model.extract_features,
)

new_state = trainer.train(state, rngs, train_data, val_data, wandb_configs=configs_dict)

# implement average of topk layers
# control more on decay using jax condition
# checkout if masking is working properly
