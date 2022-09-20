import copy
import math
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import flax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax.training import train_state
from transformers import AutoTokenizer

from data2vec.constants import HF_TOKEN, IGNORE_INDEX
from data2vec.data2vec_text import (Data2VecTextStudent, Data2VecTextStudentConfig, Data2VecTextTeacher, Data2VecTextTeacherConfig,
                                    ema_step)
from data2vec.training import (BaseConfig, Trainer, TrainerConfig,
                               TrainingStepOutput, ValidationStepOutput)
from data2vec.utils import (create_tx, custom_save_fn,
                            linear_scheduler_with_warmup, read_yaml)


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

        y = state.apply_fn_teacher(
            {"params": state.teacher_params},
            target_ids,
            attention_mask,
            deterministic=True,
            rngs=None,
            method=None,
        )

        masked_indices = input_ids == state.mask_token_id

        # TODO: following raises the error:
        # https://github.com/google/jax/issues/2765
        # x = x.at[masked_indices].get()
        # y = y.at(masked_indices).get()

        # TODO: it's definitely expensive to calculdate loss over all the items
        # and then eliminate all positions other than masked indices
        # but do we have a choice here?
        loss = state.loss_fn(x, y)

        loss = jnp.where(masked_indices, loss, 0).sum()

        # taking mean is fine as long as batches are equally distributed
        # TODO: check if data2vec authors are doing mean/sum
        return loss / jnp.sum(masked_indices)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")

    decay = state.get_decay()
    teacher_params =  jax.lax.cond(decay < 1, lambda: state.ema_step(decay), lambda: state.teacher_params)

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

    masked_indices = input_ids == state.mask_token_id
    loss = state.loss_fn(x, y)

    loss = jnp.where(masked_indices, loss, 0).sum()
    loss = loss / jnp.sum(masked_indices)

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
        input_ids, labels = self.mask_tokens(
            inputs.pop("input_ids"), special_tokens_mask
        )

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
    ema_anneal_end_step: float
    total_steps: int

    mask_token_id: int

    apply_fn_teacher: Callable = flax.struct.field(pytree_node=False)

    loss_fn: Callable = flax.struct.field(pytree_node=False)
    lr_scheduler: Callable = flax.struct.field(pytree_node=False)

    def ema_step(self, decay):
        # TODO: try to understand how jit will handle floats & ints under the hood        
        return ema_step(
            teacher_params=self.teacher_params,
            student_params=self.params,
            decay=decay, 
            teacher_dtype=jnp.float32,
        )

    def get_decay(self):
        return jax.lax.cond(
            self.ema_start_decay == self.ema_end_decay,
            lambda: self.ema_start_decay,
            self._get_decay,
        )

    def _get_decay(self):
        return jax.lax.cond(
            self.step >= self.ema_anneal_end_step,
            lambda: self.ema_end_decay,
            self.get_annealed_rate,
        )

    def get_annealed_rate(self):
        r = self.ema_end_decay - self.ema_start_decay
        pct_remaining = 1 - self.step / self.total_steps
        return self.ema_end_decay - r * pct_remaining

configs_dict = read_yaml("config.yaml")
rngs = jax.random.PRNGKey(0)
print(configs_dict)
print(jax.devices())

common_config = configs_dict["common"]

dtype = jnp.float16 if common_config.pop("is_fp16") else jnp.float32
tokenizer_id = common_config.pop("tokenizer_id")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

if common_config.pop("vocab_size") is None:
    common_config["vocab_size"] = tokenizer.vocab_size

student_config = {**common_config, **configs_dict["student"]}
teacher_config = {**common_config, **configs_dict["teacher"]}

student = Data2VecTextStudent(Data2VecTextStudentConfig(**student_config), dtype=dtype)
teacher = Data2VecTextTeacher(Data2VecTextTeacherConfig(**teacher_config), dtype=dtype)

# initializing the model weights
init_rngs, rngs = jax.random.split(rngs, num=2)

student_rngs, teacher_rngs = jax.random.split(init_rngs)
input_ids, attn_mask = jnp.ones((2, 3), dtype="i4"), jnp.ones((2, 3), dtype="i4")
student_params = student.init(student_rngs, input_ids, attn_mask)["params"]

# TODO: there is a possibility of efficient way here
# teacher may have some extra layer compared to teacher 
# and hence we would have to init teacher separately
teacher_params = teacher.init(teacher_rngs, input_ids, attn_mask)["params"]
# but we want to initialize same parameters for the layers which are common in teacher & student
teacher_params = unflatten_dict({**flatten_dict(teacher_params), **flatten_dict(student_params)})

# TODO: why we need to unfreeze here???
student_params = flax.core.unfreeze(student_params)

# print(model.tabulate(init_rngs, input_ids, attn_mask))

datacollator_config = DataCollatorForMLMConfig.from_dict(configs_dict["data_collator"])
collate_fn = DataCollatorForMLM(datacollator_config, tokenizer)

# TODO: we need to save teacher model as well
save_fn = partial(
    custom_save_fn,
    config_dict=student.config.to_dict(),
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
    apply_fn=student.apply,
    params=student_params,
    tx=tx,
    loss_fn=smooth_l1_loss,
    lr_scheduler=lr_scheduler,
    # data2vec specifc arguments
    teacher_params=teacher_params,
    ema_anneal_end_step=num_steps,
    ema_start_decay=configs_dict["ema"]["ema_start_decay"],
    ema_end_decay=configs_dict["ema"]["ema_end_decay"],
    total_steps=num_steps,
    apply_fn_teacher=teacher.apply,
    mask_token_id=tokenizer.mask_token_id,
)

new_state = trainer.train(state, rngs, train_data, val_data, wandb_configs=configs_dict)

# checkout if masking is working properly
