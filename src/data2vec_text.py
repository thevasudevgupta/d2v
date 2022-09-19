from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModule
from transformers.models.roberta.configuration_roberta import RobertaConfig

import jax
import flax.linen as nn
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

# TODO: make sure initialization is happening correctly

@dataclass
class Data2VecTextModelConfig:
    encoder: RobertaConfig
    num_head_layers: int = 2
    approximate_gelu: bool = True


class Data2VecTextModel(nn.Module):
    config: Data2VecTextModelConfig
    dtype: jnp.dtype = jnp.float32  # dtype of the computation

    def setup(self):
        self.encoder = FlaxRobertaModule(self.config.encoder, dtype=self.dtype)

        encoder_hidden_size = self.config.encoder.hidden_size
        head_layers = [nn.Dense(encoder_hidden_size * 2, dtype=self.dtype) for _ in range(self.config.num_head_layers - 1)]
        self.head_layers = head_layers + [nn.Dense(encoder_hidden_size, dtype=self.dtype)]

    def __call__(self, input_ids, attention_mask, deterministic: bool = True):
        hidden_states = self.encoder(input_ids, attention_mask, deterministic=deterministic)
        for layer in self.head_layers:
            hidden_states = layer(hidden_states)
            hidden_states = nn.gelu(hidden_states, approximate=self.config.approximate_gelu)
        return hidden_states


@dataclass
class EMAConfig:
    decay: float
    fp32: bool = True

# TODO: how do we apply filter
# we want to ema only the transformers layers??
def ema_step(
    teacher_params,
    student_params,
    decay: jnp.ndarray,
    teacher_dtype: jnp.dtype = jnp.float32
):
    """
        stateless following JAX philosophy

        takes current params of the teacher model and latest params of the student model,
        update them and returns the new params of the teacher model
    """
    def _step_internal(teacher_param, student_param):
        teacher_param = teacher_param.astype(teacher_dtype)
        return teacher_param * decay + student_param.astype(teacher_param.dtype) * (1 - decay)

    return jax.tree_map(_step_internal, teacher_params, student_params)


class Data2VecFramework:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids, attention_mask, deterministic: bool = True, labels: Optional[jnp.ndarray] = None):

        return
