from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModule

# TODO: make sure initialization is happening correctly
# TODO: instead of mse loss, we can still try cross entropy loss (or say dot product)??
# but why do you think cross entropy loss would be better than mse??


class Data2VecTextModelConfig(RobertaConfig):
    num_head_layers: int = 2
    approximate_gelu: bool = True


class Data2VecTextModel(FlaxRobertaModule):
    """model weights can be directly loaded using `FlaxRobertaModel.from_pretrained(...)`"""

    config: Data2VecTextModelConfig
    dtype: jnp.dtype = jnp.float32  # dtype of the computation

    def setup(self):
        super().setup()

        hidden_size = self.config.hidden_size
        head_layers = [
            nn.Dense(hidden_size * 2, dtype=self.dtype)
            for _ in range(self.config.num_head_layers - 1)
        ]
        self.head_layers = head_layers + [
            nn.Dense(hidden_size, dtype=self.dtype)
        ]

    def __call__(self, input_ids, attention_mask, deterministic: bool = True):
        hidden_states = super().__call__(
            input_ids, attention_mask, deterministic=deterministic
        ).last_hidden_state
        for layer in self.head_layers:
            hidden_states = layer(hidden_states)
            hidden_states = nn.gelu(
                hidden_states, approximate=self.config.approximate_gelu
            )
        return hidden_states

    def extract_features(self, input_ids, attention_mask, deterministic: bool = True):
        hidden_states = super().__call__(
            input_ids, attention_mask, deterministic=deterministic
        ).last_hidden_state
        return hidden_states


# TODO: how do we apply filter
# we want to ema only the transformers layers??
def ema_step(
    teacher_params,
    student_params,
    decay: jnp.ndarray,
    teacher_dtype: jnp.dtype = jnp.float32,
):
    """
    stateless following JAX philosophy

    takes current params of the teacher model and latest params of the student model,
    update them and returns the new params of the teacher model
    """

    def _step_internal(teacher_param, student_param):
        teacher_param = teacher_param.astype(teacher_dtype)
        return teacher_param * decay + student_param.astype(teacher_param.dtype) * (
            1 - decay
        )

    return jax.tree_map(_step_internal, teacher_params, student_params)
