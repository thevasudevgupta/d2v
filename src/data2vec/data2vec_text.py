
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModule
from flax.traverse_util import flatten_dict, unflatten_dict

# TODO: make sure initialization is happening correctly
# TODO: instead of mse loss, we can still try cross entropy loss (or say dot product)??
# but why do you think cross entropy loss would be better than mse??


class Data2VecTextStudentConfig(RobertaConfig):
    num_head_layers: int = 2
    approximate_gelu: bool = True


class Data2VecTextStudent(FlaxRobertaModule):
    """model weights can be directly loaded using `FlaxRobertaModel.from_pretrained(...)`"""

    config: Data2VecTextStudentConfig
    dtype: jnp.dtype = jnp.float32  # dtype of the computation

    def setup(self):
        super().setup()

        hidden_size = self.config.hidden_size
        head_layers = [
            nn.Dense(hidden_size * 2, dtype=self.dtype)
            for _ in range(self.config.num_head_layers - 1)
        ]
        self.head_layers = head_layers + [nn.Dense(hidden_size, dtype=self.dtype)]

    def __call__(self, input_ids, attention_mask, deterministic: bool = True):
        outputs = super().__call__(
            input_ids,
            attention_mask,
            deterministic=deterministic,
        )
        hidden_states = outputs.last_hidden_state

        for layer in self.head_layers:
            hidden_states = layer(hidden_states)
            hidden_states = nn.gelu(
                hidden_states, approximate=self.config.approximate_gelu
            )
        return hidden_states


class Data2VecTextTeacherConfig(RobertaConfig):
    average_top_k_layers: int = 4


class Data2VecTextTeacher(FlaxRobertaModule):
    """model weights can be directly loaded using `FlaxRobertaModel.from_pretrained(...)`"""

    config: Data2VecTextTeacherConfig
    dtype: jnp.dtype = jnp.float32  # dtype of the computation

    def setup(self):
        super().setup()
        self.layer_norm_target_layer = nn.LayerNorm(dtype=jnp.float32)

    def __call__(self, input_ids, attention_mask, deterministic: bool = True):
        outputs = super().__call__(input_ids, attention_mask, deterministic=deterministic, output_hidden_states=True)

        outputs = outputs.hidden_states[-self.config.average_top_k_layers:]
        outputs = (self.layer_norm_target_layer(output) for output in outputs)

        # TODO: if sum operation efficient here?
        outputs = sum(outputs) / self.config.average_top_k_layers
        # TODO: do we need to give float inputs or setting computation dtype is enough??

        return outputs


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

    teacher_params = flatten_dict(teacher_params)
    student_params = flatten_dict(student_params)

    common_keys = set(student_params.keys()) & set(teacher_params.keys())
    extra_keys = set(teacher_params.keys()) - set(student_params.keys())
    teacher_extra_params = {k: v for k, v in teacher_params.items() if k in extra_keys}

    teacher_params = {k: v for k, v in teacher_params.items() if k in common_keys}
    student_params = {k: v for k, v in student_params.items() if k in common_keys}

    teacher_params = jax.tree_map(_step_internal, teacher_params, student_params)

    # TODO: is following efficient?
    teacher_params = {**teacher_extra_params, **teacher_params}

    return unflatten_dict(teacher_params)
