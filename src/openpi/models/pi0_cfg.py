import logging
import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

from openpi.models.pi0_base import make_attn_mask, posemb_sincos

@dataclasses.dataclass(frozen=True)
class Pi0CFGConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    pi05: bool = False
    discrete_state_input: bool = None  # type: ignore

    # cfg setting
    cfg_dropout: float = 0.1

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05_CFG
        return _model.ModelType.PI0_CFG

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0_CFG":
        return Pi0_CFG(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.CFGObservation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.CFGObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                none_tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                none_tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),

            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0_CFG(_model.BaseModel):
    def __init__(self, config: Pi0CFGConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.cfg_dropout = config.cfg_dropout

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.CFGObservation, rng:at.KeyArrayLike, train:bool=False, uncond:bool=False
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        ### dropout text when train
        if train:
            bsz = obs.tokenized_prompt.shape[0]
            use_none = jax.random.bernoulli(rng, p=self.cfg_dropout, shape=(bsz,),)
            use_none_bt = use_none[:, None]
            token_ids = jnp.where(use_none_bt, obs.none_tokenized_prompt, obs.tokenized_prompt,)
            token_mask = jnp.where(use_none_bt, obs.none_tokenized_prompt_mask,obs.tokenized_prompt_mask,)
        
        ### inference (condition or uncondition)
        else:
            if uncond:
                token_ids = obs.none_tokenized_prompt
                token_mask = obs.none_tokenized_prompt_mask
            else:
                token_ids = obs.tokenized_prompt
                token_mask = obs.tokenized_prompt_mask

        tokenized_inputs = self.PaliGemma.llm(token_ids, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(token_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask



    @at.typecheck
    def embed_suffix(
        self, obs: _model.CFGObservation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond


    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.CFGObservation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_cfg_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # different from origin
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, rng=dropout_rng, train=train, uncond=False)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.CFGObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        guidance_scale: float = 1.5,
    ) -> _model.Actions:
        
        observation = _model.preprocess_cfg_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # conditional prefix
        prefix_tokens_c, prefix_mask_c, prefix_ar_mask_c = self.embed_prefix(observation, train=False, uncond=False, rng=rng)
        prefix_attn_mask_c = make_attn_mask(prefix_mask_c, prefix_ar_mask_c)
        positions_c = jnp.cumsum(prefix_mask_c, axis=1) - 1

        _, kv_cache_c = self.PaliGemma.llm(
            [prefix_tokens_c, None],
            mask=prefix_attn_mask_c,
            positions=positions_c,
        )

        # unconditional prefix
        prefix_tokens_u, prefix_mask_u, prefix_ar_mask_u = self.embed_prefix(observation, train=False, uncond=True, rng=rng)
        prefix_attn_mask_u = make_attn_mask(prefix_mask_u, prefix_ar_mask_u)
        positions_u = jnp.cumsum(prefix_mask_u, axis=1) - 1

        _, kv_cache_u = self.PaliGemma.llm(
            [prefix_tokens_u, None],
            mask=prefix_attn_mask_u,
            positions=positions_u,
        )


        def step(carry):
            x_t, time = carry
            # cond
            suffix_tokens_c, suffix_mask_c, suffix_ar_mask_c, adarms_cond_c = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask_c = make_attn_mask(suffix_mask_c, suffix_ar_mask_c)
            prefix_attn_mask_c2 = einops.repeat(prefix_mask_c, "b p -> b s p", s=suffix_tokens_c.shape[1])
            full_attn_mask_c = jnp.concatenate([prefix_attn_mask_c2, suffix_attn_mask_c], axis=-1)
            positions_c2 = (jnp.sum(prefix_mask_c, axis=-1)[:, None] + jnp.cumsum(suffix_mask_c, axis=-1)- 1)

            (_, suffix_out_c), _ = self.PaliGemma.llm(
                [None, suffix_tokens_c],
                mask=full_attn_mask_c,
                positions=positions_c2,
                kv_cache=kv_cache_c,
                adarms_cond=[None, adarms_cond_c],
            )
            v_cond = self.action_out_proj(suffix_out_c[:, -self.action_horizon :])

            # uncond
            suffix_tokens_u, suffix_mask_u, suffix_ar_mask_u, adarms_cond_u = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask_u = make_attn_mask(suffix_mask_u, suffix_ar_mask_u)
            prefix_attn_mask_u2 = einops.repeat(prefix_mask_u, "b p -> b s p", s=suffix_tokens_u.shape[1])
            full_attn_mask_u = jnp.concatenate([prefix_attn_mask_u2, suffix_attn_mask_u], axis=-1)

            positions_u2 = (jnp.sum(prefix_mask_u, axis=-1)[:, None]+ jnp.cumsum(suffix_mask_u, axis=-1)- 1)

            (_, suffix_out_u), _ = self.PaliGemma.llm(
                [None, suffix_tokens_u],
                mask=full_attn_mask_u,
                positions=positions_u2,
                kv_cache=kv_cache_u,
                adarms_cond=[None, adarms_cond_u],
            )

            v_uncond = self.action_out_proj(suffix_out_u[:, -self.action_horizon :])

            # CFG combine
            v = v_uncond + guidance_scale * (v_cond - v_uncond)

            return x_t + dt * v, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
