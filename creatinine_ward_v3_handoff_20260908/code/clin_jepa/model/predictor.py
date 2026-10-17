"""Action-conditioned Transformer predictor: the Clin-JEPA latent trajectory predictor.

Predicts the next-step patient state embedding from a history of state, action,
and demographics embeddings, all in the same 4096-dim encoder latent space.

Sequence layout (length ``2L + 1``): ``[d, s_1, a_1, s_2, a_2, ..., s_L, a_L]``,
with demographics ``d`` at position 0, state ``s_t`` at odd positions, action
``a_t`` at even positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_block_causal_mask(num_timesteps: int, device: torch.device) -> torch.Tensor:
    """Build the block-causal attention mask for the ``[d, s_1, a_1, ...]`` sequence.

    Position 0 (``d``) attends only to itself; every temporal token attends to
    ``d``; within timestep ``k`` the pair ``(s_k, a_k)`` is bidirectional; across
    timesteps ``k`` attends to ``1..k`` (causal).

    Returns a ``(2T+1, 2T+1)`` float mask where ``0`` is allowed and ``-inf`` is
    blocked (a :class:`torch.nn.TransformerEncoder` ``src_mask``).
    """
    T = num_timesteps
    seq_len = 2 * T + 1
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

    mask[0, 0] = 0.0

    # For each timestep k (1-indexed): s_k at 2k-1, a_k at 2k
    for k in range(1, T + 1):
        s_pos = 2 * k - 1
        a_pos = 2 * k

        # Both s_k and a_k attend to d
        mask[s_pos, 0] = 0.0
        mask[a_pos, 0] = 0.0

        # Within block k: s_k <-> a_k
        mask[s_pos, s_pos] = 0.0
        mask[s_pos, a_pos] = 0.0
        mask[a_pos, s_pos] = 0.0
        mask[a_pos, a_pos] = 0.0

        # Attend to all past timesteps 1..k-1
        for j in range(1, k):
            s_j = 2 * j - 1
            a_j = 2 * j
            mask[s_pos, s_j] = 0.0
            mask[s_pos, a_j] = 0.0
            mask[a_pos, s_j] = 0.0
            mask[a_pos, a_j] = 0.0

    return mask


class ACTransformerPredictor(nn.Module):
    """Action-conditioned Transformer predictor (~92.5M params).

    Three linear projections map the 4096-dim state, action, and demographics
    embeddings into a 1024-dim hidden space. The interleaved sequence is processed
    by a 6-layer pre-norm Transformer encoder with block-causal attention, and the
    state-position outputs are projected back to 4096 dims as next-step predictions.

    Args:
        prediction_mode: ``"residual"`` returns ``z_t + delta_t``;
            ``"absolute"`` returns ``delta_t`` directly.
    """

    def __init__(
        self,
        state_dim: int = 4096,
        action_dim: int = 4096,
        static_dim: int = 4096,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        max_timesteps: int = 72,
        ffn_dropout: float = 0.15,
        attn_dropout: float = 0.1,
        prediction_mode: str = "absolute",
        use_continuous_time: bool = False,
        absolute_time_additive: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps
        self.prediction_mode = prediction_mode
        self.use_continuous_time = use_continuous_time
        self.absolute_time_additive = absolute_time_additive

        if prediction_mode not in ("residual", "absolute"):
            raise ValueError(
                f"prediction_mode must be 'residual' or 'absolute', got {prediction_mode!r}"
            )

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        if use_continuous_time and not absolute_time_additive:
            self.time_proj = nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        if absolute_time_additive:
            # T(t) = W2 GELU(W1 [Fourier(t), log(1+t)]) → 4096, added to z.
            self.abs_time_proj = nn.Sequential(
                nn.Linear(9, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )

        max_seq_len = 2 * (max_timesteps + 2) + 1
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer.self_attn.dropout = attn_dropout
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, state_dim)

        # Cache masks per (T, device)
        self._cached_masks: dict[int, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        if self.prediction_mode == "residual":
            # Exact identity at initialization: z_hat_{t+1}=z_t. This is a
            # strong and clinically appropriate baseline for sticky 24-hour
            # ward states, and keeps early predictions on the encoder manifold.
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            nn.init.normal_(self.output_proj.weight, std=0.02)
            nn.init.zeros_(self.output_proj.bias)
        # Preserve exact compatibility with a discrete-time ICU checkpoint at
        # initialization; the continuous-time residual is learned during
        # hospital adaptation.
        if self.use_continuous_time and not self.absolute_time_additive:
            nn.init.zeros_(self.time_proj[-1].weight)
            nn.init.zeros_(self.time_proj[-1].bias)
        if self.absolute_time_additive:
            nn.init.zeros_(self.abs_time_proj[-1].weight)
            nn.init.zeros_(self.abs_time_proj[-1].bias)

    @staticmethod
    def _fourier_time(time_hours: torch.Tensor) -> torch.Tensor:
        """Absolute-time Fourier features: [log1p(days), sin, cos] → 9-d."""
        time_days = time_hours.float() / 24.0
        frequencies = torch.tensor(
            [1.0, 2.0, 4.0, 8.0],
            device=time_hours.device,
            dtype=torch.float32,
        )
        phase = time_days.unsqueeze(-1) * frequencies
        return torch.cat([
            torch.log1p(time_days).unsqueeze(-1),
            torch.sin(phase),
            torch.cos(phase),
        ], dim=-1)

    @staticmethod
    def _time_features(
        time_hours: torch.Tensor,
        duration_hours: torch.Tensor,
    ) -> torch.Tensor:
        """v0.1 Fourier features for absolute time and transition Δt."""
        time_days = time_hours.float() / 24.0
        duration_days = duration_hours.float() / 24.0
        frequencies = torch.tensor(
            [1.0, 2.0, 4.0, 8.0],
            device=time_hours.device,
            dtype=torch.float32,
        )
        phase = time_days.unsqueeze(-1) * frequencies
        return torch.cat([
            torch.log1p(time_days).unsqueeze(-1),
            torch.log1p(duration_days).unsqueeze(-1),
            torch.sin(phase),
            torch.cos(phase),
        ], dim=-1)

    def _get_block_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if T not in self._cached_masks or self._cached_masks[T].device != device:
            self._cached_masks[T] = build_block_causal_mask(T, device)
        return self._cached_masks[T]

    def _build_padding_mask(
        self,
        lengths: torch.Tensor,
        T_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a ``src_key_padding_mask`` for variable-length trajectories.

        ``True`` marks a padded position. Position 0 (``d``) is always valid;
        positions ``1..2L`` are valid for a trajectory of length ``L``.
        """
        B = lengths.shape[0]
        seq_len = 2 * T_max + 1
        mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        for b in range(B):
            L = int(lengths[b].item())
            mask[b, 0] = False
            mask[b, 1 : 2 * L + 1] = False

        return mask

    def forward(
        self,
        z_states: torch.Tensor,
        z_static: torch.Tensor,
        z_actions: torch.Tensor,
        lengths: torch.Tensor,
        noise_std: float = 0.0,
        return_last_only: bool = False,
        time_hours: torch.Tensor | None = None,
        duration_hours: torch.Tensor | None = None,
        action_time_hours: torch.Tensor | None = None,
        action_duration_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next-state embeddings.

        Args:
            z_states: ``(B, T, 4096)`` state embeddings (cast to fp32 internally).
            z_static: ``(B, 4096)`` demographics embedding.
            z_actions: ``(B, T, 4096)`` action embeddings.
            lengths: ``(B,)`` actual trajectory length per sample.
            noise_std: Gaussian-noise std for input augmentation, applied to
                ``z_states`` and ``z_actions`` during training only; ``0`` disables.
            return_last_only: If ``True``, return only the last state position's
                prediction, shape ``(B, 4096)``; otherwise return all
                teacher-forced positions, shape ``(B, T-1, 4096)``.
            action_time_hours: Optional action-token times. Defaults to
                ``time_hours`` for the legacy same-bin ``(s_t,a_t)`` layout.
                Target-interval training passes ``t_{t+1}`` here while keeping
                the state token at ``t_t``.
            action_duration_hours: Optional action-interval durations. Defaults
                to ``duration_hours``.
        """
        B, T_max, _ = z_states.shape
        device = z_states.device
        needs_time = self.use_continuous_time or self.absolute_time_additive
        if needs_time and time_hours is None:
            raise ValueError(
                "time_hours is required when continuous/absolute time is on; "
                "refusing to silently evaluate a timed predictor as "
                "a discrete-time model"
            )

        # Cast inputs to float32 for stable Transformer compute
        z_states_f = z_states.float()
        z_actions_f = z_actions.float()
        z_static_f = z_static.float()
        if action_time_hours is None:
            action_time_hours = time_hours
        if action_duration_hours is None:
            action_duration_hours = duration_hours

        # Optional embedding-noise augmentation (training only)
        if noise_std > 0.0 and self.training:
            z_states_f = z_states_f + torch.randn_like(z_states_f) * noise_std
            z_actions_f = z_actions_f + torch.randn_like(z_actions_f) * noise_std

        # v0.3: add T(t_abs) in encoder space so the predictor still emits
        # a pure patient latent (time is a condition, not part of z_hat).
        if time_hours is not None and self.absolute_time_additive:
            state_time_embedding = self.abs_time_proj(
                self._fourier_time(time_hours)
            )
            action_time_embedding = self.abs_time_proj(
                self._fourier_time(action_time_hours)
            )
            z_states_f = z_states_f + state_time_embedding
            z_actions_f = z_actions_f + action_time_embedding

        # 1. Project all three modalities to hidden_dim
        s = self.state_proj(z_states_f)
        a = self.action_proj(z_actions_f)
        d = self.static_proj(z_static_f)
        if (
            time_hours is not None
            and self.use_continuous_time
            and not self.absolute_time_additive
        ):
            if duration_hours is None:
                raise ValueError("duration_hours is required with time_hours")
            state_time_embedding = self.time_proj(
                self._time_features(time_hours, duration_hours)
            )
            action_time_embedding = self.time_proj(
                self._time_features(action_time_hours, action_duration_hours)
            )
            s = s + state_time_embedding
            a = a + action_time_embedding

        # 2. Interleave into [d, s_1, a_1, s_2, a_2, ..., s_T, a_T]
        seq_len = 2 * T_max + 1
        tokens = torch.zeros(B, seq_len, self.hidden_dim, device=device, dtype=s.dtype)
        tokens[:, 0, :] = d
        tokens[:, 1::2, :] = s
        tokens[:, 2::2, :] = a

        # 3. Add positional encoding
        pos_ids = torch.arange(seq_len, device=device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        # 4. Attention masks
        block_mask = self._get_block_causal_mask(T_max, device)
        padding_mask = self._build_padding_mask(lengths, T_max, device)

        # 5. Transformer forward
        out = self.transformer(
            tokens,
            mask=block_mask,
            src_key_padding_mask=padding_mask,
        )

        # 6. Extract state-position outputs (odd positions: 1, 3, 5, ...)
        state_out = out[:, 1::2, :]

        # 7. Output projection -> encoder space
        delta = self.output_proj(state_out)

        # 8. Form predictions
        if return_last_only:
            if self.prediction_mode == "residual":
                return z_states.float()[:, -1, :] + delta[:, -1, :]
            return delta[:, -1, :]

        # Teacher forcing: predict z_{t+1} for t=1..T-1.
        # Compare against raw patient latent, not time-augmented h.
        raw_states = z_states.float()
        if self.prediction_mode == "residual":
            preds = raw_states[:, :-1, :] + delta[:, :-1, :]
        else:
            preds = delta[:, :-1, :]

        return preds

    def forward_rollout_step(
        self,
        z_history: torch.Tensor,
        z_static: torch.Tensor,
        za_history: torch.Tensor,
        za_next: torch.Tensor,
        z_last: torch.Tensor,
        time_hours: torch.Tensor | None = None,
        duration_hours: torch.Tensor | None = None,
        action_time_hours: torch.Tensor | None = None,
        action_duration_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a single autoregressive rollout step.

        Args:
            z_history: ``(B, H, 4096)`` state embeddings for hours ``1..H``.
            z_static: ``(B, 4096)`` demographics embedding.
            za_history: ``(B, H, 4096)`` action embeddings for hours ``1..H``.
            za_next: ``(B, 4096)`` action embedding at the current step.
            z_last: ``(B, 4096)`` current state embedding (real or predicted).
            time_hours: ``(B, H+1)`` absolute hours for the concatenated sequence.
            duration_hours: optional ``(B, H+1)`` durations (v0.1 path only).
            action_time_hours: optional action-token times for target-interval
                alignment.
            action_duration_hours: optional action-token interval durations.

        Returns:
            ``(B, 4096)`` predicted next state.
        """
        z_seq = torch.cat([z_history, z_last.unsqueeze(1)], dim=1)
        za_seq = torch.cat([za_history, za_next.unsqueeze(1)], dim=1)
        lengths = torch.full(
            (z_seq.shape[0],), z_seq.shape[1],
            device=z_seq.device, dtype=torch.long,
        )

        return self.forward(
            z_states=z_seq,
            z_static=z_static,
            z_actions=za_seq,
            lengths=lengths,
            return_last_only=True,
            time_hours=time_hours,
            duration_hours=duration_hours,
            action_time_hours=action_time_hours,
            action_duration_hours=action_duration_hours,
        )

    def num_params(self) -> int:
        """Total parameter count (~92.5M at default hyperparameters)."""
        return sum(p.numel() for p in self.parameters())
