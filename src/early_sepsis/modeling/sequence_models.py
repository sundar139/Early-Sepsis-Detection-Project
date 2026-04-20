from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

ModelKind = Literal["gru", "lstm", "patchtst"]


@dataclass(slots=True)
class SequenceModelConfig:
    """Model hyperparameters for sequence classifiers."""

    model_type: ModelKind = "patchtst"
    include_mask: bool = True
    include_static: bool = True

    recurrent_hidden_dim: int = 128
    recurrent_num_layers: int = 2
    recurrent_dropout: float = 0.2
    recurrent_bidirectional: bool = True

    patch_len: int = 4
    patch_stride: int = 2
    patch_d_model: int = 128
    patch_num_heads: int = 4
    patch_num_layers: int = 3
    patch_ff_dim: int = 256
    patch_dropout: float = 0.2


class RecurrentSequenceClassifier(nn.Module):
    """GRU/LSTM baseline sequence classifier for ICU trajectories."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        sequence_length: int,
        config: SequenceModelConfig,
    ) -> None:
        super().__init__()
        del sequence_length

        self.include_mask = config.include_mask
        self.include_static = config.include_static

        effective_input_dim = input_dim * (2 if self.include_mask else 1)
        if effective_input_dim <= 0:
            msg = "Effective input dimension must be positive."
            raise ValueError(msg)

        rnn_class = nn.GRU if config.model_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=effective_input_dim,
            hidden_size=config.recurrent_hidden_dim,
            num_layers=config.recurrent_num_layers,
            dropout=config.recurrent_dropout if config.recurrent_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.recurrent_bidirectional,
        )

        output_dim = config.recurrent_hidden_dim * (2 if config.recurrent_bidirectional else 1)
        self.static_projection: nn.Module | None = None
        if self.include_static and static_dim > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(static_dim, output_dim),
                nn.GELU(),
            )
            output_dim *= 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(config.recurrent_dropout),
            nn.Linear(output_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        missing_mask: Tensor | None = None,
        static_features: Tensor | None = None,
    ) -> Tensor:
        if self.include_mask:
            if missing_mask is None:
                msg = "missing_mask is required when include_mask=True"
                raise ValueError(msg)
            sequence_input = torch.cat([features, missing_mask], dim=-1)
        else:
            sequence_input = features

        outputs, hidden = self.rnn(sequence_input)
        del outputs

        if isinstance(hidden, tuple):
            hidden_state = hidden[0]
        else:
            hidden_state = hidden

        if self.rnn.bidirectional:
            final_hidden = torch.cat([hidden_state[-2], hidden_state[-1]], dim=-1)
        else:
            final_hidden = hidden_state[-1]

        if self.include_static and self.static_projection is not None:
            if static_features is None:
                msg = "static_features are required when include_static=True"
                raise ValueError(msg)
            static_embedding = self.static_projection(static_features)
            final_hidden = torch.cat([final_hidden, static_embedding], dim=-1)

        logits = self.classifier(final_hidden).squeeze(-1)
        return logits


class PatchTSTClassifier(nn.Module):
    """PatchTST-style transformer classifier with temporal patch embedding."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        sequence_length: int,
        config: SequenceModelConfig,
    ) -> None:
        super().__init__()

        self.include_mask = config.include_mask
        self.include_static = config.include_static
        self.patch_len = config.patch_len
        self.patch_stride = config.patch_stride

        if sequence_length < self.patch_len:
            msg = "sequence_length must be greater than or equal to patch_len"
            raise ValueError(msg)

        effective_input_dim = input_dim * (2 if self.include_mask else 1)
        patch_vector_dim = self.patch_len * effective_input_dim

        self.num_patches = 1 + (sequence_length - self.patch_len) // self.patch_stride
        if self.num_patches <= 0:
            msg = "Invalid patch configuration produced zero patches."
            raise ValueError(msg)

        self.patch_embedding = nn.Linear(patch_vector_dim, config.patch_d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, config.patch_d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.patch_d_model,
            nhead=config.patch_num_heads,
            dim_feedforward=config.patch_ff_dim,
            dropout=config.patch_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.patch_num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.patch_num_layers,
            )

        output_dim = config.patch_d_model
        self.static_projection: nn.Module | None = None
        if self.include_static and static_dim > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(static_dim, config.patch_d_model),
                nn.GELU(),
            )
            output_dim += config.patch_d_model

        self.head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(config.patch_dropout),
            nn.Linear(output_dim, 1),
        )

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        features: Tensor,
        missing_mask: Tensor | None = None,
        static_features: Tensor | None = None,
    ) -> Tensor:
        if self.include_mask:
            if missing_mask is None:
                msg = "missing_mask is required when include_mask=True"
                raise ValueError(msg)
            sequence_input = torch.cat([features, missing_mask], dim=-1)
        else:
            sequence_input = features

        patches = sequence_input.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        batch_size, patch_count, feature_dim, patch_len = patches.shape

        if patch_count != self.num_patches:
            msg = (
                f"Unexpected patch count {patch_count}; expected {self.num_patches}. "
                "Ensure inference window length matches training."
            )
            raise ValueError(msg)

        if patch_len != self.patch_len:
            msg = f"Unexpected patch length {patch_len}; expected {self.patch_len}."
            raise ValueError(msg)

        patch_tokens = patches.permute(0, 1, 3, 2).contiguous().view(
            batch_size,
            patch_count,
            self.patch_len * feature_dim,
        )
        token_embeddings = self.patch_embedding(patch_tokens)
        token_embeddings = token_embeddings + self.position_embedding[:, :patch_count, :]

        encoded = self.encoder(token_embeddings)
        pooled = encoded.mean(dim=1)

        if self.include_static and self.static_projection is not None:
            if static_features is None:
                msg = "static_features are required when include_static=True"
                raise ValueError(msg)
            static_embedding = self.static_projection(static_features)
            pooled = torch.cat([pooled, static_embedding], dim=-1)

        logits = self.head(pooled).squeeze(-1)
        return logits


def build_sequence_model(
    input_dim: int,
    static_dim: int,
    sequence_length: int,
    config: SequenceModelConfig,
) -> nn.Module:
    """Builds a configured sequence classifier."""

    if config.model_type in {"gru", "lstm"}:
        return RecurrentSequenceClassifier(
            input_dim=input_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            config=config,
        )
    if config.model_type == "patchtst":
        return PatchTSTClassifier(
            input_dim=input_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            config=config,
        )

    msg = f"Unsupported model_type: {config.model_type}"
    raise ValueError(msg)
