from __future__ import annotations

import pytest


def test_gru_forward_pass_shapes() -> None:
    torch = pytest.importorskip("torch")

    from early_sepsis.modeling.sequence_models import SequenceModelConfig, build_sequence_model

    config = SequenceModelConfig(
        model_type="gru",
        include_mask=True,
        include_static=True,
        recurrent_hidden_dim=16,
        recurrent_num_layers=1,
        recurrent_dropout=0.1,
        recurrent_bidirectional=False,
    )
    model = build_sequence_model(input_dim=5, static_dim=3, sequence_length=8, config=config)

    features = torch.randn(4, 8, 5)
    masks = torch.zeros(4, 8, 5)
    static = torch.randn(4, 3)

    logits = model(features=features, missing_mask=masks, static_features=static)
    assert tuple(logits.shape) == (4,)
    assert torch.isfinite(logits).all()


def test_patchtst_forward_pass_shapes() -> None:
    torch = pytest.importorskip("torch")

    from early_sepsis.modeling.sequence_models import SequenceModelConfig, build_sequence_model

    config = SequenceModelConfig(
        model_type="patchtst",
        include_mask=True,
        include_static=True,
        patch_len=4,
        patch_stride=2,
        patch_d_model=32,
        patch_num_heads=4,
        patch_num_layers=2,
        patch_ff_dim=64,
        patch_dropout=0.1,
    )
    model = build_sequence_model(input_dim=6, static_dim=2, sequence_length=8, config=config)

    features = torch.randn(3, 8, 6)
    masks = torch.zeros(3, 8, 6)
    static = torch.randn(3, 2)

    logits = model(features=features, missing_mask=masks, static_features=static)
    assert tuple(logits.shape) == (3,)
    assert torch.isfinite(logits).all()
