# src/tabtransformer_library/model_library.py

"""
Builds a TabTransformer model using pytorch-tabular 1.1.1
This version removes all unsupported TrainerConfig arguments and ALL callbacks.
"""

from pytorch_tabular import TabularModel
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    TrainerConfig,
    OptimizerConfig,
)

import torch


def build_tabtransformer_library(numeric_cols, categorical_cols, target_col):
    """
    Builds a clean, compatible TabTransformer model.
    """

    # ----------------------------
    # Data Configuration
    # ----------------------------
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    # ----------------------------
    # Model Configuration
    # ----------------------------
    model_config = TabTransformerConfig(
        task="classification",
        metrics=["accuracy"],  # pytorch-tabular 1.1.1 expects metric names
        embedding_dim=32,
        num_heads=4,
        dropout=0.1,
    )

    # ----------------------------
    # Optimizer Configuration
    # ----------------------------
    optimizer_config = OptimizerConfig(
        optimizer="Adam",
        lr=1e-3,
    )

    # ----------------------------
    # Trainer Configuration
    # (IMPORTANT: No callbacks, no progress bar args)
    # ----------------------------
    trainer_config = TrainerConfig(
        batch_size=512,
        max_epochs=3,
        gpus=0,         # ensures CPU-only unless CUDA is available
        auto_lr_find=False,
    )

    # ----------------------------
    # Instantiate TabularModel
    # ----------------------------
    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    return model