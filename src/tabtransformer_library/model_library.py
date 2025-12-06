import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

def build_tabtransformer_library(numeric_cols, categorical_cols, target_col="isFraud"):
    """
    Builds and returns a TabTransformer model using pytorch-tabular.
    """

    data_config = DataConfig(
        target=[target_col],
        continuous_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    model_config = TabTransformerConfig(
        task="classification"
    )

    trainer_config = TrainerConfig(
        batch_size=1024,
        max_epochs=5,
        accelerator="cpu"
    )

    optimizer_config = OptimizerConfig(
        optimizer="Adam"
    )

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    return model