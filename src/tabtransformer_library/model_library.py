"""
TabTransformer - Library (pytorch_tabular) version

This file builds a TabularModel using the TabTransformer backbone from
pytorch_tabular. I’m keeping things fairly simple and explicit so that
it’s easy to read and to tweak later in the project.
"""

import os

# This disables the Rich progress bar that Lightning uses by default.
# On some setups (like Mac with MPS) Rich can occasionally crash with
# "IndexError: pop from empty list", so I just turn it off here.
os.environ["PL_DISABLE_RICH_PROGRESS_BAR"] = "1"

from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig


def build_tabtransformer_library(
    numeric_cols,
    categorical_cols,
    target_col: str = "isFraud",
):
    """
    Build and return a pytorch_tabular TabularModel using the TabTransformer.

    numeric_cols: list of numeric feature names
    categorical_cols: list of categorical feature names
    target_col: name of the label column
    """

    # 1) Data config: tells pytorch_tabular what each column type is
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=numeric_cols,
        categorical_cols=categorical_cols,
        # IMPORTANT: use only transforms that this version of pytorch_tabular
        # + torchmetrics + lightning can agree on.
        # "quantile_normal" is supported and avoids the "standardize" error.
        continuous_feature_transform="quantile_normal",
    )

    # 2) Model config: TabTransformer hyperparameters
    # IMPORTANT: we ONLY use "accuracy" here. Using "f1", "auroc",
    # or "average_precision" makes pytorch_tabular call older torchmetrics
    # APIs like `torchmetrics.functional.f1` which do not exist anymore
    # in your installed torchmetrics version.
    model_config = TabTransformerConfig(
        task="classification",
        metrics=["accuracy"],  # keep this minimal to avoid AttributeError in torchmetrics
    )

    # 3) Optimizer / training hyperparameters
    # NOTE: OptimizerConfig in pytorch_tabular expects optimizer_params,
    # not a bare "lr" argument. We keep this small & standard.
    optim_config = OptimizerConfig(
        optimizer="Adam",
    )

    # IMPORTANT: TrainerConfig must only use arguments that exist in the
    # version of pytorch_lightning bundled with pytorch_tabular.
    # I’m keeping this minimal to avoid keyword errors.
    trainer_config = TrainerConfig(
        max_epochs=5,        # keep it small so runs finish
        deterministic=True,  # for reproducibility
        progress_bar="simple",  # use basic tqdm bar instead of Rich to avoid IndexError
    )

    # 4) Put it all together
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optim_config,
        trainer_config=trainer_config,
        verbose=True,
    )

    return tabular_model