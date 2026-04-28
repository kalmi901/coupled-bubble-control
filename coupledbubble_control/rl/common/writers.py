from __future__ import annotations
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any
import os


class TFWriter(SummaryWriter):
    def __init__(self, log_dir: str, project_name: str, trial_name: str, model: Any):
        run_folder = os.path.join(log_dir, trial_name)
        super().__init__(run_folder)

        if hasattr(model, "metadata"):
            self.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join(str(f"|{param}|{getattr(model, param)}|") for param in model.metadata["hyperparameters"] if hasattr(model, param)))
                )
