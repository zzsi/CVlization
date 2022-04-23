from dataclasses import dataclass
from typing import Union, Optional, Any, Iterable, List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import os
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)

from ..data.ml_dataset import MLDataset, ModelInput, ModelTarget
from .torch_model import TorchModel
from .torch_dataset import MapDataset, GeneratorDataset
from .torch_model_factory import TorchModelFactory
from ..base_trainer import BaseTrainer


LOGGER = logging.getLogger(__name__)


@dataclass
class TorchTrainer(BaseTrainer):

    # ## Model (a trainable)
    model: Union[TorchModel, nn.Module]
    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]

    # ## Datasets
    train_dataset: Union[MLDataset, Iterable]
    val_dataset: Union[MLDataset, Iterable] = None
    num_workers: int = 0

    # ## Precision
    precision: Optional[str] = "fp32"

    # ## Checkpoint directory
    log_dir: str = os.getcwd()

    # ## Device
    device: Optional[torch.device] = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ## Training loop
    epochs: Optional[int] = 10
    # initial_epoch set to 1, so that the pre-training eval step can log metrics correctly.
    initial_epoch: Optional[int] = 1
    train_steps_per_epoch: Optional[int] = None
    val_steps_per_epoch: Optional[int] = None

    # TODO: lr_finder not implemented.
    use_lr_finder: Optional[bool] = False
    n_gradients: int = (
        1  # Number of steps for gradient accumulation before updating the weights.
    )
    early_stopping_patience: int = 20

    # ## Info
    name: Optional[str] = "torch_trainer"

    # ## Tracking
    experiment_tracker: Optional[Any] = None
    experiment_name: Optional[
        str
    ] = "cvlab"  # this is the project name in wandb, experiment name in mlflow
    run_name: Optional[str] = None
    checkpoint_root_dir: Optional[str] = None

    def __post_init__(self):
        self._validate_fields()
        super().__init__(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            experiment_tracker=self.experiment_tracker,
        )
        limit_train_batches = self.train_steps_per_epoch or 1.0
        limit_val_batches = self.val_steps_per_epoch or 1.0

        LOGGER.info(f"limit train batches: {limit_train_batches}")
        LOGGER.info(f"limit val batches: {limit_val_batches}")

        if not self.log_dir:
            weights_save_path = None
            enable_checkpointing = False
            LOGGER.info("Disable checkpointing in pytorch-lightning.")
        else:
            weights_save_path = self.log_dir
            enable_checkpointing = True

        logger_for_experiment_tracking = self._set_up_experiment_tracker()
        callbacks = [
            EarlyStopping(
                patience=self.early_stopping_patience,
                monitor="val_loss",
                mode="min",
            ),
        ]
        if self.experiment_tracker is not None:
            callbacks.append(
                LearningRateMonitor(
                    logging_interval="epoch",
                    log_momentum=True,
                )
            )
        self._lightning_trainer = Trainer(
            # TODO: RuntimeError: No GPUs available.
            # auto_select_gpus=True,
            gpus=1 if torch.cuda.is_available() else 0,
            auto_select_gpus=False,
            max_epochs=self.epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            default_root_dir=self.checkpoint_root_dir,
            weights_save_path=weights_save_path,
            checkpoint_callback=enable_checkpointing,
            enable_checkpointing=enable_checkpointing,
            logger=logger_for_experiment_tracking,
            # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/6170
            #    lr scheduler may interfere with grad accumulation.
            accumulate_grad_batches=self.n_gradients,
            precision=16 if self.precision == "fp16" else 32,
            callbacks=callbacks,
            enable_progress_bar=True,
        )
        self._loss_function = self._get_or_create_loss_function(self.model).to(
            self.device
        )

    def log_params(self, params: dict):
        if self.experiment_tracker == "wandb":
            self._logger_for_experiment_tracking.experiment.config.update(params)
        elif self.experiment_tracker is not None:
            LOGGER.info(
                f"Logging params for {self.experiment_tracker} not implemented yet."
            )

    def _set_up_experiment_tracker(self):
        if self.experiment_tracker == "wandb":
            from pytorch_lightning.loggers import WandbLogger

            # TODO: for wandb, customize the summary method.
            # define a metric we are interested in the maximum of
            # e.g. wandb.define_metric("acc", summary="max")
            #
            # MLFlow, on the other hand, does not support this yet:
            #   https://github.com/mlflow/mlflow/issues/4750

            self._logger_for_experiment_tracking = WandbLogger(
                project=self.experiment_name, experiment=self.run_name, save_dir="wandb"
            )
        elif self.experiment_tracker == "mlflow":
            raise NotImplementedError("mlflow logger not implemented yet.")
        elif self.experiment_tracker is None:
            self._logger_for_experiment_tracking = None
            LOGGER.info("No experiment tracking.")
        else:
            raise ValueError(f"Unknown experiment tracker: {self.experiment_tracker}")
        return self._logger_for_experiment_tracking

    def _watch_model(self):
        if self.experiment_tracker == "wandb":
            if isinstance(self.model, nn.Module) or isinstance(
                self.model, LightningModule
            ):
                self._logger_for_experiment_tracking.watch(self.model)
            else:
                raise ValueError(f"Cannot watch model: {self.model}")
        elif self.experiment_tracker is not None:
            LOGGER.info(
                f"Watching model for {self.experiment_tracker} not implemented yet."
            )

    def _training_loop(self):
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        num_workers = self.num_workers

        class DataModule(LightningDataModule):
            def train_dataloader(self):
                if isinstance(train_dataset, MLDataset):
                    return DataLoader(
                        MapDataset(train_dataset),
                        batch_size=train_dataset.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                    )
                else:
                    # TODO: if the dataset is already a DataLoader, no need to wrap it.
                    return DataLoader(
                        GeneratorDataset(train_dataset), num_workers=num_workers
                    )

            def val_dataloader(self):
                if isinstance(val_dataset, MLDataset):
                    return DataLoader(
                        MapDataset(val_dataset),
                        batch_size=val_dataset.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                    )
                else:
                    v = GeneratorDataset(val_dataset)
                    # batch = next(v)
                    return DataLoader(v, num_workers=num_workers)

            def test_dataloader(self):
                return None

        dm = DataModule()
        if isinstance(train_dataset, MLDataset):
            LOGGER.info("loading some data ...")
            d = DataLoader(
                MapDataset(val_dataset),
                batch_size=val_dataset.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            d = [MapDataset(val_dataset)[1]]
            for b in d:
                print(len(b), type(b))
                break
        # When using LazyModules, call `forward` with a dummy batch to initialize
        # the parameters before calling torch functions.
        one_batch = next(iter(dm.train_dataloader()))
        self.model.forward(one_batch[0])
        self._watch_model()

        self._lightning_trainer.fit(self.model, dm)

    def get_metrics(self) -> dict:
        return self._lightning_trainer.callback_metrics

    def _get_optimizer_class(self):
        if isinstance(self.optimizer_name, str) and hasattr(optim, self.optimizer_name):
            return getattr(optim, self.optimizer_name)
        LOGGER.warning(f"Cannot find optimizer {self.optimizer_name}. Using Adam.")
        return optim.Adam

    def _get_current_epoch(self, checkpoint_dir: str) -> int:
        raise NotImplementedError("Read epoch info from checkpoint_dir")

    def _get_or_create_loss_function(self, model: TorchModel):
        if hasattr(model, "loss_function"):
            return model.loss_function
        else:
            return TorchModelFactory(
                model_inputs=self.model_inputs,
                model_targets=self.model_targets,
            ).create_loss_function()

    def _validate_fields(self):
        if self.model is None:
            raise ValueError("Please set the model field to a TorchModel.")
        if not self.n_gradients >= 1:
            raise ValueError("n_gradients must be >= 1")
        self.n_gradients = int(self.n_gradients)


class DavidTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 10,
        train_batch_size: int = 512,
        use_cached_cifar10: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.epochs = epochs
        self.batch_size = train_batch_size
        self.use_cached_cifar10 = use_cached_cifar10

    def _training_loop(self):
        from functools import partial
        import numpy as np
        from .net.davidnet.core import (
            PiecewiseLinear,
            Const,
            union,
            Timer,
            preprocess,
            pad,
            normalise,
            transpose,
            Transform,
            Crop,
            Cutout,
            FlipLR,
        )
        from .net.davidnet.torch_backend import (
            SGD,
            MODEL,
            LOSS,
            OPTS,
            Table,
            train_epoch,
            x_ent_loss,
            # trainable_params,
            cifar10,
            cifar10_mean,
            cifar10_std,
            DataLoader as DataLoaderDavid,
        )

        batch_size = self.batch_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class WrappedDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader

            def __iter__(self):
                for x, y in self.dataloader:
                    yield {"input": x.to(device).half(), "target": y.to(device).long()}

            def __len__(self):
                return len(self.dataloader)

        if self.use_cached_cifar10:
            dataset = cifar10("./data")
            transforms = [
                partial(
                    normalise,
                    mean=np.array(cifar10_mean, dtype=np.float32),
                    std=np.array(cifar10_std, dtype=np.float32),
                ),
                partial(transpose, source="NHWC", target="NCHW"),
            ]
            train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
            train_set = list(
                zip(
                    *preprocess(
                        dataset["train"], [partial(pad, border=4)] + transforms
                    ).values()
                )
            )
            test_set = list(zip(*preprocess(dataset["valid"], transforms).values()))
            train_batches = DataLoaderDavid(
                Transform(train_set, train_transforms),
                batch_size,
                shuffle=True,
                set_random_choices=True,
                drop_last=True,
            )
            val_batches = DataLoaderDavid(
                test_set, batch_size, shuffle=False, drop_last=False
            )
        else:
            train_batches = WrappedDataLoader(self.train_dataset)
            val_batches = WrappedDataLoader(self.val_dataset)

        model = self.model

        loss = x_ent_loss
        epochs = self.epochs
        lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
        n_train_batches = int(50000 / batch_size)
        timer = Timer(synch=torch.cuda.synchronize)
        opts = [
            SGD(
                # trainable_params(model).values(),
                model.parameters(),
                {
                    "lr": (
                        lambda step: lr_schedule(step / n_train_batches) / batch_size
                    ),
                    "weight_decay": Const(5e-4 * batch_size),
                    "momentum": Const(0.9),
                },
            )
        ]
        logs, state = Table(), {MODEL: model, LOSS: loss, OPTS: opts}
        for epoch in range(epochs):
            logs.append(
                union(
                    {"epoch": epoch + 1},
                    train_epoch(state, timer, train_batches, val_batches),
                )
            )
