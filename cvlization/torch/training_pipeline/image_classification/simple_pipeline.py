from dataclasses import dataclass
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .lightning import ImageClassifier, ImageClassifierCallback
from .image_classification_config import ImageClassificationConfig
from .transform import TransformedDataset


class SimpleImageClassificationPipeline:
    """An example image classification training pipeline with minimal dependencies,
    while having enough capacity to train high quality models and being reproducible.
    """

    @dataclass
    class Config(ImageClassificationConfig):
        ...

    def __init__(self, config: Config):
        self._config = config

    def fit(self, dataset_builder):
        pl_model = self._create_model()
        experiment_tracker = self._setup_experiment_tracker()
        trainer = self._create_trainer(experiment_tracker)
        train_dataloader = self._create_training_dataloader(dataset_builder)
        val_dataloader = self._create_validation_dataloader(dataset_builder)
        trainer.fit(pl_model, train_dataloader, val_dataloader)

    def _create_model(self):
        model_constructor = getattr(
            torchvision.models, self._config.model_name)
        model = model_constructor(pretrained=self._config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, self._config.num_classes)
        pl_model = ImageClassifier(
            model=model, num_classes=self._config.num_classes, lr=self._config.lr)
        return pl_model

    def _create_trainer(self, experiment_tracker):
        trainer = Trainer(
            default_root_dir=self._config.lightning_root_dir,
            deterministic=True,
            accelerator=self._config.accelerator,
            gpus=self._config.gpus,
            limit_train_batches=self._config.train_steps_per_epoch or 1.0,
            limit_val_batches=self._config.val_steps_per_epoch or 1.0,
            max_epochs=self._config.epochs,
            logger=experiment_tracker,
            callbacks=[ImageClassifierCallback()],
        )
        return trainer
    
    def _setup_experiment_tracker(self):
        if self._config.tracking_uri is None:
            tracking_uri = None
        else:
            tracking_uri = f"file:{self._config.tracking_uri}"
        self._experiment_tracker = experiment_tracker = MLFlowLogger(
            experiment_name=self._config.experiment_name,
            run_name=self._config.run_name,
            tracking_uri=tracking_uri,
        )
        
        for k, v in self._config.__dict__.items():
            experiment_tracker.experiment.log_param(run_id=experiment_tracker.run_id, key=k, value=v)
        return self._experiment_tracker

    def _transform_training_dataset(self, dataset):
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        )
        return TransformedDataset(dataset, transform_train)

    def _transform_validation_dataset(self, dataset):
        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return TransformedDataset(dataset, transform_val)

    def _create_training_dataloader(self, dataset_builder):
        train_ds = self._transform_training_dataset(
            dataset_builder.training_dataset()
        )
        return DataLoader(
            train_ds,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers
        )

    def _create_validation_dataloader(self, dataset_builder):
        val_ds = self._transform_validation_dataset(
            dataset_builder.validation_dataset()
        )
        return DataLoader(
            val_ds,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers
        )
