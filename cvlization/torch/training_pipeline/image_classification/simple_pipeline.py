from dataclasses import dataclass
from multiprocessing import cpu_count
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from .lightning import ImageClassifier, ImageClassifierCallback
from .image_classification_config import ImageClassificationConfig


class SimpleImageClassificationPipeline:
    """An example image classification training pipeline with minimal dependencies,
    while having enough capacity to train high quality models and being reproducible.
    """

    # TODO: include image augmentations.

    @dataclass
    class Config(ImageClassificationConfig):
        ...

    def __init__(self, config: Config):
        self._config = config

    def fit(self, dataset_builder):
        pl_model = self._create_model()
        trainer = self._create_trainer()
        train_dataloader = self.create_training_dataloader(dataset_builder)
        val_dataloader = self.create_validation_dataloader(dataset_builder)
        trainer.fit(pl_model, train_dataloader, val_dataloader)

    def _create_model(self):
        model_constructor = getattr(
            torchvision.models, self._config.model_name)
        model = model_constructor(pretrained=self._config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, self._config.num_classes)
        pl_model = ImageClassifier(
            model=model, num_classes=self._config.num_classes, lr=self._config.lr)
        return pl_model

    def _create_trainer(self):
        trainer = Trainer(
            deterministic=True,
            limit_train_batches=self._config.train_steps_per_epoch or 1.0,
            limit_val_batches=self._config.val_steps_per_epoch or 1.0,
            max_epochs=self._config.epochs,
            logger=MLFlowLogger(
                experiment_name=self._config.experiment_name,
                run_name=self._config.run_name,
                tracking_uri=self._config.tracking_uri
            ),
            callbacks=[ImageClassifierCallback()]
        )
        return trainer

    def create_training_dataloader(self, dataset_builder):
        return DataLoader(
            dataset_builder.training_dataset(),
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers
        )

    def create_validation_dataloader(self, dataset_builder):
        return DataLoader(
            dataset_builder.validation_dataset(),
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers
        )
