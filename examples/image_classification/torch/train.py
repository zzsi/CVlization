import logging

from cvlization.specs.ml_framework import MLFramework
from cvlization.specs import ModelSpec
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.specs.prediction_tasks import ImageClassification
from cvlization.training_pipeline import TrainingPipeline
from cvlization.lab.experiment import Experiment
from cvlization.torch.encoder.torch_image_backbone import image_backbone_names


LOGGER = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        prediction_task = ImageClassification(
            n_classes=10,
            num_channels=3,
            image_height=32,
            image_width=32,
            channels_first=True,
        )

        training_pipeline = TrainingPipeline(
            ml_framework=MLFramework.PYTORCH,
            model=ModelSpec(
                image_backbone=self.args.net,
                model_inputs=prediction_task.get_model_inputs(),
                model_targets=prediction_task.get_model_targets(),
            ),
            loss_function_included_in_model=False,
            collate_method=None,
            epochs=50,
            train_batch_size=128,
            val_batch_size=128,
            train_steps_per_epoch=500,
            val_steps_per_epoch=None,
            optimizer_name="Adam",
            lr=0.0001,
            n_gradients=1,
            experiment_tracker=None,
        )

        Experiment(
            # The interface (inputs and outputs) of the model.
            prediction_task=ImageClassification(
                n_classes=10,
                num_channels=3,
                image_height=32,
                image_width=32,
                channels_first=True,
            ),
            # Dataset and transforms.
            dataset_builder=TorchvisionDatasetBuilder(dataset_classname="CIFAR10"),
            # Training pipeline: model, trainer, optimizer.
            training_pipeline=training_pipeline,
        ).run()


if __name__ == "__main__":
    """
    python -m examples.image_classification.torchvision.train
    """

    from argparse import ArgumentParser

    options = image_backbone_names()
    parser = ArgumentParser(
        epilog=f"""
                options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--net", default="resnet18")
    args = parser.parse_args()
    TrainingSession(args).run()
