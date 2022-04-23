from dataclasses import dataclass
from unittest.mock import patch
from cvlization.lab.experiment import Experiment, SplittedDataset
from cvlization.lab.model_specs import ImageClassification
from cvlization.training_pipeline import (
    TrainingPipeline,
    TrainingPipelineConfig,
    MLFramework,
)


@dataclass
class MockDataset(SplittedDataset):
    field1: str = ""

    def training_dataset(self, **kwargs):
        return []

    def validation_dataset(self, **kwargs):
        return []

    def transform_training_dataset_tf(self, dataset, **kwargs):
        return dataset

    def transform_validation_dataset_tf(self, dataset, **kwargs):
        return dataset


class MockExperimentTracker:
    def setup(self):
        return self

    def log_params(self, params):
        assert isinstance(params, dict)
        self._params = params


def test_experiment_can_get_config_dict():
    prediction_task = ImageClassification()
    training_pipeline_config = TrainingPipelineConfig()
    e = Experiment(
        prediction_task=prediction_task,
        dataset=MockDataset(),
        training_pipeline=TrainingPipeline(
            config=training_pipeline_config, framework=MLFramework.TENSORFLOW
        ),
    )
    d = e.get_config_dict()
    assert d == {
        "field1": "",
        **prediction_task.__dict__,
        **training_pipeline_config.__dict__,
        "framework": MLFramework.TENSORFLOW.value,
    }


def test_experiment_can_run():
    # TODO: run an mnist exp.
    prediction_task = ImageClassification()
    training_pipeline_config = TrainingPipelineConfig(image_backbone="simple")
    training_pipeline = TrainingPipeline(
        config=training_pipeline_config, framework=MLFramework.TENSORFLOW
    )
    e = Experiment(
        prediction_task=prediction_task,
        dataset=MockDataset(),
        training_pipeline=training_pipeline,
    )
    with patch.object(training_pipeline, "run") as mock_run_func:
        e.run()
    mock_run_func.assert_called_once()


def test_experiment_can_log_params():
    tracker = MockExperimentTracker()
    prediction_task = ImageClassification()
    training_pipeline_config = TrainingPipelineConfig()
    training_pipeline = TrainingPipeline(
        config=training_pipeline_config, framework=MLFramework.TENSORFLOW
    )
    e = Experiment(
        prediction_task=prediction_task,
        dataset=MockDataset(),
        training_pipeline=training_pipeline,
        experiment_tracker=tracker,
    )
    with patch.object(training_pipeline, "create_model") as mock_fn:
        mock_fn.return_value = training_pipeline
        with patch.object(training_pipeline, "prepare_datasets") as mock_feed_data:
            mock_feed_data.return_value = training_pipeline
            with patch.object(
                training_pipeline, "create_trainer"
            ) as mock_create_trainer:
                mock_create_trainer.return_value = training_pipeline
                with patch.object(training_pipeline, "run") as mock_run:
                    e.run()
    assert tracker._params == e.get_config_dict()
