from dataclasses import dataclass
from multiprocessing import cpu_count


@dataclass
class ImageClassificationConfig:
    # model
    num_classes: int
    model_name: str = "resnet18"
    pretrained: bool = True

    # training loop
    epochs: int = 10
    train_steps_per_epoch: int = None
    val_steps_per_epoch: int = None

    # optimizer
    lr: float = 0.0001

    # data loading
    batch_size: int = 32
    num_workers: int = cpu_count()

    # device
    accelerator: str = "gpu"  # None
    gpus: int = 1  # None

    # experiment tracking
    experiment_name: str = "image_classification"
    run_name: str = None
    tracking_uri: str = "./mlruns"
    lightning_root_dir: str = "./lightning_logs"
