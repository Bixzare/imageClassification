from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Sequence
import torch
import os
from dataclasses import dataclass, field
import wandb

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = "/teamspace/studios/this_studio/mosquito-classification"
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT  + "/package/data/MosquitoCNNComparison/DepictionDataset/body"
# Sous-répertoires pour train, val, et test
TRAIN_DIR = DATA_DIR + "/train"
VAL_DIR = DATA_DIR + "/val"
TEST_DIR = DATA_DIR + "/test"
#INTERIM_DATA_DIR = DATA_DIR / "interim"
#PROCESSED_DATA_DIR = DATA_DIR / "processed"
#EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT + "models"

REPORTS_DIR = PROJ_ROOT + "reports"
FIGURES_DIR = REPORTS_DIR + "figures"

@dataclass
class Args:
    """This class saves the arguments used in the experiments. 
    The arguments can be set here or through command line. when using the command line every underscore ('_') become a dash ('-').
    """

    # -- wandb
    wandb_entity: str  # Placez ceci en premier
    project_name: str = 'mosquito'
    run_name: str = 'debug'
  
    tag: Sequence[str] = ('V0')

    # -- random seed
    seed: int = 41

    # data location
    data_dir: Path = DATA_DIR
    train_dir: Path = TRAIN_DIR
    test_dir: Path = TEST_DIR
    val_dir: Path = VAL_DIR
    use_all_train_data: bool = True

    # -- data augmentation
    debug_mode: bool = False
    pin_memory: bool = True  # dataloader
    num_workers: int = os.cpu_count() // 2
    rotate_degree: float = 180.0
    resizing_mode: str = 'nearest'  # Another option is "None" as a string
    input_size: int = 224
    normalizer_mean: Sequence[float] = (0.485, 0.456, 0.406)
    normalizer_std: Sequence[float] = (0.229, 0.224, 0.225)
    train_augmentations: Sequence[str] = ("Normalize",
                                           "ShiftScaleRotate",
                                           "HorizontalFlip",
                                            "Resize",
                                           "GaussianBlur")
    val_augmentations: Sequence[str] = ("Normalize","Resize")

    # Architecture info
    encoder_name: str = 'efficientnet-b3'  # 'efficientnetb3-pytorch', 'resnet18', or another encoder name.
    output_path: Optional[Path] = None
    num_classes: int = 4  # number of classes predicted by the neural network
    pretrained_encoder_weights: str = ''  #
    freeze_encoder: bool = False
    save_weights_only: bool = False  # says if weights should be logged
    log_weights: bool = False
    checkpoint_path: str = ""

    # labels
    label_names = ("aegypti", "albopictus", "koreicus", "japonicus")
    label_indices = (0, 1, 2, 3)  # TODO

    # -- Training params
    lr: float = 1e-4  #
    auto_tune_lr: bool = False
    optimizer: str = "Adam"  # or "SGD"
    batch_size: int = 32
    max_epochs: int = 10
    max_steps: int = -1
    criterion: str = "ce"  # loss function
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 1.0
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    #max_steps: int = 10  # if set to another value then it will be the earliest between max_epochs and max_steps
    max_time = "00:12:00:00"  # stop after 12 hours
    weight_decay: float = 1e-3
    precision: str = "32"  # 16-mixed 32
    # threshold_prediction: float = 0.5

    # -- Lr scheduling
    metric_to_monitor_lr_scheduler: str = "f1_score_macro_valid"
    lr_scheduler: str = "CosineAnnealingLR"  # Options are "CosineAnnealingWarmRestarts", "MultiplicativeLR", OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
    exp_lr_gamma: float = 0.85
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.1  # <1 !
    min_lr: float = 1e-5
    lr_scheduler_mode: str = 'max'  # goal is to maximize the metric to monitor
    T_0: int = 15  # period for decreasing lr
    T_mult: int = 1

    # -- Early stopping
    metric_to_monitor_early_stop: str = "f1_score_macro_valid"
    early_stopping_patience: int = 5
    early_stopping: bool = False
    early_stop_mode: str = "max"  # goal is to maximize the metric to monitor

    # -- Explainability
    log_pred_every_nstep: int = 1000
    log_pred_every_nepoch: int = 10
    log_saliency_map: bool = False
    colormap: str = 'viridis'
    saliency_map_method: str = 'gcam'  # 'ggcam', 'gcampp', 'gbp'
    attention_layer: str = ''
    disable_media_logging: bool = False

