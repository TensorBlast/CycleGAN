from dataclasses import dataclass, field
from typing import List, Union, Optional

@dataclass
class Options:
    # Required options
    dataroot: str # Path to images

    # Options with default values
    name: str = "experiment_name"
    gpu_ids: Union[str, List[int]] = "0"
    checkpoints_dir: str = "./checkpoints"

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netD: str = "basic"
    netG: str = "resnet_9blocks"
    n_layers_D: int = 3
    norm: str = "instance"
    init_type: str = "normal"
    init_gain: float = 0.02
    no_dropout: bool = False

    dataset_mode: str = "unaligned"
    direction: str = "AtoB"
    serial_batches: bool = False
    num_threads: int = 4
    batch_size: int = 1
    load_size: int = 286
    crop_size: int = 256
    max_dataset_size: int = float("inf")
    preprocess: str = "resize_and_crop"
    no_flip: bool = False
    display_winsize: int = 256

    epoch: str = "latest"
    load_iter: int = 0
    verbose: bool = False
    suffix: str = ""

    display_freq: int = 400
    display_ncols: int = 4
    display_id: int = 1
    display_server: str = "http://localhost"
    display_env: str = "main"
    display_port: int = 8097
    update_html_freq: int = 1000
    print_freq: int = 100
    no_html: bool = False
    save_latest_freq: int = 5000
    save_epoch_freq: int = 5
    save_by_iter: bool = False
    continue_train: bool = False
    epoch_count: int = 1
    phase: str = "train"
    n_epochs: int = 60
    n_epochs_decay: int = 40
    beta1: float = 0.5
    lr: float = 0.0002
    gan_mode: str = "lsgan"
    pool_size: int = 50
    lr_policy: str = "linear"
    lr_decay_iters: int = 50
    isTrain: bool = True

    results_dir: str = "./results/"
    aspect_ratio: float = 1.0
    eval: bool = False
    num_test: int = 50
    model: str = "test"
    # Overriding load_size for test phase
    load_size: int = field(default=256)

    use_wandb: bool = False  # Default: if specified, then init wandb logging
    wandb_project_name: str = "CycleGAN-and-pix2pix"

    def __post_init__(self):
        # Convert comma-separated gpu_ids to a list of integers
        if isinstance(self.gpu_ids, str):
            self.gpu_ids = [int(id_) for id_ in self.gpu_ids.split(',') if id_.strip().isdigit()]

        if self.isTrain:
            self.lambda_A : float = 10
            self.lambda_B : float = 10
            self.lambda_identity : float = 0.5


@dataclass
class TrainOptions:
    display_freq: int = 400
    display_ncols: int = 4
    display_id: int = 1
    display_server: str = "http://localhost"
    display_env: str = "main"
    display_port: int = 8097
    update_html_freq: int = 1000
    print_freq: int = 100
    no_html: bool = False
    save_latest_freq: int = 5000
    save_epoch_freq: int = 5
    save_by_iter: bool = False
    continue_train: bool = False
    epoch_count: int = 1
    phase: str = "train"
    n_epochs: int = 60
    n_epochs_decay: int = 40
    beta1: float = 0.5
    lr: float = 0.0002
    gan_mode: str = "lsgan"
    pool_size: int = 50
    lr_policy: str = "linear"
    lr_decay_iters: int = 50
    isTrain: bool = True