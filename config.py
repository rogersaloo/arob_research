import torch
import albumentations as a
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 4
LEARNING_RATE = 4e-4
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh29_9.pth.tar"
CHECKPOINT_GEN_Z = "genz29_9.pth.tar"
CHECKPOINT_CRITIC_H = "disc29_9.pth.tar"
CHECKPOINT_CRITIC_Z = "disc29_9.pth.tar"

transforms = a.Compose(
    [
        a.Resize(width=256, height=256),
        a.HorizontalFlip(p=0.5),
        a.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)