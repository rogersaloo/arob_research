import torch
import warnings
import pandas as pd
import image_tabular as it
from typing import Tuple
from fastai.vision import *
from fastai.tabular import *
# from image_tabular import core
# from image_tabular.dataset import ImageTabDataset
# from image_tabular.core import *
# from image_tabular.dataset import *
# from image_tabular.model import *
# from image_tabular.metric import *
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
# use gpu by default if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = ('alldata/')

#Import train and test based on the tabular data
train_df = pd.read_csv(f"{data_path}train_metadata.csv")
test_df = pd.read_csv(f"{data_path}test_metadata.csv")

print(len(train_df), len(test_df))

it.core.get_valid_index()