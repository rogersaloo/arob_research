from random import random
import torch
from fastai.vision import *
from fastai.tabular import *
from image_tabular.core import *
from image_tabular.dataset import *
from image_tabular.model import *
from image_tabular.metric import *

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# use gpu by default if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


#Set the path of the data
data_path = ('data/synth_real_image/')