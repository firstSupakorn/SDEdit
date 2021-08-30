import sys, torch
import yaml
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functions.process_data import *
from main import dict2namespace
from runners.image_editing import *
from models.diffusion import Model
from colab_utils.utils import *

import warnings
warnings.filterwarnings("ignore")


sys.path.append('/content/tutorial_code')
if not torch.cuda.is_available():
    print("Change runtime type to include a GPU.")
device = "cuda"


print(device)
dataset = "LSUN"
category = "church_outdoor"
data_name = "lsun_church"
sample_step = 50    
model, betas, num_timesteps, logvar = load_model(dataset, category, "church.yml")
total_noise_levels = 500
SDEditing(betas, logvar, model, data_name, sample_step, total_noise_levels, n=3)