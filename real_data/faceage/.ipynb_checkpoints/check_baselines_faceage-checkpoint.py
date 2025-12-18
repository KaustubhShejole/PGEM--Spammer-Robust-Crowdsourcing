import numpy as np
import choix
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from matplotlib import colors
import pandas as pd
import seaborn as sns
import pickle
import os
import sys


sys.path.append(os.path.abspath('../../'))
from metrics import compute_acc, compute_weighted_acc
from opt_fair import *


import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


