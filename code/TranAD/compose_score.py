
from src.evaluate import bf_search
import numpy as np
import pandas as pd
from data_config import *
from tqdm import tqdm

alpha_g = 0.95
for machine_id in eval_data_list:
    score_g = np.load(exp_dir/f'result/{machine_id}/test_score_g.npy')
    score_d = np.load(exp_dir/f'result/{machine_id}/test_score_d.npy')
    score_gd = score_g * alpha_g + (1-alpha_g)*score_d
    np.save(exp_dir/f'result/{machine_id}/test_score_gd_{955}.npy', score_gd)