from spammer_types import add_random_spammer, add_anti_personas, add_position_biased_spammers

#Take these as arguments
spammer_percent = 40
spammer_type = 'combine'

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
from collections import defaultdict

with open("../data/FaceAgeDF1.pickle", 'rb') as handle:
    df_faceage = pickle.load(handle)

import pandas as pd
df = pd.read_csv('../age_dataset/crowd_labels.csv')
print(df.tail())

if spammer_type == 'random':
    df, spammer_ids = add_random_spammer(df, spammer_percent*0.01)
elif spammer_type == 'anti':
    df, spammer_ids = add_anti_personas(df, spammer_percent*0.01)
elif spammer_type == 'left':
    df, spammer_ids = add_position_biased_spammers(df, spammer_percent*0.01, 'left')
else:
    df, spammer_ids = add_position_biased_spammers(df, spammer_percent*0.01, 'right')

    
    
import os

# Ensure the output directory exists
output_dir = "spammer_data"
os.makedirs(output_dir, exist_ok=True)

# Construct the filename
filename = f"df_FaceAge_{spammer_type}_{int(spammer_percent)}.csv"
filepath = os.path.join(output_dir, filename)

df.rename(columns={"performer": "worker"}, inplace=True)
# Save the DataFrame
df.to_csv(filepath, index=False)

print(f"Saved spammer dataset to: {filepath}")    

# Get unique performers in the DataFrame
unique_performers = list(df['worker'].unique())
print(len(unique_performers))
# Create a mapping from performer to a numeric label
performer_label_dict = defaultdict(lambda: -1)
for i, performer in enumerate(unique_performers):
    performer_label_dict[performer] = i

item_labels = list(df_faceage['full_path'])
item_label_dict = defaultdict(lambda:-1)
for i, item in enumerate(item_labels):
    item_label_dict[item] = i
PC_faceage_spm = {}
ans = 0
for performer, group in df.groupby('worker'):
    key = performer_label_dict[performer]  # map performer to key
    if key not in PC_faceage_spm:
        PC_faceage_spm[key] = list()  # initialize if not present

    for _, row in group.iterrows():
        left = row["left"].split("/")[-1]
        right = row["right"].split("/")[-1]
        winner = row["label"].split("/")[-1]  # whichever side was chosen

        # Map to indices
        left_label = item_label_dict[left]
        right_label = item_label_dict[right]
        winner_label = item_label_dict[winner]

        # Ensure pair stored as (winner, loser)
        if winner_label == left_label:
            PC_faceage_spm[key].append((left_label, right_label))
        else:
            PC_faceage_spm[key].append((right_label, left_label))
        
print(ans)
import pickle
import os

# Make sure the directory exists
os.makedirs("spammer_data", exist_ok=True)

# Save PC_passage
with open(f"spammer_data/FaceAge_{spammer_type}_{spammer_percent}.pickle", 'wb') as handle:
    pickle.dump(PC_faceage_spm, handle, protocol=pickle.HIGHEST_PROTOCOL)