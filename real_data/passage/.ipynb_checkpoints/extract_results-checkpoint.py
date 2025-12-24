import pandas as pd
import re

raw_text = """GradEM -- Accuracy: 0.6985240876674652 ± 0.0008998396815844701, Weighted Accuracy: 0.7583310902118683 ± 0.0009670574004366093, Kendall's Tau: 0.3748170366665421 ± 0.0016989014194708297
PGEM -- Accuracy: 0.6946465194225311 ± 0.0005217393074741697, Weighted Accuracy: 0.755181896686554 ± 0.0006608944057229961, Kendall's Tau: 0.3674961233890967 ± 0.0009850565225552452
Simple BT -- Accuracy: 0.6804668307304382, Weighted Accuracy: 0.74333256483078, Kendall's Tau: 0.3408721163582005
BARP -- Accuracy: 0.6920762658119202, Weighted Accuracy: 0.7587689757347107, Kendall's Tau: 0.36267695314604337 
RC -- Accuracy: 0.6521800756454468, Weighted Accuracy: 0.71303790807724, Kendall's Tau: 0.2891796316210329
CrowdBT -- Accuracy: 0.7044227123260498 ± 0.0008876429468347528, Weighted Accuracy: 0.7640507102012635 ± 0.000879096424574269, Kendall's Tau: 0.38595374488838885 ± 0.001675885754137508
FactorBT -- Accuracy: 0.6973661780357361, Weighted Accuracy: 0.74295973777771, Kendall's Tau: 0.3726308644381563"""

lines = raw_text.strip().split('\n')
data = []

# Robust regex for numbers: handles floats, integers, and scientific notation (e.g., 6.4e-05)
# This pattern matches: [optional sign][digits][optional dot + digits][optional 'e' + sign + digits]
num_pattern = r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?"

# Combined pattern: Group 1 is the Mean, Group 2 is the Standard Deviation (if it exists)
# Uses (?:\s*±\s*) to non-capturingly match the plus-minus sign with flexible spacing
metric_pattern = rf"({num_pattern})(?:\s*±\s*({num_pattern}))?"

for line in lines:
    if " -- " not in line: continue
    method_part, metrics_part = line.split(" -- ")
    method = method_part.strip()
    
    def extract_metric(label, text):
        match = re.search(f"{label}: {metric_pattern}", text)
        if match:
            mean = float(match.group(1))
            # group(2) will be None if the ± part isn't present
            std = float(match.group(2)) if match.group(2) else 0.0
            return mean, std
        return 0.0, 0.0

    acc_mean, acc_std = extract_metric("Accuracy", metrics_part)
    wacc_mean, wacc_std = extract_metric("Weighted Accuracy", metrics_part)
    tau_mean, tau_std = extract_metric("Kendall's Tau", metrics_part)
    
    data.append([method, acc_mean, acc_std, wacc_mean, wacc_std, tau_mean, tau_std])

cols = ['Method', 'acc_mean', 'acc_std', 'wacc_mean', 'wacc_std', 'tau_mean', 'tau_std']
df_baselines = pd.DataFrame(data, columns=cols)

# Display to verify scientific notation was captured correctly
print(df_baselines[['Method', 'acc_std']])

# Save to CSV
df_baselines.to_csv('baselines_passage.csv', index=False)