import pandas as pd
import re

raw_text = """GradEM -- Accuracy: 0.7915597438812256 ± 6.419252992928882e-05, Weighted Accuracy: 0.8731638073921204 ± 6.222083046756577e-05, Kendall's Tau: 0.5783516802614771 ± 0.00012732512702058125
PGEM -- Accuracy: 0.7917580664157867 ± 0.00013823902492913592, Weighted Accuracy: 0.8732296645641326 ± 0.00012233771098530712, Kendall's Tau: 0.5787451048521977 ± 0.0002742117532629092
Simple BT -- Accuracy: 0.7900686264038086, Weighted Accuracy: 0.8719564080238342, Kendall's Tau: 0.575393885555222
BARP -- Accuracy: 0.7911810278892517, Weighted Accuracy: 0.8728458285331726, Kendall's Tau: 0.5776003949914079 
RC -- Accuracy: 0.7804011106491089, Weighted Accuracy: 0.8644054532051086, Kendall's Tau: 0.5582118883509699 
CrowdBT -- Accuracy: 0.7906020522117615 ± 0.00020234240149123216, Weighted Accuracy: 0.8724876284599304 ± 0.00013249322393198598, Kendall's Tau: 0.5764519553075644 ± 0.00040137813726117994
FactorBT -- Accuracy: 0.7904868125915527, Weighted Accuracy: 0.8285281658172607, Kendall's Tau: 0.5762237655043003"""

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
df_baselines.to_csv('baselines_faceage.csv', index=False)