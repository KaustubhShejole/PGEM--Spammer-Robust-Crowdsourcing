import pandas as pd
import re

raw_text = """GradEM -- Accuracy: 0.7915597438812256 ± 6.419252992928882e-05, Weighted Accuracy: 0.8731638073921204 ± 6.222083046756577e-05, Kendall's Tau: 0.5783516802614771 ± 0.00012732512702058125
PGEM -- Accuracy: 0.7917580664157867 ± 0.00013823902492913592, Weighted Accuracy: 0.8732296645641326 ± 0.00012233771098530712, Kendall's Tau: 0.5787451048521977 ± 0.0002742117532629092
Simple BT -- Accuracy: 0.7900686264038086, Weighted Accuracy: 0.8719564080238342, Kendall's Tau: 0.575393885555222
RC -- Accuracy: 0.6521800756454468, Weighted Accuracy: 0.71303790807724, Kendall's Tau: 0.2891796316210329
CrowdBT -- Accuracy: 0.7044227123260498 ± 0.0008876429468347528, Weighted Accuracy: 0.7640507102012635 ± 0.000879096424574269, Kendall's Tau: 0.38595374488838885 ± 0.001675885754137508
FactorBT -- Accuracy: 0.6973661780357361, Weighted Accuracy: 0.74295973777771, Kendall's Tau: 0.3726308644381563"""

lines = raw_text.strip().split('\n')
data = []

for line in lines:
    method_part, metrics_part = line.split(" -- ")
    method = method_part.strip()
    
    # Regex pattern: identifies the number and the optional ± part
    pattern = r"([\d\.]+)( ± ([\d\.]+))?"
    
    acc_match = re.search(f"Accuracy: {pattern}", metrics_part)
    wacc_match = re.search(f"Weighted Accuracy: {pattern}", metrics_part)
    tau_match = re.search(f"Kendall's Tau: {pattern}", metrics_part)
    
    def get_val_std(match):
        if match:
            mean = float(match.group(1))
            std = float(match.group(3)) if match.group(3) else 0.0
            return mean, std
        return None, 0.0

    acc_mean, acc_std = get_val_std(acc_match)
    wacc_mean, wacc_std = get_val_std(wacc_match)
    tau_mean, tau_std = get_val_std(tau_match)
    
    data.append([method, acc_mean, acc_std, wacc_mean, wacc_std, tau_mean, tau_std])

cols = ['Method', 'acc_mean', 'acc_std', 'wacc_mean', 'wacc_std', 'tau_mean', 'tau_std']
df = pd.DataFrame(data, columns=cols)

# Save to CSV
df.to_csv('baselines_faceage.csv', index=False)