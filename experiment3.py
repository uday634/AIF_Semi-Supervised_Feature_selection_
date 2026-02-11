# plot_aif_vs_logistic_vs_anova.py
"""
Plots the results from aif_vs_logistic_vs_anova.csv
Generates clear comparison charts for AUC, AP, Prec@1%, Runtime, and Selected Features
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV
df = pd.read_csv("aif_vs_logistic_vs_anova.csv")

# Clean dataset names for plotting
df['dataset'] = df['dataset'].str.replace('.npz', '', regex=False)

print("Data loaded successfully!")
print(df.round(4))

# ────────────────────────────────────────────────────────────────
# Plot Settings
# ────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8')
x = np.arange(len(df))
width = 0.28

# 1. AUC Comparison
plt.figure(figsize=(16, 8))
plt.bar(x - width, df['AUC_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x, df['AUC_LOGISTIC'], width, label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['AUC_ANOVA'], width, label='AIF + ANOVA FS', color='#2ca02c')

plt.xticks(x, df['dataset'], rotation=45, ha='right', fontsize=10)
plt.ylabel('AUC')
plt.title('AUC Comparison: Original AIF vs Logistic FS vs ANOVA FS')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_AUC_comparison.png', dpi=300)
plt.show()

# 2. Average Precision Comparison
plt.figure(figsize=(16, 8))
plt.bar(x - width, df['AP_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x, df['AP_LOGISTIC'], width, label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['AP_ANOVA'], width, label='AIF + ANOVA FS', color='#2ca02c')

plt.xticks(x, df['dataset'], rotation=45, ha='right', fontsize=10)
plt.ylabel('Average Precision')
plt.title('Average Precision Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_AP_comparison.png', dpi=300)
plt.show()

# 3. Precision @ Top 1% (Most Important Metric)
plt.figure(figsize=(16, 8))
plt.bar(x - width, df['Prec@1%_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x, df['Prec@1%_LOGISTIC'], width, label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['Prec@1%_ANOVA'], width, label='AIF + ANOVA FS', color='#2ca02c')

plt.xticks(x, df['dataset'], rotation=45, ha='right', fontsize=10)
plt.ylabel('Precision @ Top 1%')
plt.title('Precision @ Top 1% Comparison (Critical Metric)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_Prec1_comparison.png', dpi=300)
plt.show()

# 4. Selected Features (Logistic vs ANOVA)
plt.figure(figsize=(16, 8))
plt.bar(x - width/2, df['features_logistic'], width, label='Logistic FS', color='#ff7f0e')
plt.bar(x + width/2, df['features_anova'], width, label='ANOVA FS', color='#2ca02c')

plt.xticks(x, df['dataset'], rotation=45, ha='right', fontsize=10)
plt.ylabel('Number of Selected Features')
plt.title('Feature Selection Count Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_FeatureSelection_Count.png', dpi=300)
plt.show()

# 5. Runtime Comparison
plt.figure(figsize=(16, 8))
plt.bar(x - width, df['runtime_s'], width, label='Original AIF', color='#1f77b4')
plt.bar(x, df['runtime_s'], width, label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['runtime_s'], width, label='AIF + ANOVA FS', color='#2ca02c')

plt.xticks(x, df['dataset'], rotation=45, ha='right', fontsize=10)
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_Runtime_comparison.png', dpi=300)
plt.show()

print("\nAll plots saved successfully!")
print("Files saved:")
print("   plot_AUC_comparison.png")
print("   plot_AP_comparison.png")
print("   plot_Prec1_comparison.png")
print("   plot_FeatureSelection_Count.png")
print("   plot_Runtime_comparison.png")