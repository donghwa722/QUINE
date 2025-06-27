import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scienceplots

# 설정
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 11
})
plt.style.use('science')

plt.rc('axes', labelsize=13)
plt.rc('legend', fontsize=12)

df = pd.read_csv('data/both_ub.csv')
fig, ax = plt.subplots(figsize=(8, 4))

color_tight = "#1f77b4" 
color_loose = "#6baed6"
bound_tight = "#d62728"
bound_loose = "#9467bd"

ax.plot(df['Iteration'], df['Tight_Exact'], linestyle='--', color=bound_tight, label='Tight lower bound', linewidth=1.5)
ax.plot(df['Iteration'], df['Loose_Exact'], linestyle='-.', color=bound_loose, label='Loose upper bound', linewidth=1.5)

ax.plot(df['Iteration'], df['Tight_Estimation'], color=color_tight, label='Tight estimation', linewidth=2, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=1.5, markevery=5)
ax.plot(df['Iteration'], df['Loose_Estimation'], color=color_loose, label='Loose estimation', linewidth=2, marker='^', markersize=8, markerfacecolor='white', markeredgewidth=1.5, markevery=5)
ax.fill_between(df['Iteration'], df['Tight_Estimation'], df['Loose_Estimation'], color='gray', alpha=0.3, edgecolor='black', linewidth = 0.0, hatch = '//', label='Bound gap')

handles, labels = plt.gca().get_legend_handles_labels()
new_order = [3, 1, 2, 0, 4] 
ax.legend([handles[i] for i in new_order], [labels[i] for i in new_order], loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)

ax.set_xlabel('Iteration')
ax.set_ylabel('Entropy')
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, linestyle='--', alpha=0.3)



plt.tight_layout()
plt.subplots_adjust(right=0.75)

plt.savefig('fig/both_ub.pdf', dpi=400, bbox_inches='tight')
