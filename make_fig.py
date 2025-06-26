import matplotlib.pyplot as plt
import scienceplots
import scipy.stats
import scipy.special
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":11})  
plt.style.use('science')

#plt.figure(figsize=(4, 4))

plt.rc('axes', labelsize=13) 
plt.rc('legend', fontsize=13) 

df = pd.read_csv('data/tight_lb.csv')

plt.plot(df['Iteration'], df['Estimation'], color='b', label='Estimation', linewidth = 1)
plt.plot(df['Iteration'], df['Exact'], linestyle='--', color='r', label='Exact bound', linewidth = 1)

err = []
c = (df['Estimation'][0] - df['Exact'][0])
for i in range(len(df['Iteration'])):
    err.append(c / np.sqrt(i + 1))
err = np.array(err)

plt.fill_between(df['Iteration'], df['Estimation'] - err, df['Estimation'] + err, color='gray', alpha=0.3, label='Error band')
#plt.plot(df['Iteration'], df['Estimation'] + err, linestyle='--', color='gray')
#plt.plot(df['Iteration'], df['Estimation'] - err, linestyle='--', color='gray')

plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("fig/tight_lb.pdf", dpi=400)