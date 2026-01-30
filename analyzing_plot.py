import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("dataset_baseline_npc/labels.csv")
df2 = pd.read_csv("data_npc_36K/labels.csv")

plt.figure(figsize=(6,4))
plt.hist(df1["steer"], bins=100, density=True)
plt.xlabel("Steering value")
plt.ylabel("Density")
plt.title("Steering distribution (36K NPC dataset)")
plt.show()

import numpy as np

plt.figure(figsize=(6,4))
plt.hist(np.abs(df1["steer"]), bins=100, density=True)
plt.xlabel("|Steering|")
plt.ylabel("Density")
plt.title("Absolute steering distribution")
plt.show()


thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
for t in thresholds:
    frac = (df1["steer"].abs() > t).mean()
    print(f"|steer| > {t:.2f}: {frac*100:.2f}%")
'''
plt.figure(figsize=(6,4))
plt.hist(df2["steer"], bins=100, density=True)
plt.xlabel("Steering value")
plt.ylabel("Density")
plt.title("Steering distribution (36K NPC dataset)")
plt.show()


plt.figure(figsize=(6,4))
plt.hist(np.abs(df2["steer"]), bins=100, density=True)
plt.xlabel("|Steering|")
plt.ylabel("Density")
plt.title("Absolute steering distribution")
plt.show()


thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
for t in thresholds:
    frac = (df2["steer"].abs() > t).mean()
    print(f"|steer| > {t:.2f}: {frac*100:.2f}%")
'''


plt.figure(figsize=(6,4))
plt.hist(np.abs(df1["steer"]), bins=100, density=True, alpha=0.6, label="6K")
plt.hist(np.abs(df2["steer"]), bins=100, density=True, alpha=0.6, label="36K")
plt.xlabel("|Steering|")
plt.ylabel("Density")
plt.legend()
plt.title("Absolute steering comparison")
plt.show()
