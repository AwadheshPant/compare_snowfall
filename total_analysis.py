import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


# %%
print("Starting total analysis...")


# Load the data from the CSV files
csv_file_path = '/work/apant/carra/project_root/notebooks/outputs/'
# file_path = '/work/apant/carra/project_root/notebooks/outputs/AFLUX_P5_RF08_high_level.csv'
# data_rf08 = pd.read_csv(file_path)
csv_files = glob.glob(csv_file_path + '*.csv')

plt.figure(figsize=(10, 6), dpi=600)
for file in csv_files:
    data = pd.read_csv(file)
    plt.scatter(data['tp'], data['snow_rate_mean'], color='blue')
    # plt.hexbin(data['tp'], data['snow_rate_mean'], gridsize=10, cmap='Blues', mincnt=1, edgecolors='black', linewidths=0.1)

plt.xlim(0.001, 1)
plt.ylim(0.001, 1)
plt.plot([0, 1], [0, 1], 'k-',alpha=0.4)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xscale('log')
plt.yscale('log')
# plt.colorbar(label='Flight longitude')
plt.xlabel('ERA5 Snow Rate (mm/hr)')
plt.ylabel('Flight Snow Rate (mm/hr)')
# plt.legend()
plt.show()



# %%
