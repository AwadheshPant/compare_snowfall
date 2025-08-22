import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
# %%
import matplotlib.gridspec as gridspec
from py import log


# %%
print("Starting total analysis...")


# Load the data from the CSV files
csv_file_path = '/work/apant/carra/project_root/notebooks/outputs/'
# file_path = '/work/apant/carra/project_root/notebooks/outputs/AFLUX_P5_RF08_high_level.csv'
# data_rf08 = pd.read_csv(file_path)
csv_files = glob.glob(csv_file_path + '*.csv')

all_tp = []
all_snow_mean_rates = []

# %%


custom_bins = np.arange(0.001, 0.21, 0.01)
log_bins = np.logspace(np.log10(0.001), np.log10(0.21), num=22)
print(custom_bins) # for histogram for both the values

for file in csv_files:
    data = pd.read_csv(file)
    all_tp.extend(data['tp'])
    all_snow_mean_rates.extend(data['snow_rate_mean'])

    # ax0 = fig.add_subplot(spec[0, 0])
    # ax0.scatter(data['tp'], data['snow_rate_mean'], color='blue')

    # ax1 = fig.add_subplot(spec[0, 1])
    # ax1.hist(data['tp'], bins=custom_bins, histtype='bar', color='blue', linewidth=1.5, alpha=0.4, label='ERA5 Snow Rate')
    # ax1.hist(data['snow_rate_mean'], bins=custom_bins, histtype='step', color='orange', label='Flight Snow Rate')
    # ax1.set_xlabel('Snow Rate (mm
    # plt.hexbin(data['tp'], data['snow_rate_mean'], gridsize=10, cmap='Blues', mincnt=1, edgecolors='black', linewidths=0.1)
print(len(all_tp), len(all_snow_mean_rates))
# %%
fig = plt.figure(figsize=(10, 4), dpi=300)
spec = fig.add_gridspec(nrows=1, ncols=2)

ax0 = fig.add_subplot(spec[0, 0]) # for scatter plot
ax0.scatter(all_tp, all_snow_mean_rates, color='blue', alpha=0.4)
ax0.set_xlim(0.0001, 1)
ax0.set_ylim(0.0001, 1)
ax0.plot([0, 1], [0, 1], 'k-',alpha=0.4)
ax0.grid(True, linestyle='--', alpha=0.3)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel('ERA5 Snow Rate (mm/hr)')
ax0.set_ylabel('Flight Snow Rate (mm/hr)')
ax1 = fig.add_subplot(spec[0, 1])
ax1.hist(all_tp, bins=log_bins, histtype='bar', color='blue', linewidth=1.5, alpha=0.4, label='ERA5 Snow Rate')
ax1.hist(all_snow_mean_rates, bins=log_bins, histtype='step', color='orange', label='Flight Snow Rate')
ax1.set_xlabel('mm/hr')
ax1.set_xscale('log')
ax1.set_ylabel('Count')
plt.legend()
plt.show()

# makea joint scatter and histogram plot
# %%
def joint_scatter_histogram(x, y, bins=custom_bins):
    fig = plt.figure(figsize=(8,8), dpi=300)
    gspec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[4, 1], height_ratios=[1, 4],
                      hspace=0.05, wspace=0.05)
    ax_scatter = fig.add_subplot(gspec[1,0])
    ax_histx = fig.add_subplot(gspec[0,0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gspec[1,1], sharey=ax_scatter)

    # now plotting
    ax_scatter.scatter(x, y, alpha=0.4, color='green')
    ax_scatter.set_xlim(0.0001, 1)
    ax_scatter.set_ylim(0.0001, 1)
    ax_scatter.plot([0, 1], [0, 1], 'k-',alpha=0.4)
    ax_scatter.grid(True, linestyle='--', alpha=0.3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.set_xlabel('ERA5 Snow Rate (mm/hr)')
    ax_scatter.set_ylabel('Flight Snow Rate (mm/hr)')


    # histogram plot with log bins and log scales
    logbins_x = np.logspace(np.log10(0.001), np.log10(0.21), num=22)
    logbins_y = np.logspace(np.log10(0.001), np.log10(0.21), num=22)
    # ax_histx.set_xlabel('mm/hr')
    # ax1.set_xscale('log')
    # ax1.set_ylabel('Count')
    ax_histx.set_xscale('log')
    # ax_histy.set_yscale('log')
    ax_histx.hist(x, bins=logbins_x, histtype='step', color='blue', linewidth=1.5, alpha=0.4, label='ERA5')
    ax_histy.hist(y, bins=logbins_y, histtype='step', orientation='horizontal', linewidth=1.5, color='orange', alpha=0.7, label='Flight')
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    ax_histx.legend()
    ax_histy.legend()
    ax_histx.set_ylabel('Count')
    ax_histy.set_xlabel('Count')
    ax_histx.grid(True, linestyle='--', alpha=0.3)
    ax_histy.grid(True, linestyle='--', alpha=0.3)
    plt.show()

joint_scatter_histogram(all_tp, all_snow_mean_rates, bins=custom_bins)
    
# %%





# %%
