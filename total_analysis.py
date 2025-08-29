import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
# %%
import matplotlib.gridspec as gridspec
# from py import log


# %%
print("Starting total analysis...")


# Load the data from the CSV files
csv_file_path = '/work/apant/carra/project_root/notebooks/outputs/'
# file_path = '/work/apant/carra/project_root/notebooks/outputs/AFLUX_P5_RF08_high_level.csv'
# data_rf08 = pd.read_csv(file_path)
csv_files = glob.glob(csv_file_path + '*.csv')
# %%
all_dfs = []
# all_snow_mean_rates = []

custom_bins = np.arange(0.001, 0.21, 0.01)
log_bins = np.logspace(np.log10(0.001), np.log10(0.21), num=22)
print([f"{x:.3f}" for x in log_bins])

# print(custom_bins) # for histogram for both the values

for file in csv_files:
    data = pd.read_csv(file)
    df_subset = data[['valid_time', 'latitude', 'longitude', 'tp', 'snow_rate_mean']]
    all_dfs.append(df_subset)
    # all_tp.extend(data['tp'])
    # all_snow_mean_rates.extend(data['snow_rate_mean'])
df_all = pd.concat(all_dfs, ignore_index=True)  # main dataframe now

fig = plt.figure(figsize=(10, 4), dpi=300)
spec = fig.add_gridspec(nrows=1, ncols=2)
all_tp = df_all['tp'].values
all_snow_mean_rates = df_all['snow_rate_mean'].values
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
# adding era5 temp at 1000hpa variable for scatter plot

# load the file from a nc file
import xarray as xr
era5_temp_aflux = xr.open_dataset('/work/apant/carra/era5_temp_1000_2019_03_04.nc')

print(f'loading era5 temp dat for the AFLUX time periond 2019 03/04')
# %%

# now extract the selected values from the xarray corresponding to the dataframe
t_1000 = era5_temp_aflux['t'].sel(pressure_level=1000) - 273.15
# t_750 = 
# t_850 = 

def temp_select(temp_at_level, pressure_level, df):
    temp_select_level = temp_at_level.sel(
        valid_time = xr.DataArray(df['valid_time'].values, dims="points"),
        latitude = xr.DataArray(df['latitude'].values, dims="points"),
        longitude = xr.DataArray(df['longitude'].values, dims="points")
    )
    df[f'temp_{pressure_level}'] = temp_select_level.values
    return df


temp_values_1000 = t_1000.sel(
    valid_time = xr.DataArray(df_all['valid_time'].values, dims='points'),
    latitude = xr.DataArray(df_all['latitude'].values, dims='points'),
    longitude = xr.DataArray(df_all['longitude'].values, dims='points')
                              )
df_all['t_1000'] = temp_values_1000.values
print(df_all.head())

# %%
# now scatter plot tp and snow_rate_mean with t_1000 as color
plt.figure(figsize=(8, 6), dpi=300)
sc = plt.scatter(df_all['tp'], df_all['snow_rate_mean'], c=df_all['t_1000'], cmap='cividis', alpha=0.9)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.0001, 1)
plt.ylim(0.0001, 1)
plt.plot([0, 1], [0, 1], 'k-',alpha=0.4)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlabel('ERA5 Snow Rate (mm/hr)')
plt.ylabel('Flight Snow Rate (mm/hr)')
cbar = plt.colorbar(sc)
cbar.set_label('ERA5 Temp_1000hPa (°C)')
# plt.title('Snow Rate Comparison Colored by ERA5 Temp at 1000hPa')
plt.show()
# %%

# bins = [-100, -15, -10, -5, 0, 5]  # °C
bins = [-100, -15, -5, 0, 5]  # °C

labels = ["< -15°C","-15 to -5°C", "-5 to 0°C", "0+°C"]
df_all["temp_bin"] = pd.cut(df_all["t_1000"], bins=bins, labels=labels)

plt.figure(figsize=(8, 6), dpi=300)
for label, group in df_all.groupby("temp_bin"):
    plt.scatter(group['tp'], group['snow_rate_mean'], label=label, alpha=0.6)
plt.xscale('log'); plt.yscale('log')
plt.xlim(0.0001, 1); plt.ylim(0.0001, 1)
plt.plot([0, 1], [0, 1], 'k-', alpha=0.4)
plt.grid(True, ls='--', alpha=0.3)
plt.xlabel("ERA5 Snow Rate (mm/hr)")
plt.ylabel("Flight Snow Rate (mm/hr)")
plt.legend(title="Temp bins")
plt.show()

# %%
# Function to compute stats for each bin
from scipy.stats import pearsonr

def compute_stats(group):
    n = len(group)
    mean_obs = group["snow_rate_mean"].mean()
    mean_mod = group["tp"].mean()
    bias = mean_mod - mean_obs
    rel_bias = np.nan
    if mean_obs != 0:
        rel_bias = 100 * bias / mean_obs
    r = np.nan
    if n > 1:
        r, _ = pearsonr(group["tp"], group["snow_rate_mean"])
    return pd.Series({
        "N": n,
        "Mean_Flight": mean_obs,
        "Mean_ERA5": mean_mod,
        "Bias": bias,
        "Rel_Bias(%)": rel_bias,
        "Corr_r": r
    })

# Apply per bin
summary = df_all.groupby("temp_bin").apply(compute_stats).reset_index()
print(summary)
# %%
# print(np.min(all_tp))
np.min(all_tp[all_tp > 0])
np.min(all_snow_mean_rates[all_snow_mean_rates > 0])