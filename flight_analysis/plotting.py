import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_analysis(flight_id, seg_id, processed_segment_data, df_raw_flight, final_df, ds_era5, out_dir="outputs/plots"):
    """
    Generates the final multi-panel plot for a single segment.
    """
    # --- full plotting code from your script ---
    # (unchanged, copy everything you had)
    ...
