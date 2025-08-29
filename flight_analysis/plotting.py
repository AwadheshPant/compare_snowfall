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
    flight_plot_dir = os.path.join(out_dir, flight_id)
    os.makedirs(flight_plot_dir, exist_ok=True)


    fig = plt.figure(figsize=(17, 5), facecolor="white", layout='constrained', dpi=600)
    spec = fig.add_gridspec(2, 3)

    # Panel 1: Vertical Profile
    
    ax0 = fig.add_subplot(spec[0, 0])
    
    
    segment_data = processed_segment_data[seg_id]
    time = pd.to_datetime(segment_data['time'])
    ax0.plot(time, segment_data['alt'] * 1e-3, label=f'Flight Altitude', color='k')
    pcm = ax0.pcolormesh(time, segment_data['height_above_0'] * 1e-3, segment_data['dbz_filtered'].T, cmap='jet', shading='nearest', vmin=-20, vmax=20)
    
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label('dBZ')
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_ylabel('Height (km)')
    ax0.set_ylim(0, 4)
    # show title with only date and time in format hh:mm    
    ax0.set_title(f'{seg_id}') # only date and time hh:mm

    # ax0.set_title(f'{seg_id}{segment_data["time"][0]} - {segment_data["time"][-1]}') # only date and time hh:mm


    # Panel 2: Snow Rate Time Series

    try_join = df_raw_flight.merge(final_df, on='segment_id', how='left')
    try_join_new = try_join[['segment_id', 'time_x', 'snow_rate_mmhr', 'snow_rate_mean', 'tp']]
    # print(f"try_join_new: {try_join_new.head()}")

    ax1 = fig.add_subplot(spec[1, 0], sharex=ax0)
    ax1.plot(df_raw_flight['time'], df_raw_flight['snow_rate_mmhr'], label='Flight Snow Rate', color='lightblue', marker='.', linestyle='-', alpha=0.1)
    ax1.plot(try_join_new['time_x'], try_join_new['snow_rate_mean'], label='Resampled Rate', color='green', linestyle='-', alpha=0.5)
    ax1.scatter(final_df['time'], final_df['tp'], label='ERA5', s=60, color='red', marker='+')
    ax1.set_ylabel("Snow Rate (mm/hr)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.legend(ncols=3)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 3: ERA5 Map with Flight Track
    levels = np.arange(0, 0.22, 0.02)
    colors = ['#E0E0E0', '#B0C4DE', '#87CEEB', '#6495ED', '#4169E1', '#3A5FCD', '#324F9A', '#3C8E8F', '#45A947',
              '#66CD00', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500',
              '#D12727', '#B22222']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    proj = ccrs.NorthPolarStereo()
    ax20 = fig.add_subplot(spec[:, 1], projection=proj)
    ax20.set_extent([3.5, 15, 77.8, 81], crs=ccrs.PlateCarree())
    ax20.coastlines()
    ax20.add_feature(cfeature.LAND, facecolor='lightgray')
    gl = ax20.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    
    era5_map_time = final_df['time_near'].min()
    lsm_era5 = xr.open_dataset("/work/apant/land_sea_mask/era5_lsm.nc")
    tp_mm_sea_time = ds_era5['tp'].sel(valid_time=era5_map_time) * 1000
    tp_mm_sea = tp_mm_sea_time.where(lsm_era5['lsm'].isel(valid_time=0)==0)

    mesh = ax20.pcolormesh(tp_mm_sea.longitude, tp_mm_sea.latitude, tp_mm_sea, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
    
    ax20.scatter(df_raw_flight['lon'], df_raw_flight['lat'], s=10, color='grey', transform=ccrs.PlateCarree(), alpha=0.2)
    ax20.scatter(df_raw_flight['lon'][0], df_raw_flight['lat'][0], color='red', s=40, transform=ccrs.PlateCarree(), marker='<', label='start')
    sc_flight = ax20.scatter(final_df['lon'], final_df['lat'], c=final_df['snow_rate_mean'], cmap=cmap, norm=norm, s=100, transform=ccrs.PlateCarree(), label='Flight_resample')

    bbox = ax20.get_position()
    cax = fig.add_axes([bbox.x1 + 0.00001, bbox.y0 + 0.03, 0.01, bbox.height + 0.03])
    fig.colorbar(mesh, cax=cax, orientation='vertical', ticks=levels)
    ax20.legend(fontsize='x-small')
    ax20.set_title(f'ERA5 snow rate, mm/hr [{era5_map_time.strftime("%H:%M UTC")}]')

    # Panel 4: Box Plot Comparison
    ax21 = fig.add_subplot(spec[:, 2])

    # Now merge df_raw_flight with final_df on the new 'segment_id' column
    
    groups = try_join.groupby('tp')['snow_rate_mmhr']

    positions = []
    data = []
    for tp_value, group in groups:
        positions.append(tp_value)
        data.append(group.values)

    mean_props = {'marker': 'o', 'markerfacecolor': 'green', 'markersize': 5}

    ax21.boxplot(data, 
                 positions=positions, widths=0.005, whis=0, 
                 showmeans=True, showfliers=False,
                 meanprops=mean_props)
    
    ax21.set_xlim(-0.005, 0.2)  # 0.15 to 0.3 and 0.05 to 0.1
    ax21.set_ylim(-0.005, 0.4)
    tick_values_x = [0.0, 0.05, 0.1, 0.15, 0.2]
    tick_values_y = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # tick_values = [0.0, 0.1, 0.2, 0.3]
    ax21.set_xticks(tick_values_x)
    ax21.set_yticks(tick_values_y)
    ax21.set_xticklabels(tick_values_x)
    ax21.set_yticklabels(tick_values_y)
    
    # max_val = max(final_df['tp'].max(), final_df['snow_rate_mean'].max())
    # plt.plot([0, max_val + 0.05], [0, max_val + 0.05], 'k--')
    plt.plot([0, 0.2 + 0.05 ], [0, 0.2 + 0.05], 'k--')

    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    ax21.set_xlabel("ERA5 Snow Rate")
    ax21.set_ylabel("Flight Snow Rate")
    
    # plt.suptitle(f"Analysis for Segment: {seg_id}", fontsize=16) 
    # plt.show()
    plot_path = os.path.join(flight_plot_dir, f"{seg_id}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {plot_path}")