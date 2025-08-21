import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import sys
import ac3airborne
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
from geopy.distance import geodesic
from dotenv import load_dotenv

plt.rcParams['font.family'] = ['monospace']
ac3airborne.__version__
meta = ac3airborne.get_flight_segments()
cat = ac3airborne.get_intake_catalog()
from ac3airborne.tools.is_land import is_land



# Define ERA5 data directory for dynamic loading
ERA5_DATA_DIR = "/work/apant/carra/project_root/data/raw/era5/2018_19"

def get_era5_data_for_date(flight_date): #3----------------
    """
    Load ERA5 data for a given date based on time taken from first flight segment, hence get the flight day date.
    
    """
    era5_filename = "f949ed754fc9dda78f2bf0a0ef6bb70f.nc" # hardcoded for now, what to do here??

    era5_path = os.path.join(ERA5_DATA_DIR, era5_filename)
    
    if not os.path.exists(era5_path):
        print(f"ERA5 file not found: {era5_path}. Please check the file path.")
        return None
    
    ds = xr.open_dataset(era5_path)
    # Filter by date, using a buffer of a few hours
    start_time = pd.to_datetime(flight_date).floor('D')
    end_time = start_time + timedelta(days=1)
    
    ds_sel = ds.sel(valid_time=slice(start_time, end_time))
    
    return ds_sel

def process_single_segment_data(ds, seg_id, meta):  #2
    """
    Processes a single flight segment and returns the processed data.
    """
    segments = {
        s.get("segment_id"): s
        for camp in meta.values()
        for plat in camp.values()
        for flight in plat.values()
        for s in flight["segments"]
    }

    if seg_id not in segments:
        print(f"Segment {seg_id} not found.")
        return None, None
        
    seg = segments[seg_id]
    ds_sel = ds.sel(time=slice(seg["start"], seg["end"]))
    
    lon = ds_sel['lon'].values
    lat = ds_sel['lat'].values
    is_land_vector = np.vectorize(is_land)
    sea_mask = ~is_land_vector(lon, lat)
    
    lon = lon[sea_mask]
    lat = lat[sea_mask]
    alt = ds_sel['alt'].values[sea_mask]
    time = ds_sel['time'].values[sea_mask]
    ze = ds_sel['Ze'][sea_mask]
    height = ds_sel['height']

    height_above_0 = height.where(height > 0, drop=True)
    ze_filtered = ze.where(height_above_0 > 150)
    dbz_filtered = 10 * np.log10(ze_filtered)

    lowest_bin_idx = ze_filtered.notnull().argmax(dim='height')
    ze_lowest = ze_filtered.isel(height=lowest_bin_idx)
    S_surface = (ze_lowest / 9.2) ** (1 / 1.1)
    S_surface_filtered = S_surface.where(S_surface < 2)
    S_surface_filtered = S_surface_filtered.fillna(0)

    df_seg = pd.DataFrame({
        "time": time,
        "lat": lat,
        "lon": lon,
        "snow_rate_mmhr": S_surface_filtered.values
    })

    processed_segment_data = {
        "time": time,
        "lon": lon,
        "lat": lat,
        "alt": alt,
        "height_above_0": height_above_0,
        "dbz_filtered": dbz_filtered,
        "S_surface_filtered": S_surface_filtered
    }
    return df_seg, processed_segment_data

def segment_and_match_flight_with_era5(df_flight, df_era5, samples_per_segment): # -------4--------------
    """
    Segments the flight data and matches it to the ERA5 grid.
    """
    df_flight = df_flight.copy()
    df_e = df_era5.copy()
    df_flight['segment_id'] = df_flight.index // samples_per_segment

    def get_segment_stats(segment):
        if segment.empty:
            return pd.Series({'snow_rate_mean': np.nan, 'snow_rate_std': np.nan, 'lat': np.nan, 'lon': np.nan, 'time': pd.NaT, 'segment_id': np.nan})
        center_idx = len(segment) // 2
        center = segment.iloc[center_idx]
        return pd.Series({
            'snow_rate_mean': segment['snow_rate_mmhr'].mean(),
            'snow_rate_std': segment['snow_rate_mmhr'].std(),
            'lat': center['lat'],
            'lon': center['lon'],
            'time': center['time'],
            'segment_id': segment['segment_id'].iloc[0]
        })

    # The group-by is on the new 'segment_id' created above
    seg_flight = df_flight.groupby('segment_id').apply(get_segment_stats).reset_index(drop=True)
    seg_flight.dropna(inplace=True)
    
    seg_flight['lat_near'] = (seg_flight['lat'] / 0.25).round() * 0.25
    seg_flight['lon_near'] = (seg_flight['lon'] / 0.25).round() * 0.25
    seg_flight['time_near'] = seg_flight['time'].dt.ceil('h')
    
    final_df = pd.merge(
        seg_flight,
        df_e,
        left_on=['lat_near', 'lon_near', 'time_near'],
        right_on=['latitude', 'longitude', 'valid_time'],
        how='inner'
    )
    return final_df, df_flight

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

def process_flight_segments_individually(flight_id, segment_ids, samples_per_segment=300):
    """
    Main function to process each flight segment individually.
    """
    print(f"Processing flight: {flight_id} with segments: {segment_ids}")
    
    # Load ERA5 data once for the flight day
    era5_data_loaded = False
    ds_era5 = None
    df_era5 = None

    # Load the MiRAC-A data once
    campaign, platform, rf = flight_id.split('_')
    ds = cat[campaign][platform]['MiRAC-A'][flight_id]().to_dask()

    total_data = {}

    for seg_id in segment_ids:
        print(f"\n--- Processing segment: {seg_id} ---")
        
        # function to process a single segment -------#2--------
        df_raw_flight, processed_segment_data = process_single_segment_data(ds=ds,seg_id=seg_id, meta=meta)

        if df_raw_flight is None or df_raw_flight.empty:
            print(f"No flight data found for segment {seg_id}.")
            continue # move to next segment

        if not era5_data_loaded:
            first_segment_time = df_raw_flight['time'].min()
            # ---------------------#3--------------------
            ds_era5 = get_era5_data_for_date(first_segment_time)
            if ds_era5 is None:
                return
            df_era5 = ds_era5.to_dataframe().reset_index()
            df_era5['tp'] = df_era5['tp'] * 1000
            era5_data_loaded = True
        
        # ------------------------#4--------------------
        
        final_df, df_with_id = segment_and_match_flight_with_era5(
            df_raw_flight,
            df_era5,
            samples_per_segment=samples_per_segment
        )

        if final_df.empty:
            print(f"No ERA5 data found to match with segment {seg_id}.")
            continue

        # append all the final_df for every segemnt into total_data dictionary
        total_data[seg_id] = final_df   

        # Pass the single-segment data to the plotting function -------------------5------------------
        plot_analysis(flight_id, seg_id, {seg_id: processed_segment_data}, df_with_id, final_df, ds_era5)

    return total_data


campaign = "AFLUX"
platform = "P5"

meta = ac3airborne.get_flight_segments()
cat = ac3airborne.get_intake_catalog()


def select_segments(meta, campaign, platform, kind="high_level"):
    """
    Build a dictionary of flights and their selected segment IDs.
    kind: 'high_level', 'low_level', 'major_ascent', etc.
    """
    flight_hs_dict = {}
    flights_with_data = list(cat[campaign][platform]['MiRAC-A'].keys())

    for flight_id in flights_with_data:
        segments = meta[campaign][platform][flight_id].get('segments', [])
        high_level_segments = [
            seg['segment_id'] for seg in segments
            if kind in seg.get('kinds', [])
            and seg.get('start') is not None
            and seg.get('end') is not None ]
        
        if high_level_segments:
            flight_hs_dict[flight_id] = high_level_segments
        else:
            print(f"{flight_id} has no valid {kind} segments")
    return flight_hs_dict


def run_campaign(meta, cat, campaign, platform, kind="high_level", out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    
    flight_segments_dict = select_segments(meta, campaign, platform, kind=kind)

    for flight_id, seg_ids in flight_segments_dict.items():
        
        if not seg_ids:
            print(f"Skipping {flight_id} (no {kind} segments)")
            continue
        
        print(f"\nProcessing {flight_id} with {len(seg_ids)} {kind} segments")
        
        total_final_data = process_flight_segments_individually(
            flight_id=flight_id,
            segment_ids=seg_ids
        )
        
        if not total_final_data:
            print(f"No results for {flight_id}")
            continue
        
        
        df_all = pd.concat(total_final_data.values(), ignore_index=True)
        csv_path = os.path.join(out_dir, f"{flight_id}_{kind}.csv")
        df_all.to_csv(csv_path, index=False)
        print(f"Saved results for {flight_id} to {csv_path}")


if __name__ == "__main__":
    run_campaign(meta, cat, campaign, platform, kind="high_level", out_dir="outputs")
