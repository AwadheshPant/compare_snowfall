import numpy as np
import pandas as pd
from ac3airborne.tools.is_land import is_land

def process_single_segment_data(ds, seg_id, meta):
    """
    Processes a single flight segment and returns processed data.
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
