import os
import xarray as xr
import pandas as pd
from datetime import timedelta

ERA5_DATA_DIR = "/work/apant/carra/project_root/data/raw/era5/2018_19"

def get_era5_data_for_date(flight_date):
    """
    Load ERA5 data for a given date based on time taken from first flight segment.
    """
    era5_filename = "f949ed754fc9dda78f2bf0a0ef6bb70f.nc"  # replace

    era5_path = os.path.join(ERA5_DATA_DIR, era5_filename)
    
    if not os.path.exists(era5_path):
        print(f"ERA5 file not found: {era5_path}. Please check the file path.")
        return None
    
    ds = xr.open_dataset(era5_path)
    start_time = pd.to_datetime(flight_date).floor('D')
    end_time = start_time + timedelta(days=1)
    ds_sel = ds.sel(valid_time=slice(start_time, end_time))
    return ds_sel
