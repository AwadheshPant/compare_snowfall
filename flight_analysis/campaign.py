import os
import pandas as pd
import ac3airborne
from .processing import process_single_segment_data
from .io_utils import get_era5_data_for_date
from .matching import segment_and_match_flight_with_era5
from .plotting import plot_analysis

def select_segments(meta, campaign, platform, kind="high_level"):
    flight_hs_dict = {}
    flights_with_data = list(ac3airborne.get_intake_catalog()[campaign][platform]['MiRAC-A'].keys())

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

def process_flight_segments_individually(flight_id, segment_ids, samples_per_segment=300):
    print(f"Processing flight: {flight_id} with segments: {segment_ids}")
    
    era5_data_loaded = False
    ds_era5 = None
    df_era5 = None

    campaign, platform, rf = flight_id.split('_')
    cat = ac3airborne.get_intake_catalog()
    ds = cat[campaign][platform]['MiRAC-A'][flight_id]().to_dask()

    total_data = {}

    for seg_id in segment_ids:
        print(f"\n--- Processing segment: {seg_id} ---")
        
        df_raw_flight, processed_segment_data = process_single_segment_data(ds=ds, seg_id=seg_id, meta=ac3airborne.get_flight_segments())
        if df_raw_flight is None or df_raw_flight.empty:
            print(f"No flight data found for segment {seg_id}.")
            continue

        if not era5_data_loaded:
            first_segment_time = df_raw_flight['time'].min()
            ds_era5 = get_era5_data_for_date(first_segment_time)
            if ds_era5 is None:
                return
            df_era5 = ds_era5.to_dataframe().reset_index()
            df_era5['tp'] = df_era5['tp'] * 1000
            era5_data_loaded = True
        
        final_df, df_with_id = segment_and_match_flight_with_era5(
            df_raw_flight,
            df_era5,
            samples_per_segment=samples_per_segment
        )

        if final_df.empty:
            print(f"No ERA5 data found to match with segment {seg_id}.")
            continue

        total_data[seg_id] = final_df   
        plot_analysis(flight_id, seg_id, {seg_id: processed_segment_data}, df_with_id, final_df, ds_era5)

    return total_data

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
