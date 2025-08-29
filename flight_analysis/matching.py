import numpy as np
import pandas as pd

def segment_and_match_flight_with_era5(df_flight, df_era5, samples_per_segment):
    """
    Segments the flight data and matches it to the ERA5 grid.
    """
    df_flight = df_flight.copy()
    df_e = df_era5.copy()
    df_flight['segment_id'] = df_flight.index // samples_per_segment

    def get_segment_stats(segment):
        if segment.empty:
            return pd.Series({'snow_rate_mean': np.nan, 'snow_rate_std': np.nan,
                              'lat': np.nan, 'lon': np.nan, 'time': pd.NaT,
                              'segment_id': np.nan})
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
