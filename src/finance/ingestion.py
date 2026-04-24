import pandas as pd
import numpy as np

def process_tick_data(file_path, chunk_size=500000):
    """
    Processes large tick data CSV in chunks to avoid memory overflow.
    Resamples tick data into 1-minute OHLCV bars.
    """
    print(f"Starting ingestion of {file_path}...")
    bars_list = []
    
    # Process in chunks. 
    # Assumes CSV columns: date, time, price, volume
    # Adjust names based on actual file header if necessary.
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Create a unified timestamp
        chunk['timestamp'] = pd.to_datetime(chunk['date'] + ' ' + chunk['time'])
        chunk.set_index('timestamp', inplace=True)
        
        # Resample to 1-minute OHLCV bars
        ohlcv = chunk['price'].resample('1min').ohlc()
        ohlcv['volume'] = chunk['volume'].resample('1min').sum()
        
        # Drop NaN for incomplete bars within this chunk
        bars_list.append(ohlcv.dropna())
        print(f"Processed chunk of {len(chunk)} ticks.")
        
    # Combine all chunks and resample again to handle boundary overlaps
    full_data = pd.concat(bars_list)
    final_bars = full_data.resample('1min').agg({
        'open': 'first', 
        'high': 'max', 
        'low': 'min', 
        'close': 'last', 
        'volume': 'sum'
    }).dropna()
    
    print(f"Finished ingestion. Total bars: {len(final_bars)}")
    return final_bars
