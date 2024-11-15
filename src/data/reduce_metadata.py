"""
This script reads the very large yt_metadata_en JSONLines file and splits it into smaller CSV files (chunk) in order to make the data more manageable.
Each CSV file contains a chunk of the data, with a specified number of records per chunk (10_000_000 for chunks 0 to 5 and 12_924_794 for last chunk).
The script uses pandas to read the JSONL file, process the data, and save each chunk as a CSV file. It also create a dictionary with the mapping of the channel to the chunk(s) containing its records.
"""

import pandas as pd
import json
import os

from preprocessing import map_column_to_week

# Function to process each chunk and save it as a CSV
def save_as_chunk(file_path, start_line, end_line, chunk_index, columns):
    """
    Save a chunk of the data as a CSV file
    
    Parameters:
    file_path (str): Path to the file
    start_line (int): Start line of the chunk
    end_line (int): End line of the chunk
    chunk_index (int): Index of the chunk
    columns (list): List of columns to keep
    """
    processed_data = []
    with open(file_path, 'r') as file:
        # Skip to the start line
        for _ in range(start_line):
            file.readline()

        # Process the chunk
        for i in range(start_line, end_line):
            line = file.readline()
            if not line:
                break  # End of file
            try:
                record = json.loads(line)
                processed_data.append({col: record[col] for col in columns})
            except json.JSONDecodeError:
                print(f"Skipping malformed line at index {i}")

    # Convert the chunk to a DataFrame
    df_chunk = pd.DataFrame(processed_data)

    # Write to CSV and clear the DataFrame from memory
    output_path = os.path.join(output_folder, f'chunk_{chunk_index}.csv')
    df_chunk.to_csv(output_path, index=False)
    print(f"Saved chunk {chunk_index} to {output_path}")
    del df_chunk  # Free up memory

def process_chunk(file_path, chunk_index):
    """
    Process chunks to get first and last week

    Parameters:
    file_path (str): Path to the file
    """
    # Load chunk
    print(f"Loading chunk {chunk_index} ...")
    chunk = pd.read_csv(file_path)
    print(f"Loading complete for chunk {chunk_index}")

    print(f"Processing chunk {chunk_index} ...")

    # Change column name from channel_id to channel
    chunk.rename(columns={'channel_id': 'channel'}, inplace=True)

    # Drop all rows with missing values (data cleaning)
    chunk = chunk.dropna()

    # Convert date to week
    chunk['upload_date'] = pd.to_datetime(chunk['upload_date'])
    chunk = map_column_to_week(chunk, 'upload_date')

    # Set index to (week, channel)
    chunk.set_index(['channel', 'week'], inplace=True)
    chunk.sort_index(inplace=True)

    print(f"Processing complete for chunk {chunk_index}")

    print(f"Saving processed chunk {chunk_index} ...")

    contained_channels = chunk.index.get_level_values('channel').unique()
    print(f"Chunk {chunk_index} contains {len(contained_channels)} channels")

    # Write first and last week to file name
    output_path = file_path.replace('.csv', f'_preprocessed.csv')
    chunk.to_csv(output_path, index=False)
    print(f"Saved chunk {chunk_index} to {output_path}")

    return contained_channels

if __name__ == '__main__':
    # Set up variables
    file_path = 'yt_metadata_en.jsonl'
    chunk_size = 10_000_000
    columns = ['channel_id', 'upload_date', 'title', 'description']
    output_folder = 'chunks'

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the file and process each chunk (0 to 5)
    chunk_index = 0
    start_line = 0

    for chunk_index in range(0, 6):
        end_line = start_line + chunk_size
        save_as_chunk(file_path, start_line, end_line, chunk_index, columns)
        start_line = end_line

    # Process the last chunk (all remaining values)
    chunk_index = 6
    start_line = 60_000_001
    chunk_size = 12_924_794

    end_line = start_line + chunk_size
    save_as_chunk(file_path, start_line, end_line, chunk_index, columns) 

    # Create a dictionary for the chunk index for each channel
    channel_chunk_dict = {}

    # Perform the preprocessing on all chunks
    for i in range(0, 7):
        contained_channels = process_chunk(f'../../data/chunks/chunk_{i}.csv', i)
        # Update the dictionary with the new chunk index for each channel
        for channel in contained_channels:
            if channel in channel_chunk_dict:
                # Handle the case where a channel is present in multiple chunks
                print(f"Channel {channel} is present in multiple chunks")
                # Special index for channels present in multiple chunks
                # E.g. 10 for a channel present in chunk 1 and 0, 21 for a channel present in chunk 2 and 1, 32 for a channel present in chunk 3 and 2 etc...
                special_index = i*10 + (i-1)
                channel_chunk_dict.update({channel: special_index})
            else :
                channel_chunk_dict.update({channel: i})

    # Save the dictionary to a file
    with open('../../data/channel_chunk_dict.json', 'w') as file:
        json.dump(channel_chunk_dict, file)
        
    print("Dictionary saved to file")