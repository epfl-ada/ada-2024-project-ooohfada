import pandas as pd
from src.data.preprocessing import map_column_to_week

def extract_videos_around_declines(decline_events, METADATA_FILENAME, TO_FILE):
    """
    Extract the videos that are uploaded around the decline events
    This function considers the video during the decline and between (start - duration) and (start).
    """

    # In batches of 5000 rows, extract the tags, duration, channel_id and week of the videos that are in the time frame for the respective channel
    CHUNK_SIZE = 5000

    # Get the channels that are in the decline events
    channels = decline_events['Channel'].unique()

    pd.options.mode.chained_assignment = None # remove SettingWithCopyWarning: 

    i = 0
    for chunk in pd.read_json(METADATA_FILENAME, lines=True, chunksize=CHUNK_SIZE):

        try:
            init_shape = int(chunk.shape[0])
            
            chunk = chunk[chunk['channel_id'].isin(channels)]
            
            chunk.loc[:, 'upload_date'] = pd.to_datetime(chunk['upload_date'])

            chunk = map_column_to_week(chunk, 'upload_date')

            # keep the videos that are in the time frame for the respective channel
            mask = []
            for video in chunk.itertuples():
                channel_mask = decline_events['Channel'].isin([video.channel_id])
                start_mask = decline_events['Start'] - decline_events['Duration'] <= video.week
                end_mask = decline_events['End'] >= video.week

                mask.append(decline_events[channel_mask & start_mask & end_mask].shape[0] > 0)

            kept = chunk[mask]

            if kept.shape[0] == 0:
                print(f'Chunk {i} (lines {i*CHUNK_SIZE} to {(i+1)*CHUNK_SIZE}): 0/{init_shape} videos kept')
                i += 1
                continue

            cols_of_interest = ['channel_id', 'week', 'tags', 'duration']

            kept = kept[cols_of_interest]

            print(f'Chunk {i} (lines {i*CHUNK_SIZE} to {(i+1)*CHUNK_SIZE}): {kept.shape[0]}/{init_shape} videos kept')

            kept.to_csv(TO_FILE, index=False, mode='a', header=False) # Temporary filename

            i += 1
        
        except Exception as e:
            print(f'Error in chunk {i} (lines {i*CHUNK_SIZE} to {(i+1)*CHUNK_SIZE}): {e}')