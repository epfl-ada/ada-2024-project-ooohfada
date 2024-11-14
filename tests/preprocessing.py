# Some tests to check the preprocessing
if __name__ == '__main__':
    timeseries_example = pd.DataFrame({
        'datetime': ['2020-01-01', '2020-01-01', '2020-01-08', '2020-01-08'],
        'channel': ['A', 'A', 'A', 'A'],
        'category': ['Music', 'Music', 'Music', 'Music'],
        'views': [100, 200, 320, 430],
        'delta_views': [0, 0, 0, 0],
        'subs': [10, 28, 40, 38],
        'delta_subs': [0, 0, 0, 0],
        'videos': [1, 2, 3, 5],
        'delta_videos': [0, 1, 1, 2],
        'activity': [1, 2, 3, 4],
    })

    metadata_helper_example = pd.DataFrame({
        'upload_date': ['2020-01-01', '2020-01-01', '2020-01-08', '2020-01-08'],
        'channel': ['A', 'A', 'A', 'A'],
        'view_count': [100, 200, 320, 430],
        'like_count': [10, 20, 30, 40],
        'dislike_count': [1, 5, 10, 20],
        'categories': ['Music', 'Music', 'Music', 'Music'],
        'crawl_date': ['2020-01-01', '2020-01-01', '2020-01-08', '2020-01-08'],
        'display_id': ['1', '2', '3', '4'],
        'duration': [1, 2, 3, 5],
    })

    print("Before preprocessing:")
    print('\nTimeseries:')
    print(timeseries_example)
    print('\nMetadata helper:')
    print(metadata_helper_example)

    print("\nAfter preprocessing:")
    print('\nTimeseries:')
    print(apply_timeseries_preprocessing(timeseries_example))
    print('\nMetadata helper:')
    print(apply_metadata_helper_preprocessing(metadata_helper_example))
    print('\nComplete preprocessing:')
    print(apply_complete_preprocessing(timeseries_example, metadata_helper_example))
    