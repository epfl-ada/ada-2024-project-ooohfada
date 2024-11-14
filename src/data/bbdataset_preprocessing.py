import numpy as np
from tqdm import tqdm
from time import time
from dataloader import load_timeseries, load_metadata_helper, load, DATA_PATH
from preprocessing import apply_complete_preprocessing
import os.path

# To get the path to data/ regardless of where this script is called from :
PROCESSED_BAD_BUZZ_PATH = os.path.join(DATA_PATH, 'df_bb_data_en_processed.tsv')

# The number of rows to load at once
CHUNK_SIZE = 1000

 # List of the bad buzz channels
BAD_BUZZ_CHANNELS = ['UCEHf6KUY7Zw7hlXQ7hDemwQ', #tmartn
                        'UCnEn0EUV13IR-_TK7fiIp3g', #AlfieDeyes
                        'UCV9_KinVpV-snHe3C3n1hvA', #ShaneDawson
                        'UClWD8su9Sk6GzZDwy9zs3_w', #TanaMongeau
                        'UCKGiTasUqLcZUuUjQiyKotw', #SamPepper
                        'UC0v-tlzsn0QZwJnkiaUSJVQ', #TheFineBros
                        'UCX6OQ3DkcsbYNE6H8uQQuVA', #LoganPaul
                        'UC-lHJZR3Gqxm24_Vd_AJ5Yw', #PewDiePie
                        'UC1Q9JKC7S8GzYdtp4vkqkcA', #VitalyZdoorovetskiy
                        'UCl0MsYaBUplB7VwEry901og', #KeemStar (Drama Alert)
                        'UC4sEmXUuWIFlxRIFBRV6VXQ', #ContentCop
                        'UCoiIt_v1D-6z75LmrdIU2aw', #NikocadoAvocado
                        'UCm1hYxe9Ztst7Dzh3W_1flg', #Ricegum
                        'UCucot-Zp428OwkyRm2I7v2Q', #JamesCharles
                        'UCu2rmC8OXrf-R5DdLG68dKw', #JeffreeStar
                        'UCVtFOytbRpEvzLjvqGG5gxQ', #KSI
                        'UCdJdEguB1F1CiYe7OEi3SBg', #JonTronShow
                        'UC7Tq0KZSgtwCh6NcrfnRHvQ', #NicoleArbour
                        'UC4qk9TtGhBKCkoWz5qGJcGg', #TatiWestbrook
                        'UCVJK2AT3ea5RTXNRjX_kz8A', #TobyTurner
                        'UCXhSCMRRPyxSoyLSPFxK7VA', #MatthewSantoro
                        'UC8lV8KIVWvfsaqOi_d3Wu3w', #DaddyOFive #deleted
                        'UCzKc6JrWSt_67UpEYIefrJQ', #MarinaJoyce
                        'UCxJf49T4iTO_jtzWX3rW_jg', #LeafyIsHere #deleted
                        'UC2e0bNZ6CzT-Xvr070VaGsw', #ProJared
                        'UC_DptbqTndVt_Im3KkuIK5Q', #KianAndJC
                        'UCAq9s3QQVCDMvg1iWQBVtxQ', #SamandNia
                        'UCg5rY7_sfwepQJ5Fg1VmZPA', #AustinJones
                        'UC1r4VtVE__5K6c_L_3Vlxxg', #FouseyTube
                        'UCy_YiQx1t8oOgz74QIB4Jrw', #Myka Stauffer
                        'UCtVubfONoPpn4kNuuZ1h6iQ', #EugeniaConey
                        'UCcgVECVN4OKV6DH1jLkqmcA', #JakePaul
                        'UCiH828EtgQjTyNIMH6YiOSw', #ChannelAwesome
                        'UCDsO-0Yo5zpJk575nKXgMVA', #RocketJump
                        'UCKlhpmbHGxBE6uw9B_uLeqQ', #SkyDoesMinecraft
                        'UCBHu7LsKiwiYViR230RtsCA', #JoeySalads
                        'UCdoLeDxfcGwvj_PRl7TLTzQ', #Onision
                        'UCJZ7f6NQzGKZnFXzFW9y9UQ', #Shaytards
                        'UC9fUm_9ZouDuLIMlml6bw5w', #ToyFreaks (one of them)
                        'UC6-NBhOCP8DJqnpZE4TNE-A', #LanceStewart
                        'UCWwWOFsW68TqXE-HZLC3WIA', #TheACEFamily
                        'UCKMugoa0uHpjUuq14yOpagw', #LauraLee
                        'UCZ__vn_T9SK44jcM85rnt4A', #PrankInvasion
                        'UCzJIliq68IHSn-Kwgjeg2AQ', #JackieAina
                        'UCFIHxULKUBYdU7ZkZh_5p1g', #N&AProductions
                        'UCj2HtBTppiQLVrZfEjcFxig', #SevenSuperGirls
                        'UC6hUgRphnVoS7x-XkErFZxQ', #InvisibleChildren
                        ] 

# To be used in notebooks
CHANNEL_NAMES = {
                        'UCEHf6KUY7Zw7hlXQ7hDemwQ': 'tmartn',
                        'UCnEn0EUV13IR-_TK7fiIp3g' : 'AlfieDeyes',
                        'UCV9_KinVpV-snHe3C3n1hvA' : 'ShaneDawson', 
                        'UClWD8su9Sk6GzZDwy9zs3_w' : 'TanaMongeau',
                        'UCKGiTasUqLcZUuUjQiyKotw' : 'SamPepper',
                        'UC0v-tlzsn0QZwJnkiaUSJVQ' : 'TheFineBros',
                        'UCX6OQ3DkcsbYNE6H8uQQuVA' : 'LoganPaul',
                        'UC-lHJZR3Gqxm24_Vd_AJ5Yw' : 'PewDiePie',
                        'UC1Q9JKC7S8GzYdtp4vkqkcA' : 'VitalyZdoorovetskiy',
                        'UCl0MsYaBUplB7VwEry901og' : 'KeemStar (Drama Alert)',
                        'UC4sEmXUuWIFlxRIFBRV6VXQ' : 'ContentCop',
                        'UCoiIt_v1D-6z75LmrdIU2aw' : 'NikocadoAvocado',
                        'UCm1hYxe9Ztst7Dzh3W_1flg' : 'Ricegum',
                        'UCucot-Zp428OwkyRm2I7v2Q' : 'JamesCharles',
                        'UCu2rmC8OXrf-R5DdLG68dKw' : 'JeffreeStar',
                        'UCVtFOytbRpEvzLjvqGG5gxQ' : 'KSI',
                        'UCdJdEguB1F1CiYe7OEi3SBg' : 'JonTronShow',
                        'UC7Tq0KZSgtwCh6NcrfnRHvQ' : 'NicoleArbour',
                        'UC4qk9TtGhBKCkoWz5qGJcGg' : 'TatiWestbrook',
                        'UCVJK2AT3ea5RTXNRjX_kz8A' : 'TobyTurner',
                        'UCXhSCMRRPyxSoyLSPFxK7VA' : 'MatthewSantoro',
                        'UC8lV8KIVWvfsaqOi_d3Wu3w' : 'DaddyOFive', #deleted
                        'UCzKc6JrWSt_67UpEYIefrJQ' : 'MarinaJoyce',
                        'UCxJf49T4iTO_jtzWX3rW_jg' : 'LeafyIsHere', #deleted
                        'UC2e0bNZ6CzT-Xvr070VaGsw' : 'ProJared',
                        'UC_DptbqTndVt_Im3KkuIK5Q' : 'KianAndJC',
                        'UCAq9s3QQVCDMvg1iWQBVtxQ' : 'SamandNia',
                        'UCg5rY7_sfwepQJ5Fg1VmZPA' : 'AustinJones',
                        'UC1r4VtVE__5K6c_L_3Vlxxg' : 'FouseyTube',
                        'UCy_YiQx1t8oOgz74QIB4Jrw' : 'Myka Stauffer',
                        'UCtVubfONoPpn4kNuuZ1h6iQ' : 'EugeniaConey',
                        'UCcgVECVN4OKV6DH1jLkqmcA' : 'JakePaul',
                        'UCiH828EtgQjTyNIMH6YiOSw' : 'ChannelAwesome',
                        'UCDsO-0Yo5zpJk575nKXgMVA' : 'RocketJump',
                        'UCKlhpmbHGxBE6uw9B_uLeqQ' : 'SkyDoesMinecraft',
                        'UCBHu7LsKiwiYViR230RtsCA' : 'JoeySalads',
                        'UCdoLeDxfcGwvj_PRl7TLTzQ' : 'Onision',
                        'UCJZ7f6NQzGKZnFXzFW9y9UQ' : 'Shaytards',
                        'UC9fUm_9ZouDuLIMlml6bw5w' : 'ToyFreaks', #(one of them)
                        'UC6-NBhOCP8DJqnpZE4TNE-A' : 'LanceStewart',
                        'UCWwWOFsW68TqXE-HZLC3WIA' : 'TheACEFamily', 
                        'UCKMugoa0uHpjUuq14yOpagw' : 'LauraLee', 
                        'UCZ__vn_T9SK44jcM85rnt4A' : 'PrankInvasion',
                        'UCzJIliq68IHSn-Kwgjeg2AQ' : 'JackieAina', 
                        'UCFIHxULKUBYdU7ZkZh_5p1g' : 'N&AProductions',
                        'UCj2HtBTppiQLVrZfEjcFxig' : 'SevenSuperGirls',
                        'UC6hUgRphnVoS7x-XkErFZxQ' : 'InvisibleChildren'
    }

def update_processed_bb_timeseries(verbose = False):
    """
    Update the processed bad buzz time series data by filtering the bad buzz channels and processing the data,
    saving the result in 'df_bb_timeseries_en_processed.tsv'

    Parameters:
    verbose (bool): whether to print the progress
    """
    start_time = time()

    #Load the entirety of the timeserie data
    df_timeseries_en = load_timeseries(verbose=verbose)
    df_metadata_helper = load_metadata_helper(verbose=verbose)
    
    if verbose:
        print(f'Filtering...', end='\r')

    # Filter the bad buzz channels
    df_bb_timeseries_en = df_timeseries_en[df_timeseries_en['channel'].isin(BAD_BUZZ_CHANNELS)]
    df_bb_metadata_helper = df_metadata_helper[df_metadata_helper['channel_id'].isin(BAD_BUZZ_CHANNELS)]

    if verbose:
        print(f'Preprocessing...', end='\r')

    # Apply the preprocessing
    df_bb_timeseries_en = apply_complete_preprocessing(df_bb_timeseries_en, df_bb_metadata_helper)

    if verbose:
        print('Filtering and preprocessing done:')
        print(df_bb_timeseries_en.head())

        # Save the processed data
    if verbose:
        chunks = np.array_split(df_bb_timeseries_en.index, 100) # split into 100 chunks
        for chunck, subset in enumerate(tqdm(chunks, desc='Saving data', total=len(chunks))):
            if chunck == 0: # first row
                df_bb_timeseries_en.loc[subset].to_csv(PROCESSED_BAD_BUZZ_PATH, mode='w', index=True, sep='\t')
            else:
                df_bb_timeseries_en.loc[subset].to_csv(PROCESSED_BAD_BUZZ_PATH, header=None, mode='a', index=True, sep='\t')
    else:
        df_bb_timeseries_en.to_csv(PROCESSED_BAD_BUZZ_PATH, sep='\t', index=True)

    if verbose:
        duration = time() - start_time
        print(f'Processed bad buzz time series data updated in \'{PROCESSED_BAD_BUZZ_PATH}\' in {duration:.2f}s')

def load_bb_timeseries_processed(usecols = None, nrows = None, verbose = False):
    """
    Load the bad buzz df, preprocessed

    Args:
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress

    Returns:
    df: the Timeseries df
    """
    return load(PROCESSED_BAD_BUZZ_PATH, usecols=usecols, nrows=nrows, index_col=['channel', 'week'], verbose=verbose)

if __name__ == '__main__':
    update_processed_bb_timeseries(verbose=True)
    print(load_bb_timeseries_processed(verbose=True).head())