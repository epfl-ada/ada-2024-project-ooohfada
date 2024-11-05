import pandas as pd
import numpy as np


def load_timeseries_data(usecols = None):
    """Load the initial Timeseries df

    Args:
        usecols (df_colulns, optional): Selection of wanted columns to load. Defaults to None.

    Returns:
        df: the Timeseries df
    """    
    return pd.read_csv('data/df_timeseries_en.tsv', sep='\t', compression = 'infer', usecols = usecols)

def timeseries_bb_selection():
    """Keep only the bad buzz, determined using various documentation, see README with Addition of the literal name of the channel
    """    
    #Load the entirety of the timeserie data
    df_timeseries_en = load_timeseries_data()

    badbuzz_channels = ['UCEHf6KUY7Zw7hlXQ7hDemwQ', #tmartn
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
    df_bb_timeseries_en = df_timeseries_en[df_timeseries_en['channel'].isin(badbuzz_channels)]

    #Create a new column with the name of the Youtuber for simplification purpose during analysis

    channel_map = {
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

    #Mapping between name and id of channels
    df_bb_timeseries_en['channel_name'] = df_bb_timeseries_en['channel'].map(channel_map)

    # Save into a tsv file the 'bad buzz' dataset
    df_bb_timeseries_en.to_csv('data/df_bb_timeseries_en.tsv', sep='\t', index=False)


def load_bb_timeseries_data(usecols = None):
    """Load the bad buzz df 

    Args:
        usecols (df_columns, optional):  Selection of wanted columns to load. Defaults to None.

    Returns:
        df: the Timeseries df
    """
    return pd.read_csv('data/df_bb_timeseries_en.tsv', sep='\t', compression = 'infer', usecols = usecols)


timeseries_bb_selection()