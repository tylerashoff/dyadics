import pandas as pd
import numpy as np


def clean_icpsr():
    '''
    initial cleaning of the dataset
    make naming of datasets consistent
    '''

    from dictionary_keys import targets, sources

    df_og = pd.read_csv(
        '../data/icpsr.csv', encoding='latin1', names=['rows'], sep='\n')

    df_og.rows.str[0:2]
    df = pd.DataFrame({
        'year': ('19' + df_og.rows.str[0:2]).astype(int),
        'month': df_og.rows.str[2:4].astype(int),
        'day': df_og.rows.str[4:6].astype(int),
        'actor_num': df_og.rows.str[6:9],
        'target_num': df_og.rows.str[9:12],
        'source_num': df_og.rows.str[12:14],
        'activity': df_og.rows.str[14:26],
        'issue1': df_og.rows.str[30:31],
        'issue2': df_og.rows.str[31:32],
        'scale': df_og.rows.str[32:34].astype(int),
        'issue_area': df_og.rows.str[34:]
    })
    df.insert(0, 'date',
              pd.to_datetime(df[['year', 'month', 'day']], format='%%%'))

    df.insert(5, 'actor',
              [targets[key.replace(' ', '0')] for key in df['actor_num']])
    df.insert(7, 'target',
              [targets[key.replace(' ', '0')] for key in df['target_num']])
    df.insert(9, 'source',
              [sources[key.replace(' ', '0')] for key in df['source_num']])

    print('df created')

    df.to_csv('../data/icpsr.csv', index=False)

    print('written')
    return (None)


#clean_icpsr()
