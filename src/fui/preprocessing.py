# -*- coding: utf-8 -*-
import os
import pandas as pd
import pickle
import h5py
import glob
import html
from src.fui.utils import params
from src.fui.sql_insert import import_csv

def parse_for_lda(nrows=None):
    """
    Loads the data from CSV and performs some basic cleaning. Essentially the
    cleaning removes corrupted lines.
    """
    # Load the data
    df, _ = import_csv()

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df = df[df['year'] > 1999]

    #drop irrelevant section names
    start_n = df.shape[0] 
    df = df[~df['section_name'].isin(['Biler', 'Bolig-besøg', 'Bolig-indret','Bolig-items','Cannes','Cannes Lions',
                                     'Design', 'Digital Gadget', 'Gastronomi', 'Karriere', 'Kriminal', 'Kultur',
                                     'Livsstil', 'Magasin Pleasure', 'Magasin Pleasure 2. sektion', 'Magasin Pleasure 2. sektion Rejser',
                                     'Magasin Pleasure 2. sektion Rejser Hoteller Stil', 'Magasin Pleasure Biler',
                                     'Magasin Pleasure Design', 'Magasin Pleasure EM', 'Magasin Pleasure Firmabilen 2015',
                                     'Magasin Pleasure Interiør', 'Magasin Pleasure kunst & kultur', 'Magasin Pleasure Portræt',
                                     'Magasin Pleasure rejser', 'Magasin Pleasure Ure', 'Michelin', 'Motion', 'Play 2016',
                                     'Pleasure', 'Portræt', 'Profil & Karriere', 'Underholdning', 'Week-div', 'Week-golf',
                                     'Week-livsstil', 'Week-mad', 'Week-maritim', 'Week-mode', 'Week-motor', 'Week-rejser',
                                     'Weekend Diverse', 'Weekend Golf','Weekend Kultur','Weekend Livsstil',
                                     'Weekend Livstil','Weekend Mad','Weekend Maritim','Weekend Mode','Weekend Motor',
                                     'Weekend Outdoor', 'Weekend Rejser'])]
    end_n = df.shape[0]
    print('Dropped {} articles with irrelevant section names'.format(start_n-end_n))
    print(f'Current number of articles: {end_n}')

    #drop word count below 50
    df['word_count'] = df['body'].str.count(' ') + 1
    start_n = df.shape[0]
    df = df[df.word_count >= 50]
    end_n = df.shape[0]
    print('Dropped {} articles with less than 50 words'.format(start_n-end_n))
    print(f'Current number of articles: {end_n}')

    df['body'] = __clean_text(df['body'])

    # create unique row index
    df['article_id'] = df.reset_index().index
    print('Columns: ', df.columns)
    
    with h5py.File(os.path.join(params().paths['parsed_news'],params().filenames['parsed_news']), 'w') as hf:
        string_dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('parsed_strings', data=df, dtype=string_dt)

def __clean_text(series):
    """
    ProcessTools: Clean raw text
    """
    # Step 1: Remove html tags
    print('Step 1: Removing tags')
    series = series.str.replace(r'<[^>]*>','',regex=True)
    series = series.str.replace(r'https?:\/\/\S*','',regex=True)
    series = series.str.replace(r'www.\S*','',regex=True)
        
    # Step 2: Remove everything following these patterns (bylines)     
    print('Step 2: Removing bylines')
    pattern = "|".join(
        [r'\/ritzau/AFP', r'Nyhedsbureauet Direkt', r'\w*\@borsen.dk\b', r'\/ritzau/', r'\/ritzau/FINANS'])
    series = series.str.replace(pattern,'',regex=True)

    # Step 3: Remove \n, \t
    print('Step 3: Removing other')
    series = series.str.replace(r'\n', ' ')
    series = series.str.replace(r'\t', ' ')

    # Step 4: Remove additional whitespaces
    #series = series.str.split().str.join(' ')

    # Step 5: Convert to lowercase
    series = series.str.lower()
    
    # Step 6: Unescape any html entities
    print('Step 6: Unescape html')
    series = series.apply(html.unescape)
    
    # Manually remove some html
    series = series.str.replace(r'&rsquo', '')
    series = series.str.replace(r'&ldquo', '')
    series = series.str.replace(r'&rdquo', '')
    series = series.str.replace(r'&ndash', '')
    
    return series

def load_parsed_data(sample_size=None):
    filelist = glob.glob(params().paths['parsed_news']+'boersen*.pkl') 
    df = pd.DataFrame()
    for f in filelist:    
        with open(f, 'rb') as f_in:
            df_n = pickle.load(f_in)
            df = df.append(df_n)
    if sample_size is not None:
        return df.sample(sample_size)
    else:
        return df
