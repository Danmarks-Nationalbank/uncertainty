# -*- coding: utf-8 -*-

import os
import pandas as pd
import pickle
import h5py
import glob
import html
from fui.utils import params

def parse_raw_data(nrows=None):
    """
    Loads the data from CSV and performs some basic cleaning. Essentially the
    cleaning removes corrupted lines.
    """
    # Load the data
    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv']
    df = pd.read_csv(csvpath, sep=';', encoding='UTF-16', error_bad_lines=False, nrows=nrows)
    
    print('Dropping articles with NaN content...')
    start_n = df.shape[0] 
    # df = df[df['Title']!='test']
    df = df[df['ArticleContents'].apply(type) == str]
    end_n = df.shape[0]
    print('Dropped {} articles with NaN content'.format(start_n-end_n))

    print('Dropping articles with SectionName not being string...')
    start_n = df.shape[0] 
    df = df[df['SectionName'].apply(type) == str]
    df = df[df['SectionName'].str.isnumeric() == False]
    end_n = df.shape[0]
    print('Dropped {} articles with ID as SectionName'.format(start_n-end_n))

    df['ArticleDateCreated'] = pd.to_datetime(df['ArticleDateCreated'], format='%Y-%m-%d')
    df['Year'] = df['ArticleDateCreated'].dt.year
    df = df[df['Year'] > 1999]
    
    #drop some suppliers
    start_n = df.shape[0] 
    df = df[df['Supplier'].isin(['Blog', 'Ritzau', 'Børsen', 'E-Avis', 'ePaper', 'Borsen'])]
    end_n = df.shape[0]
    print('Dropped {} articles with bad suppliers'.format(start_n-end_n))

    print(f'Current number of articles: {end_n}')
    #drop irrelevant section names
    start_n = df.shape[0] 
    df = df[~df['SectionName'].isin(['Bagsiden','Hvad kan vi danskere','Pleasure',
                                     'Weekend Diverse','Bilen','Weekend Kultur',
                                     'Bagside','Underholdning','Weekend Livsstil',
                                     'Sponsoreret','Unknown Section Name','Kriminal',
                                     'Portræt','Gadget','Magasin Pleasure','Gourmet',
                                     'Rejser','Weekend Mode','Week-rejser','Weekend Golf',
                                     'Week-mad', 'Week-motor', 'Biler', 'Kultur', 'Sport',
                                     'Week-livsstil', 'Weekend Mad', 'Pleasure', 'Design',
                                     'Motion', 'Week-golf', 'Week-mode','Roskilde festival',
                                     'Magasin Pleasure EM', 'Weekend Outdoor', 'Magasin Pleasure Portræt',
                                     'Magasin Pleasure Ure', 'Magasin Pleasure Design',
                                     'Magasin Pleasure Interiør', 'Magasin Pleasure kunst & kultur',
                                     'Magasin Pleasure 2. sektion Rejser', 'Magasin Pleasure Firmabilen 2015',
                                     'Magasin Pleasure rejser', 'Magasin Pleasure 2. sektion Rejser Hoteller Stil',
                                     'Weekend Livstil','Digital Gadget','Magasin Business Firmabilen'])]
    end_n = df.shape[0]
    print('Dropped {} articles with irrelevant section names'.format(start_n-end_n))

    start_n = df.shape[0]
    df = df[df['ShowOnWebSite'].apply(type) == str]
    df = df[df['ShowOnWebSite'] == 'Yes']
    end_n = df.shape[0]
    print('Dropped {} articles not shown on website'.format(start_n-end_n))
    print(f'Current number of articles: {end_n}')
    
    #drop unneccessary columns
    df.drop(labels=[
            'ShowOnWebSite', 
            'SAXoInternalId',
            'SAXoSectionPageNumber', 
            'SAXoCategory', 
            'SAXoByline',
            'SAXoHeadline'], inplace=True, axis=1)
    
    #create unique row index
    df['article_id'] = df.reset_index().index
    
    df['ArticleContents'] = __clean_text(df['ArticleContents'])
    df['word_count'] = df['ArticleContents'].str.count(' ') + 1
    df = df[df.word_count >= 50] 
    end_n = df.shape[0]
    print(f'Current number of articles: {end_n}')
    print('Columns: ', df.columns)
    df[['Title', 'ArticleContents', 'Supplier']] = df[['Title', 'ArticleContents', 'Supplier']].astype('str')
    df = df[['Title', 'ArticleContents', 'ArticleDateCreated', 'Supplier', 'article_id', 'word_count']]
    
    df.describe()
    
    with h5py.File(os.path.join(params().paths['parsed_news'],params().filenames['parsed_news']), 'w') as hf:
        string_dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('parsed_strings', data=df, dtype=string_dt)
    
    
    #process in yearly chunks
#    df['Year'] = df['ArticleDateCreated'].dt.year
#    for year in df['Year'].unique():
#        print('Processing articles from {}'.format(year))
#        df_year = df.loc[df['Year'] == year]
#        

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
