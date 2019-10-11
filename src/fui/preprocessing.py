# -*- coding: utf-8 -*-

import os
import pandas as pd
import pickle
import re
import warnings

from fui.utils import dump_pickle

def parse_raw_data(params, nrows=None):
    """
    Loads the data from CSV and performs some basic cleaning. Essentially the
    cleaning removes corrupted lines.
    """
    # Load the data
    csvpath = \
        params['paths']['boersen_articles']+params['filenames']['boersen_csv']
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
    
    #drop some suppliers
    df = df[df['Supplier'].isin(['Blog', 'Ritzau', 'Børsen', 'E-Avis', 'ePaper', 'Borsen'])]
    
    #drop unneccessary columns
    df.drop(labels=[
            'ShowOnWebSite', 
            'SAXoInternalId',
            'SAXoSectionPageNumber', 
            'SAXoCategory', 
            'SAXoByline', 
            'SAXoHeadline'], inplace=True, axis=1)

    df['ArticleDateCreated'] = pd.to_datetime(df['ArticleDateCreated'], format='%Y-%m-%d')
    
    #create unique row index
    df['ID2'] = df.reset_index().index
    
    #process in yearly chunks
    df['Year'] = df['ArticleDateCreated'].dt.year
    for year in df['Year'].unique():
        print('Processing articles from {}'.format(year))
        df_year = df.loc[df['Year'] == year]
        df_year['ArticleContents'] = __clean_text(df_year['ArticleContents'])
        df_year['word_count'] = df_year['ArticleContents'].str.count(' ') + 1
        print('Columns: ', df_year.columns)
        dump_pickle(params['paths']['root']+params['paths']['parsed_news'],params['filenames']['parsed_news']+'_'+str(year)+'.pkl',df_year)

def __clean_text(series):
    """
    ProcessTools: Clean raw text
    """
    # Step 1: Remove html tags
    print('Step 1: Removing tags')
    series = series.str.replace(r'<[^>]*>','',regex=True)
    
    # Step 2: Remove everything following these patterns (bylines)     
    print('Step 2: Removing bylines')
    pattern = "|".join(
        [r'\/ritzau/AFP', r'Nyhedsbureauet Direkt', r'\w*\@borsen.dk\b', r'\/ritzau/', r'\/ritzau/FINANS'])
    series = series.str.split(pattern, n=1).str[0]

    # Step 3: Remove \n, \t
    print('Step 3: Removing other')
    series = series.str.replace(r'\n', ' ')
    series = series.str.replace(r'\t', ' ')

    # Step 4: Remove additional whitespaces
    series = series.str.split().str.join(' ')

    # Step 5: Convert to lowercase
    series = series.str.lower()
    return series