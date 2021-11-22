# -*- coding: utf-8 -*-
import os
import pandas as pd
import glob
import html
import numpy as np
import re
from functools import partial
from sqlalchemy import create_engine
from multiprocessing import Pool
from src.fui.utils import params
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gensim
try:
    import lemmy
except ImportError:
    import subprocess
    import sys
    lemmypath = os.path.join(os.path.dirname(__file__), 'lemmy-2.1.0-py2.py3-none-any.whl')
    print(f'Module "lemmy" not found, installing {lemmypath}...')
    subprocess.check_call([sys.executable, "-m", "pip", "install", lemmypath])


def _remove_stopwords(word_list, stopwords):
    word_list = [word for word in word_list if word not in stopwords]
    word_list = [word for word in word_list if word not in stopwords]
    return word_list

def _load_stopwords(stopfile):
    if not os.path.exists(stopfile):
        raise Exception('No stopword file in directory')
    stopwords_file = open(stopfile, "r")
    stopwords = stopwords_file.read().splitlines()
    return stopwords

def preprocess(df, idx=0, sample=False, new=True):
    lemmatizer = lemmy.load("da")
    stopfile = params().paths['input']+'stopwords.txt'
    stopwords = _load_stopwords(stopfile=str(stopfile))
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
                                     'Weekend Outdoor', 'Weekend Rejser', 'Sponsoreret'])]
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

    # # Pre-process LDA-docs
    print(f"Preprocessing {end_n} documents...")
    with Pool(params().options['threads']) as pool:
        df['body'] = pool.map(partial(_clean_text, lemmatizer=lemmatizer, stopwords=stopwords), df['body'].values.tolist())
    df = df.loc[~df['body'].isna()]

    df = uncertainty_count(df)

    print('Columns: ', df.columns)
    df = df[['DNid', 'body', 'u_count', 'n_count', 'word_count']]
    df['word_count'] = df['word_count'].astype(int)
    if sample:
        df.reset_index(drop=True).to_feather(os.path.join(params().paths['parsed_news'], params().filenames['parsed_news'])+f'_sample.ftr')
        df.reset_index(drop=True).to_csv(os.path.join(params().paths['area060'], params().filenames['parsed_news'])+f'_sample.csv', index=False)
    else:
        file_name = os.path.join(params().paths['area060'], params().filenames['parsed_news']) + f'_part{idx}.csv'
        # df.reset_index(drop=True).to_feather(file_name+'.ftr')
        df.reset_index(drop=True).to_csv(file_name, index=False)
        #if new:
        #    update_article_counts(file_name)
        #else:
        #    insert_article_counts(file_name)

    return df

def parse_for_lda(only_new=True, sample=False, chunksize=None):
    if sample:
        df = import_articles(sample=True, chunksize=None)
        df = preprocess(df, new=False)
        return df

    elif only_new:
        df = import_new_articles(check_from=2020, chunksize=None)
        df = preprocess(df, new=True)
        return df

    else:
        df = import_articles(sample=False, chunksize=chunksize)
        if chunksize:
            for i, c in enumerate(df):
                preprocess(c, i, new=False)
            return 1
        else:
            df = preprocess(df, new=False)
            return df



def _clean_text(text, lemmatizer, stopwords):
    # Step 1: Remove html tags
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'https?:\/\/\S*', '', text)
    text = re.sub(r'www.\S*', '', text)
        
    # Step 2: Remove everything following these patterns (bylines)     
    pattern = "|".join(
        [r'\/ritzau/AFP', r'Nyhedsbureauet Direkt', r'\w*\@borsen.dk\b', r'\/ritzau/', r'\/ritzau/FINANS'])
    text = re.sub(pattern, '', text)

    # Step 3: Remove \n, \t
    text = text.replace(r'\n', ' ')
    text = text.replace(r'\t', ' ')
    text = text.replace(u'\xa0', u' ')

    # Step 4: Remove additional whitespaces
    #text = text.str.split().str.join(' ')

    # Step 6: Unescape any html entities
    text = html.unescape(text)
    
    # Manually remove some html
    text = text.replace(r'&rsquo', '')
    text = text.replace(r'&ldquo', '')
    text = text.replace(r'&rdquo', '')
    text = text.replace(r'&ndash', '')

    error = ['år', 'bankernes', 'bankerne', 'dst', 'priser', 'bankers']
    drop = ['bre', 'nam', 'ritzau', 'st', 'le', 'sin', 'år', 'stor', 'me', 'når', 'se', 'dag', 'en', 'to', 'tre',
            'fire', 'fem', 'seks', 'syv', 'otte', 'ni', 'ti']

    tokens = gensim.utils.simple_preprocess(text, deacc=False, min_len=2, max_len=25)
    tokens = _remove_stopwords(tokens, stopwords)
    tokens = [lemmatizer.lemmatize("", word)[0] for word in tokens if not word in error]


    tokens = [w for w in tokens if not w in drop]
    tokens = [w.replace("bankernes", "bank") for w in tokens]
    tokens = [w.replace("bankers", "bank") for w in tokens]
    tokens = [w.replace("kris", "krise") for w in tokens]
    tokens = [w.replace("bile", "bil") for w in tokens]
    tokens = [w.replace("bankerne", "bank") for w in tokens]
    tokens = [w.replace("priser", "pris") for w in tokens]
    tokens = [w for w in tokens if not w.isdigit()]

    string = ' '.join([word for word in tokens])
    if re.search('[a-zA-Z]', string) is None:
        string = np.NaN

    return string

def load_parsed_data(sample = False):
    if sample:
        article_files = glob.glob(os.path.join(params().paths['parsed_news'],
                                               params().filenames['parsed_news']) + '_sample.ftr')[0]
        df = pd.read_feather(article_files)
    else:
        article_files = glob.glob(os.path.join(params().paths['parsed_news'],
                                               params().filenames['parsed_news']) + '_part*.ftr')
        if len(article_files) == 0:
            raise FileNotFoundError(f"{os.path.join(params().paths['parsed_news'],params().filenames['parsed_news'])} not found")
        df_list = []
        for f in article_files:
            df = pd.read_feather(f)
            df_list.append(df)
        df = pd.concat(df_list)
    return df

def import_scraped_articles():
    path = params().paths['scraped']  # use your path
    all_files = glob.glob(path + "/*/scraped*.csv")

    dfs = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep=";")
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df['date'] = df['date'].replace('404', np.nan)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['headline_web', 'date'])
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 1', 'sep=','headline_q'])
    # url ledelse/ is missing proper date (date is date of scrape), drop these
    df = df.loc[df['url'].str.find('/ledelse/') == -1]
    df.rename(columns={'headline_web': 'headline', 'bodytext': 'body', 'url': 'byline_alt'}, inplace=True)
    return df

def import_csv():
    #Step 1: merge df1 and df2 on ID. Replace Eavis arkiv articles in df1 with merged counterpart.
    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv']
    df1 = pd.read_csv(csvpath, sep=';', encoding='UTF-16', error_bad_lines=False)
    df1['ArticleDateCreated'] = pd.to_datetime(df1['ArticleDateCreated'], format="%Y-%m-%d", errors='coerce')
    df1 = df1.dropna(axis=0, subset=['ArticleDateCreated','ArticleContents', 'Title', 'ID'])
    df1['ID'] = pd.to_numeric(df1['ID'], errors='coerce')

    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv2']
    df2 = pd.read_csv(csvpath, sep=';', encoding='UTF-16', error_bad_lines=False)
    df2['dateRelease'] = pd.to_datetime(df2['dateRelease'], format="%Y-%m-%d", errors='coerce')
    df2 = df2.dropna(axis=0, subset=['dateRelease', 'headline', 'id', 'content'])
    df2['id'] = pd.to_numeric(df2['id'], errors='coerce')

    df1 = df1.merge(df2, left_on='ID', right_on='id', how='left')
    df1['ArticleContents'].loc[df1['Supplier'] =='E-Avis arkiv'] = df1['content']
    df1['Title'].loc[df1['Supplier'] =='E-Avis arkiv'] = df1['headline']
    df1['Author'].loc[df1['Supplier'] =='E-Avis arkiv'] = df1['byline']

    del df2

    keepcols = ['ID', 'Title', 'ArticleContents',
                'ArticleDateCreated',
                'Author', 'SectionName',
                'SAXoCategory', 'SAXoByline']

    df1 = df1[keepcols]

    df1.rename(columns={'ID':'id', 'Title':'headline', 'ArticleContents':'body',
                'ArticleDateCreated':'date',
                'Author':'byline', 'SectionName':'section_name',
                'SAXoCategory':'category', 'SAXoByline':'byline_alt'}, inplace=True)

    #Step 2: Keep articles until April 1st 2019. Use data from df3 for period after that.
    df1 = df1.loc[(df1['date'] < pd.Timestamp(2019,4,1))]

    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv3']
    df3 = pd.read_csv(csvpath, sep=';', encoding='latin-1', error_bad_lines=False)
    df3['dateRelease'] = pd.to_datetime(df3['dateRelease'], format="%Y-%m-%d", errors='coerce')
    df3['id'] = pd.to_numeric(df3['id'], errors='coerce')

    keepcols = ['id', 'headline', 'content', 'dateRelease', 'byline',
                'saxoStrSection', 'saxoCategory',
                'saxoAuthor']

    df3 = df3[keepcols]

    df3.rename(columns={'id':'id', 'headline':'headline', 'content':'body',
                'dateRelease':'date',
                'byline':'byline', 'saxoStrSection':'section_name',
                'saxoCategory':'category', 'saxoAuthor':'byline_alt'}, inplace=True)

    df3 = df3.loc[(df3['date'] >= pd.Timestamp(2019,4,1))]
    df3 = df3.loc[(df3['date'] < pd.Timestamp(2020,3,1))]

    start_n = df3.shape[0]
    df3 = df3.dropna(axis=0, subset=['headline', 'id', 'body'])
    end_n = df3.shape[0]
    print('Dropped {} articles with NaN headline, id or content'.format(start_n-end_n))

    start_n = df3.shape[0]
    df3 = df3.drop_duplicates(subset=['headline', 'body'])
    end_n = df3.shape[0]
    print('Dropped {} articles with duplicate headline, id or content'.format(start_n-end_n))

    df1 = df1.append(df3)

    df4 = import_scraped_articles()
    df1 = df1.append(df4)
    df1 = df1.sort_values('date')

    del df3, df4

    df1['body'] = df1['body'].str.replace(r'<[^>]*>', '', regex=True)
    df1['body'] = df1['body'].str.replace(r'https?:\/\/\S*', '', regex=True)
    df1['body'] = df1['body'].str.replace(r'www.\S*', '', regex=True)
    df1['body_len'] = df1['body'].str.len()

    df1.to_csv(params().paths['boersen_articles']+params().filenames['boersen_merged'], encoding='utf-8', index=False)

    dtypes = {'id': 'bigint', 'headline': 'NVARCHAR(500)',
              'body': 'ntext', 'date': 'datetime2',
              'byline': 'NVARCHAR(500)', 'section_name': 'NVARCHAR(500)',
              'category': 'NVARCHAR(500)', 'byline_alt': 'NVARCHAR(500)',
              'body_len': 'int'}

    return df1, dtypes

def import_new_articles(check_from=2020, chunksize=None):
    sql = f"""
            select 
              a.[DNid]  
              ,a.[date]
              ,a.[headline]
              ,a.[body]
              ,a.[byline]
              ,a.[byline_alt]
              ,a.[category]
              ,a.[section_name]
              ,a.[id]
              ,a.[article_id]
              ,a.[version_id]
                from area060.all_articles a
                LEFT JOIN area060.article_word_counts b on a.DNid = b.DNid
                where b.DNid IS NULL and year(a.date) >= {check_from}
		"""
    engine = create_engine(
        'mssql+pyodbc://SRV9DNBDBM078/workspace01?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
    df = pd.read_sql(sql, con=engine, chunksize=chunksize)
    return df

def import_articles(sample=False, chunksize=None):
    if not sample:
        sql = """SELECT
              [DNid]  
              ,[date]
              ,[headline]
              ,[body]
              ,[byline]
              ,[byline_alt]
              ,[category]
              ,[section_name]
              ,[id]
              ,[article_id]
              ,[version_id]
          FROM [workspace01].[area060].[all_articles]
        """
        engine = create_engine(
            'mssql+pyodbc://SRV9DNBDBM078/workspace01?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
        df = pd.read_sql(sql, con=engine, chunksize=chunksize)
    else:
        sql = """SELECT TOP 1000
              [DNid] 
              ,[date]
              ,[headline]
              ,[body]
              ,[byline]
              ,[byline_alt]
              ,[category]
              ,[section_name]
              ,[id]
              ,[article_id]
              ,[version_id]
          FROM [workspace01].[area060].[all_articles]
          ORDER BY NEWID()
        """
        engine = create_engine(
            'mssql+pyodbc://SRV9DNBDBM078/workspace01?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
        df = pd.read_sql(sql, con=engine, chunksize=None)
    return df

def uncertainty_count(df, extend=True, workers=3):
    """
    Counts u-words in articles. Saves result as HDF to disk.
    args:
        extend (bool): Use extended set of u-words
    """
    if extend:
        U_set = set(list(params().dicts['uncertainty_ext'].values())[0])
        filename = params().filenames['parsed_news_uc_ext']
    else:
        U_set = set(list(params().dicts['uncertainty'].values())[0])
        filename = params().filenames['parsed_news_uc']
    print(U_set)

    print(f"Counting uncertainty words...")

    #compare to dictionary
    with Pool(workers) as pool:
        df['u_count'] = pool.map(partial(_count,
                                 word_set=U_set),
                                 df['body'].values.tolist())

    N_list = list(params().dicts['negations'].values())[0]
    with Pool(workers) as pool:
        df['n_count'] = pool.map(partial(_count,
                             word_set=N_list),
                             df['body'].values.tolist())

    return df


def _count(text, word_set):
    count = 0
    for word in word_set:
        count += text.count(word)
    return count