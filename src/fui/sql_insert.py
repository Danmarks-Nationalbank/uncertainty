import sys
sys.path.insert(0, "C:\\projects\\OBM\\hackenv")
sys.path.append("C:\\projects\\OBM\\src")
from src.fui.utils import params, execute_query, bulk_insert, create_empty_table
import pandas as pd
import numpy as np
import glob

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
    df = df.drop(columns=['Unnamed: 0', 'headline_q'])
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

    df1.to_csv(params().paths['boersen_articles']+params().filenames['boersen_merged'], encoding='utf-8')

    dtypes = {'id': 'bigint', 'headline': 'NVARCHAR(500)',
              'body': 'ntext', 'date': 'datetime2',
              'byline': 'NVARCHAR(500)', 'section_name': 'NVARCHAR(500)',
              'category': 'NVARCHAR(500)', 'byline_alt': 'NVARCHAR(500)'}

    return df1, dtypes

if __name__ == '__main__':
    table_name = 'BorsenArticles'
    _, dtypes = import_csv()
    del _

    # loop over chunks
    for i, c in enumerate(pd.read_csv(params().paths['boersen_articles']+params().filenames['borsen_merged'],chunksize=200000)):
        print('Chunk ', i)
        if i == 0:
            create_empty_table(dtypes, table_name)
        df.to_csv('//srv9dnbdbm078/Analyseplatform/area028/borsen_temp.csv', index=False, encoding='utf-8', header=False)
        bulk_insert(table_name, 'borsen_temp.csv', fieldterminator=",")

    # finalize by creating index
    execute_query(f"create clustered columnstore index cl_cs_idx on workspace01.area045.{table_name}")



