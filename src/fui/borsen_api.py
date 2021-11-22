import requests
import json
import pandas as pd
import re
from datetime import timedelta, date, datetime
from src.fui.sql import execute_query, get_last_date
import time
from pathlib import Path
import os


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_articles(date, key):
    # Please contact ALEM or EGR for the API key
    datestr = date.strftime("%Y%m%d")
    URL = "https://borsen-natbank-api-project.azurewebsites.net/api/RequestArticles"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    params = {'code': key, 'article_date': datestr}
    articles = requests.get(URL, headers=headers, params=params)
    return articles

def download_articles(sdate, edate, key, path):
    start_date = datetime.strptime(sdate, '%Y%m%d').date()
    end_date = datetime.strptime(edate, '%Y%m%d').date()
    try:
        output = pd.read_feather(str(path))
        last_date = pd.to_datetime(output['date']).max().date()
        print(f"Last date downloaded is {last_date}. Loading existing file.")
        return output
    except FileNotFoundError:
        output = pd.DataFrame()
        ldate = sdate
        last_date = start_date
    if last_date > end_date:
        print("All dates in range downloaded, stopping.")
        return 0
    # loop over dates
    print(last_date, end_date)
    for single_date in daterange(last_date, end_date):
        print(single_date)
        articles = get_articles(single_date, key)
        try:
            articles_json = json.loads(articles.content)
        except json.decoder.JSONDecodeError:
            print(articles.content)
        for a in articles_json['data']:
            a_dict = {}
            if a['body'] != '[]' and a['published'] != ' ':
                body = re.sub(r'(^\[)', '', a['body'])
                body = re.sub(r'(\]$)', '', body)
                byline = re.sub(r'(^\[)', '', a['author_name'])
                byline = re.sub(r'(\]$)', '', byline)
                a_dict['headline'] = a['title']
                a_dict['body'] = body
                a_dict['byline'] = byline
                a_dict['byline_alt'] = a['altbyline']
                a_dict['category'] = a['tags']
                a_dict['section_name'] = a['home_section']
                a_dict['article_id'] = a['dcterms_identifier']
                a_dict['version_id'] = a['sequenceNumber']
                a_dict['date'] = a['published']
                output = output.append(a_dict, ignore_index=True)
    output.to_feather(str(path))
    time.sleep(2)
    return output

def upload_new_articles(key, last_date=None):
    if last_date is None:
        #set last date as today
        last_date = date.today()
    #get last date available on server
    first_date = get_last_date()

    if first_date >= (last_date - timedelta(days = 1)):
        print(f"Articles on server are current as of yesterday.")
        return 1

    last_date = last_date.strftime('%Y%m%d')
    first_date = first_date.strftime('%Y%m%d')

    print(f"Uploading articles from {first_date} to {last_date}")
    upload_articles(first_date, last_date, key)

def upload_articles(sdate, edate, key):
    folder = Path(__file__).parents[2] / 'data' / 'borsen'
    if not os.path.exists(str(folder)):
        os.makedirs(str(folder))
    path = Path(__file__).parents[2] / 'data' / 'borsen' / f'articles_{sdate}_{edate}.ftr'
    start_date = datetime.strptime(sdate, '%Y%m%d').date()
    end_date = datetime.strptime(edate, '%Y%m%d').date()
    try:
        df = pd.read_feather(str(path))
        last_date = pd.to_datetime(df['date']).max().date()
    except FileNotFoundError:
        print("File not found, downloading articles articles.")
        last_date = start_date
        download_articles(sdate, edate, key, path)
        df = pd.read_feather(str(path))
    if last_date < end_date:
        print(f"Last date saved is {last_date}, importing articles.")
        download_articles(sdate, edate, key, path)
        df = pd.read_feather(str(path))
    df['body'] = df['body'].str.replace('\n', ' ')
    df['date'] = pd.to_datetime(df['date'])
    df.to_csv(f"//srv9dnbdbm078/Analyseplatform/area060/articles_{sdate}_{edate}.csv", index=False)
    print(f"Inserting {str(path)} into area060.all_articles.")
    bulk_insert_articles(f"//srv9dnbdbm078/Analyseplatform/area060/articles_{sdate}_{edate}.csv")

def bulk_insert_articles(path):
    QUERY = f"""USE [workspace01]

    CREATE TABLE #articles(
	headline  nvarchar(2000)
    ,body nvarchar(max)
    ,byline nvarchar(2000)
    ,byline_alt nvarchar(2000)
    ,category nvarchar(2000)
    ,section_name nvarchar(2000)
	,article_id nvarchar(2000)
    ,version_id nvarchar(2000)
	,[date] date
    )

    BULK INSERT #articles 
       FROM '{path}' 
       WITH (
         FORMAT = 'CSV' 
       , CODEPAGE =  '65001'
       , FIELDQUOTE = '"'
       , FIELDTERMINATOR = ','
       , ROWTERMINATOR = '\n'
       , FIRSTROW = 2)

    insert into area060.all_articles
    ([date] ,headline ,body ,byline ,byline_alt ,category ,section_name , id, article_id, version_id)
     select new.[date] ,new.headline ,new.body ,new.byline ,new.byline_alt ,new.category ,new.section_name , new.id, new.article_id, new.version_id from (
	 select 
	 cast(new.[date] as date) as [date]
    ,new.headline
    ,new.body
    ,new.byline
    ,new.byline_alt
    ,new.category
    ,new.section_name 
    ,NULL as id
    ,new.article_id
    ,new.version_id
	,row_number() over (partition by new.article_id order by new.version_id) as rn
	from #articles new
	) new
	
	left join area060.all_articles org on org.article_id = new.article_id
	where  org.article_id is null AND rn = 1
    """
    execute_query(QUERY)
