import requests
import json
import pandas as pd
import re
from datetime import timedelta, date
import time
import pyodbc


def execute_query(sql_query):
    """
    General utility to pass a SQL query to our server
    """
    with pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                        server='SRV9DNBDBM078', database='workspace01', Trusted_Connection='Yes') as con:
        cursor = con.cursor()
        try:
            for statement in sql_query.split(';'):
                cursor.execute(statement)
            print('Query executed')
        except Exception as e:
            print(e)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def get_articles(date):
    # Please contact ALEM or EGR for the API key
    code = ''
    datestr = date.strftime("%Y%m%d")
    URL = "https://borsen-natbank-api-project.azurewebsites.net/api/RequestArticles"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    params = {'code': code, 'article_date': datestr}
    articles = requests.get(URL, headers=headers, params=params)
    return (articles)

def export_articles(start_date, end_date):
    output = pd.DataFrame()

    # loop over dates
    for single_date in daterange(start_date, end_date):
        print(single_date)
        articles = get_articles(single_date)
        articles_json = json.loads(articles.content)
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
        time.sleep(2)
    output.to_csv(f"\\srv9dnbdbm078\Analyseplatform\area060\articles_{start_str}_{end_str}.csv", index=False)
    return (output)


if __name__ == '__main__':
    start_date = date(2020, 5, 19)
    end_date = date(2021, 2, 23)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    QUERY = """USE [workspace01]


    CREATE TABLE #articles(
    article_id nvarchar(2000)
    , body nvarchar(max)
    ,byline nvarchar(2000)
    ,byline_alt nvarchar(2000)
    ,category nvarchar(2000)
    ,[date] date
    ,headline  nvarchar(2000)
    ,section_name nvarchar(2000)
    ,version_id nvarchar(2000)
    )


    BULK INSERT #articles 
       FROM '//srv9dnbdbm078/Analyseplatform/area060/articles_20200519_20210223.csv' 
       WITH (
         FORMAT = 'CSV' 
       , CODEPAGE =  '65001'
       , FIELDQUOTE = '"'
       , FIELDTERMINATOR = ','
       , ROWTERMINATOR = '\n'
       , FIRSTROW = 2)


    ALTER TABLE area060.all_articles
    ADD article_id nvarchar(2000),
        version_id nvarchar(2000);

    ALTER TABLE area060.all_articles
    ALTER COLUMN category nvarchar(2000); 

    insert into area060.all_articles
    ([date] ,headline ,body ,byline ,byline_alt ,category ,section_name , id, article_id, version_id)
    select 
     cast([date] as date)
    ,headline
    ,body
    ,byline
    ,byline_alt
    ,category
    ,section_name 
    , NULL as id
    ,article_id
    ,version_id
    from #articles
    order by [date]
    """

    export_articles(start_date, end_date)
    execute_query(QUERY)
