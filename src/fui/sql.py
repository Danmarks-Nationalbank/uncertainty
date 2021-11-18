import pyodbc
import os

def create_empty_table(dict, table_name):
    query = f'CREATE TABLE [workspace01].[area028].[{table_name}](' + \
            ', '.join(['%s %s' % (key, value) for (key, value) in dict.items()]) + \
            ');'
    execute_query(f"DROP TABLE [workspace01].[area028].[{table_name}];")
    execute_query(query)
    return query

def execute_query(sql_query, result=False):
    """
    General utility to pass a SQL query to our server
    """
    with pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                        server='SRV9DNBDBM078', database='workspace01', Trusted_Connection='Yes') as con:
        cursor = con.cursor()
        res = []
        try:
            for statement in sql_query.split(';'):
                cursor.execute(statement+';')
                if result:
                    res.append(cursor.fetchall())
            print('Query executed')
        except Exception as e:
            print(e)

    if result:
        return res
    else:
        return 1

def bulk_insert(table_name, file_name, fieldterminator = r","):
    """
    Function to bulk insert `file_name` into `table_name`. `table_name` has to
    be present already in the sql server. `file_name` should be a tab-separated
    csv file located in //srv9dnbdbm078/Analyseplatform/area028/. To modify the
    bulk inster query, edit the file under `sql/upload/create_table_parsed_news.sql`
    """
    with open('../sql/bulk_import_to_table.sql', 'r') as q:
        query = """{}""".format(
            q.read().replace('__TABLE_NAME__', table_name).replace('__FILE_NAME__', file_name).replace(
                '__FIELDTERMINATOR__', fieldterminator)
            )
    execute_query(query)

def get_last_date():
    query = """
    SELECT max([date])
    FROM [workspace01].[area060].[all_articles]
    """
    res = execute_query(query, result=True)
    return res[0][0][0]


def insert_article_counts(article_file):
    query = f"""
            USE [workspace01]

            CREATE TABLE #counts(
              DNid int not NULL
             ,body nvarchar(max)
             ,u_count int not NULL
             ,n_count int not NULL
             ,word_count int not NULL
            )

            BULK INSERT #counts 
               FROM '{article_file}' 
               WITH (
                 FORMAT = 'CSV' 
               , CODEPAGE =  '65001'
               , FIELDQUOTE = '"'
               , FIELDTERMINATOR = ','
               , ROWTERMINATOR = '\n'
               , FIRSTROW = 2)

            INSERT INTO area060.article_word_counts
            SELECT * FROM #counts 
            """
    if not os.path.isfile(article_file):
        FileNotFoundError(f"{article_file} not found.")
    else:
        execute_query(query)


def update_article_counts(article_file):
    query = f"""
            USE [workspace01]
    
            CREATE TABLE #counts(
              DNid int not NULL
             ,body nvarchar(max)
             ,u_count int not NULL
             ,n_count int not NULL
             ,word_count int not NULL
            )
            
            BULK INSERT #counts 
               FROM '{article_file}' 
               WITH (
                 FORMAT = 'CSV' 
               , CODEPAGE =  '65001'
               , FIELDQUOTE = '"'
               , FIELDTERMINATOR = ','
               , ROWTERMINATOR = '\n'
               , FIRSTROW = 2)
    
            MERGE area060.article_word_counts AS TARGET
            USING #counts AS SOURCE 
            ON (TARGET.DNid = SOURCE.DNid) 
            --When records are matched, update the records if there is any change
            WHEN MATCHED
            THEN UPDATE SET TARGET.u_count = SOURCE.u_count, TARGET.n_count = SOURCE.n_count, TARGET.body = SOURCE.body
            ,TARGET.word_count = SOURCE.word_count
            --When no records are matched, insert the incoming records from source table to target table
            WHEN NOT MATCHED BY TARGET 
            THEN INSERT (DNid, body, u_count, n_count, word_count) 
            VALUES (SOURCE.DNid, SOURCE.body, SOURCE.u_count, SOURCE.n_count, SOURCE.word_count);
            """
    if not os.path.isfile(article_file):
        FileNotFoundError(f"{article_file} not found.")
    else:
        execute_query(query)


def update_article_topics(topic_file):
    query = f"""
            USE [workspace01]
            
                CREATE TABLE #topics(
                    DNid int not NULL,
                    topic int not NULL,
                    probability float not NULL,
                )
                BULK INSERT #topics 
                   FROM '{topic_file}' 
                   WITH (
                     FORMAT = 'CSV' 
                   , CODEPAGE =  '65001'
                   , FIELDQUOTE = '"'
                   , FIELDTERMINATOR = ','
                   , ROWTERMINATOR = '\n'
                   , FIRSTROW = 2)
            
                   MERGE [area060].[article_topics_new] AS TARGET
                USING #topics AS SOURCE 
                ON (TARGET.DNid = SOURCE.DNid) 
                --When records are matched, update the records if there is any change
                WHEN MATCHED
                THEN UPDATE SET TARGET.topic = SOURCE.topic, TARGET.probability = SOURCE.probability
                --When no records are matched, insert the incoming records from source table to target table
                WHEN NOT MATCHED BY TARGET 
                THEN INSERT (DNid, topic, probability) VALUES (SOURCE.DNid, SOURCE.topic, SOURCE.probability);
            """
    if not os.path.isfile(topic_file):
        FileNotFoundError(f"{topic_file} not found.")
    else:
        execute_query(query)
