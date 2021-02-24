import sys
sys.path.insert(0, "C:\\projects\\OBM\\hackenv")
sys.path.append("C:\\projects\\OBM\\src")
from src.fui.utils import params, execute_query, bulk_insert, create_empty_table
import pandas as pd
from src.fui.preprocessing import import_csv

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



