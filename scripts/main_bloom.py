import sys
import os
#hacky spyder crap
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\projects\\FUI\\src')
sys.path.insert(1, 'C:\\projects\\FUI\\env')
sys.path.insert(1, 'C:\projects\FUI\env\Lib\site-packages') 
import json
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob 
import codecs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from zipfile import ZipFile 

from fui.preprocessing import parse_raw_data
from fui.utils import params, read_hdf, read_h5py
from fui.bloom import bloom_measure, extend_dict_w2v, plot_index, _aggregate

if __name__ == "__main__":
    
    #Step 0: Parse input parameter file
    
    #Step 1: Import Børsen articles
    #df = parse_raw_data(nrows=10000)
    df2 = read_h5py(os.path.join(params().paths['parsed_news'],params().filenames['parsed_news']), obj='parsed_strings')
    #Step 2: Extend Bloom dict using pre-trained embeddings
    #dict_extend = extend_dict_w2v('bloom_extended', n_words=10)            
    
    #Step 3: Get Bloom binary measure for each article using eval conditions in input_params
#    for logic in ['EandPandU','EandU','EorPandU','PandU']:
#        bloom_measure(dict_extend,logic=logic,start_year=2000,end_year=2019,weighted=False)
    #bloom_measure(dict_extend,name='u_weighted_extend',logic='EandP',start_year=2000,end_year=2019,weighted=True)

    
    # Step 4: Aggregate, write csv and plot
#    for logic in ['EandP']:
#        bloom_path = params().paths['bloom']+'u_weighted_extend'+'\\'+logic
#        bloom_agg = _aggregate(bloom_path, aggregation=['M'])
#        
#        corr, fig, ax = plot_index(bloom_path, bloom_agg, plot_vix=True, 
#                                   freq='M', start_year=2000, end_year=2019)
#        print('Logic: '+logic+', Corr: '+"%.2f" % round(corr,3))
        
        
#    #Step 5: Package to zip
#    files_to_zip = [f for f in glob.glob('C:/projects/FUI/data/bloom/bloom_extended_w2v/**/*') if not os.path.basename(f).endswith('pkl')]
#    files_to_zip = [os.path.join(*(f.split(os.path.sep)[1:])) for f in files_to_zip]
#    os.chdir(params().paths['bloom']+'bloom_extended_w2v')
#    with ZipFile('fui.zip','w') as zip: 
#        # writing each file one by one 
#        for file in files_to_zip: 
#            zip.write(file) 
            