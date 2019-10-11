import sys
import os
#hacky spyder crap
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\projects\\FUI\\src')
sys.path.insert(1, 'C:\\projects\\FUI\\hackenv')
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob 
from cycler import cycler
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from zipfile import ZipFile 


from fui.utils import main_directory
from fui.preprocessing import parse_raw_data
from fui.bloom import bloom_measure, extend_dict_w2v, bloom_aggregate, plot_index

if __name__ == "__main__":
    
    #Step 0: Parse input parameter file
    PARAMS_PATH = 'scripts/input_params.json'
    with open(PARAMS_PATH, encoding='utf8') as json_params:
        params = json.load(json_params)
    
    #Step 1: Import Børsen articles
    parse_raw_data(params, nrows=NROWS)
    
    #Step 2: Extend Bloom dict using pre-trained embeddings
    params['bloom_extended_w2v'] = extend_dict_w2v('bloom_extended', params, n_words=10)            
    
    #Step 3: Get Bloom binary measure for each article using eval conditions in input_params
    for logic in ['EandPandU','EandU','EorPandU','PandU','U']:
        bloom_measure(params,dict_name='bloom_extended_w2v',logic=logic,start_year=2000, end_year=2019)
    
    #Step 4: Aggregate, write csv and plot
    for logic in ['EandPandU','EandU','EorPandU','PandU','U']:
        bloom_path = params['paths']['root']+params['paths']['bloom']+'bloom_extended_w2v'+'\\'+logic
        bloom_agg = bloom_aggregate(bloom_path, params, aggregation=['M','Q','D'])
        
        #corr, fig, ax = plot_index(bloom_agg, plot_vix=True, freq='M', start_year=2012, end_year=2019)
        #print('Logic: '+logic+', Corr: '+"%.2f" % round(corr,2))
        #fig.savefig(bloom_path+'\\plot.png')
        
    #Step 5: Package to zip
    files_to_zip = [f for f in glob.glob('C:/projects/FUI/data/bloom/bloom_extended_w2v/**/*') if not os.path.basename(f).endswith('pkl')]
    files_to_zip = [os.path.join(*(f.split(os.path.sep)[1:])) for f in files_to_zip]
    os.chdir(params['paths']['root']+params['paths']['bloom']+'bloom_extended_w2v')
    with ZipFile('fui.zip','w') as zip: 
        # writing each file one by one 
        for file in files_to_zip: 
            zip.write(file) 