import re
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.stem.snowball import SnowballStemmer
from gensim.models import KeyedVectors
from cycler import cycler

import copy
import os
import glob

from fui.utils import dump_pickle, dump_csv

def extend_dict_w2v(dict_name, params, n_words=10):
    """
    Extends bloom dictionary with similar words using a pre-trained
    embedding. Default model: https://fasttext.cc/docs/en/crawl-vectors.html
    args:
    params: input_params.json
    dict_name: name of Bloom dict in params
    n_words: include n_nearest words to subject word.
    """
    
    model = KeyedVectors.load_word2vec_format(params['paths']['root']+
                                              params['paths']['w2v_model'], binary=False)
    print("Model loaded")
    dict_out = copy.deepcopy(params['bloom'])
    for k, v in params[dict_name].items():
        for val in v:
            #print('\n'+v)
            try:
                similar_words = [w[0] for w in model.most_similar(positive=val, topn=n_words)]
                dict_out[k].extend(_check_stem_duplicates(similar_words))
                #print('\n',model.most_similar(positive=v))
            except KeyError:
                continue
    return dict_out
            
def _check_stem_duplicates(word_list):
    """
    Stems list of words and removes any resulting duplicates
    """
    stemmer = SnowballStemmer("danish")
    stemmed_list = [stemmer.stem(word) for word in word_list]
    #remove duplicates after stemming
    stemmed_list = list(dict.fromkeys(stemmed_list))
    return stemmed_list
        
def bloom_measure(params, dict_name, logic, start_year=2000, end_year=2019):
    """
    Finds for articles containing words in bloom dictionary. Saves result to disk.
    args:
    params: dict of input_params
    dict_name: name of bloom dict in params
    logic: matching criteria in params
    """
    out_path = params['paths']['root']+params['paths']['bloom']+dict_name+'\\'+logic
    if not os.path.exists(out_path):
        os.makedirs(out_path)        
    
    b_E, b_P, b_U = _get_bloom_sets(params[dict_name])
    logic_str = params['options']['bloom_logic'][logic]
    print('\nLogic: '+logic_str)
    print('\n\nEconomic words: ' + repr(b_E) +
          '\n\n Political words: ' + repr(b_P) +
          '\n\n Uncertainty words: ' + repr(b_U))
    
    #get parsed articles
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'boersen*.pkl') 
    filelist = [(f,int(f[-8:-4])) for f in filelist 
                if int(f[-8:-4]) >= start_year and int(f[-8:-4]) <= end_year]
    for f in filelist:
        print('\nNow processing year '+str(f[1]))
        with open(f[0], 'rb') as data:
            try:
                df = pickle.load(data)
            except TypeError:
                print("Parsed news is not a valid pickle!")
            
        #stem articles
        with Pool() as pool:
            df['body_stemmed'] = pool.map(_stemtext, 
                                          df['ArticleContents'].values.tolist())
            
        #compare to dictionary
        with Pool() as pool:
            df['bloom'] = pool.map(partial(_bloom_compare, 
                                          logic=logic_str, 
                                          bloom_E=b_E, 
                                          bloom_P=b_P, 
                                          bloom_U=b_U), 
                                          df['body_stemmed'].values.tolist())
        
        #save to disk
        dump_pickle(out_path,'bloom'+str(f[1])+'.pkl',df[['ID2','bloom','ArticleDateCreated']])
        
        
def bloom_aggregate(folder_path, params, aggregation=['W','M','Q']):
    list_bloom = glob.glob(folder_path+'\\*.pkl')
    """
    loads saved bloom pickles and aggregates to means within 
    each aggregation frequency
    """
    if len(list_bloom):
        df = pd.DataFrame()
        for f in list_bloom:
            with open(f, 'rb') as data:
               df = df.append(pickle.load(data))
    else:
        print(f"No pickles in folder {folder_path}")
    
    agg_dict = {}
    for f in aggregation:
        bloom_idx = df[['bloom', 'ArticleDateCreated']].groupby(
            [pd.Grouper(key='ArticleDateCreated', freq=f)]
        ).agg(['mean']).reset_index()
    
        #normalize to mean = 0, std = 1
        bloom_idx.columns = bloom_idx.columns.get_level_values(0)
        bloom_idx['bloom_norm']=(bloom_idx['bloom']-bloom_idx['bloom'].mean())/bloom_idx['bloom'].std()
        bloom_idx.set_index('ArticleDateCreated', inplace=True)
        
        dump_csv(folder_path,'bloom_score_'+f+'.csv',bloom_idx)
        agg_dict[f] = bloom_idx
    return agg_dict
        
def plot_index(
    out_path, df_dict, params, plot_vix=True, freq='M', start_year=2012, end_year=2019):
    """
    """
    define_NB_colors()
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator((1,4,7,10))  
    years_fmt = mdates.DateFormatter('%Y')
    if end_year == 2019:
        end_str = '-05-31'
    else:
        end_str = ''
    
    vix = pd.read_csv(params['paths']['root']+'data/vixcurrent.csv', usecols=[0,4], names=['date','vix'], header=0)
    vix['date'] = pd.to_datetime(vix['date'])
    vix.set_index('date', inplace=True)
    vix = vix.resample(freq).mean()
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[str(start_year):str(end_year)+end_str]   
    vix['vix'] = normalize(vix['vix'])
    
    df_dict[freq] = df_dict[freq][str(start_year):str(end_year)+end_str]
    
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_dict[freq].index, df_dict[freq].bloom_norm, label='Børsen Uncertainty Index')
    if plot_vix:
        ax.plot(vix.index, vix.vix, label='VIX')
    ax.legend(frameon=False, loc='upper left')    

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    
    corr = _calc_corr(vix,df_dict[freq])
    ax.text(0.77, 0.95, 'Correlation: %.2f , frequency = %s' % (round(corr,2), freq) , transform=ax.transAxes)
    
    plt.show()
    fig.savefig(f'{out_path}\\plot_{freq}.png', dpi=300)
    return corr, fig, ax

def _calc_corr(df1,df2):
    df1 = df1.join(df2, how='inner')
    corr_mat = pd.np.corrcoef(df1.vix.values.flatten(), df1.bloom_norm.values.flatten())
    return corr_mat[0,1]

def define_NB_colors():
    """
    Defines Nationalbankens' colors and update matplotlib to use those as default
    """
    c = cycler(
        'color',
        [
            (0/255, 123/255, 209/255),
            (146/255, 34/255, 156/255),
            (196/255, 61/255, 33/255),
            (223/255, 147/255, 55/255),
            (176/255, 210/255, 71/255),
            (102/255, 102/255, 102/255)
        ])
    plt.rcParams["axes.prop_cycle"] = c
    return c

def normalize(series):
    return (series-series.mean())/series.std()

def _stemtext(text):
    # Remove any non-alphabetic character, split by space
    regex = re.compile('[^ÆØÅæøåa-zA-Z -]')
    list_to_stem = regex.sub('', text).split()

    # stem each word and join
    stemmer = SnowballStemmer("danish")
    stem_set = set([stemmer.stem(word) for word in list_to_stem])
    return stem_set

def _bloom_compare(stem_set, logic, bloom_E, bloom_P, bloom_U):          
    return eval(logic)

def _get_bloom_sets(bloom_dict):
    b_E = set(bloom_dict['economic'])
    b_P = set(bloom_dict['political'])
    b_U = set(bloom_dict['uncertainty'])
    return b_E, b_P, b_U

def _format_time_ticks(fig, ax):
    # format the ticks
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator((1,4,7,10))  
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # Format X-axis
    ax.set_xlim('2000-01-01','2019-07-01')
    fig.autofmt_xdate()

    return fig, ax

def _smooth(y, smoothing_points):
    box = np.ones(smoothing_points)/smoothing_points
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth