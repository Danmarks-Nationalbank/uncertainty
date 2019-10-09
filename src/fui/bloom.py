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

from src.fui.utils import dump_pickle

def extend_dict_w2v(dict_name, params, n_words=10):
    model = KeyedVectors.load_word2vec_format(params['paths']['w2v_model'], binary=False)
    for k, v in params[dict_name].items():
        for v in v:
            #print('\n'+v)
            try:
                similar_words = [w[0] for w in model.most_similar(positive=v, topn=n_words)]
                params[dict_name][k].extend(_check_stem_duplicates(similar_words))
                #print('\n',model.most_similar(positive=v))
            except KeyError:
                continue
    return params[dict_name]
            
def _check_stem_duplicates(word_list):
    stemmer = SnowballStemmer("danish")
    stemmed_list = [stemmer.stem(word) for word in word_list]
    #remove duplicates after stemming
    stemmed_list = list(dict.fromkeys(stemmed_list))
    return stemmed_list
        
def bloom_measure(filelist, dict_name, params, bloomtuple=None):
    if bloomtuple is None:
        b_E, b_P, b_U = _get_bloom_sets(params[dict_name])
    else:
        b_E, b_P, b_U = bloomtuple
    print('Economic words: ' + repr(b_E) +
          '\n\n Political words: ' + repr(b_P) +
          '\n\n Uncertainty words: ' + repr(b_U))
    
    #df_main = pd.DataFrame()
    for f in filelist:
        print('\nNow processing year '+str(f[1]))
        with open(f[0], 'rb') as data:
            try:
                df = pickle.load(data)
            except TypeError:
                print("Parsed news is not a valid pickle!")
        with Pool() as pool:
            df['body_stemmed'] = pool.map(_stemtext, df['ArticleContents'].values.tolist())
        print(df['body_stemmed'].head(10))
        with Pool() as pool:
            df['bloom'] = pool.map(partial(_bloom_compare, bloom_E=b_E, bloom_P=b_P, bloom_U=b_U), df['body_stemmed'].values.tolist())
        dump_pickle(params['paths']['parsed_news'],dict_name+'_E&P&U_'+str(f[1])+'.pkl',df[['ID2','bloom','ArticleDateCreated']])
        #df_main = df_main.append(df)
    #return df_main

def plot_index(
    df, index_name, frequency=None, smoothing=None, plot_gdp=False,
    export_index=False, split=None, refactor = True):
    """
    """
    # Set default parameters
    if frequency is None:
        if index_name=='afinn_norm':
            frequency = 'W'
        elif index_name == 'bloom':
            frequency = 'M'
        else:
            frequency = 'W'
    if smoothing is None:
        if index_name=='afinn_norm':
            smoothing = 15
        elif index_name == 'bloom':
            smoothing = 3
        else:
            smoothing = 10
    
    # Plot index
    plot = df[[index_name, 'ArticleDateCreated']].groupby(
        [pd.Grouper(key='ArticleDateCreated', freq=frequency)]
    ).agg(['mean', 'count']).reset_index()

    if split is not None:
        plot[index_name, 'mean'] = plot[index_name, 'mean'] - df[[not i for i in split]][[index_name, 'ArticleDateCreated']].groupby(
            [pd.Grouper(key='ArticleDateCreated', freq=frequency)]
        ).agg('mean').reset_index()[index_name]
        # print(plot)

    if index_name == 'bloom':
        plot[index_name, 'mean'] = plot[index_name, 'mean']*(-1)

    toplot = _smooth(plot[index_name, 'mean'], smoothing)

    if refactor:
        max_0 = toplot.max()
        min_0 = toplot.min()
        max_w = 3
        min_w = -2.5
        scale = (max_w - min_w)/(max_0 - min_0)
        toplot = toplot*scale - max_0*scale + max_w
        plot[index_name, 'mean'] = plot[index_name, 'mean']*scale - max_0*scale + max_w

    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(plot['ArticleDateCreated'], plot[index_name, 'mean'], s=plot[index_name, 'count']/800, label=None) 
    index_line, = ax.plot(plot['ArticleDateCreated'].loc[smoothing-1:], toplot, lw=2, label='Newspaper index')
    ax.legend([index_line], ['Newspaper index'], frameon=False)

    if plot_gdp:
        # Load gdp data
        gdp = pd.read_csv('../../data/gdp3.csv', usecols=[2,3], names=['quarter', 'growth'])
        gdp['quarter'] = pd.to_datetime(gdp['quarter'].str.replace('K', 'Q')) + pd.DateOffset(months=3)

        # plot gdp data
        # ax2 = ax.twinx()
        ax.plot(gdp['quarter'], gdp['growth'], color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1], ls = '--', label='GDP growth in quarter')
        ax.legend(loc='lower right', frameon=False)

        # ax2.set_ylabel('GDP growth (%)')

    fig, ax = _format_time_ticks(fig, ax)
    
    if export_index:
        index = pd.DataFrame()
        index['date'] = plot['ArticleDateCreated'].loc[smoothing-1:]
        index[index_name] = smooth(plot[index_name, 'mean'], smoothing)
        plot[str(index_name+'_norm')]=(df[index_name]-df[index_name].mean())/df[index_name].std()
        return fig, ax, index, plot
    else:
        return fig, ax

def _stemtext(text):
    # Remove any non-alphabetic character, split by space
    regex = re.compile('[^ÆØÅæøåa-zA-Z -]')
    list_to_stem = regex.sub('', text).split()

    # stem each word and join
    stemmer = SnowballStemmer("danish")
    stem_set = set([stemmer.stem(word) for word in list_to_stem])
    return stem_set

def _bloom_compare(stem_set, bloom_E, bloom_P, bloom_U):          
    return bool(bloom_E & stem_set) and bool(bloom_P & stem_set) and bool(bloom_U & stem_set)

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