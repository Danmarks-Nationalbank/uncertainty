import re
from afinn import Afinn
import time
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.stem.snowball import SnowballStemmer
from cycler import cycler
import json

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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

afinn = Afinn(language='da')
stemmer = SnowballStemmer("danish")

def f_writetime(time):
    hours = np.int8(np.floor(time/3600))
    minutes = np.int8(np.floor((time - hours*3600)/60))
    seconds = np.floor(time - hours*3600 - minutes*60)
    
    return str(hours) + " hour(s), " + str(minutes) + " minute(s), and " + str(seconds) + " seconds"

def cleanhtml(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', text)
    wordlist = ['/ritzau', '/Reuters']
    for word in wordlist:
        cleantext = cleantext.replace(word, '')
    cleantext = ' '.join(cleantext.split()).strip().encode("utf-8").decode("unicode-escape")
    # cleanr = re.compile('/.*')
    # cleantext = re.sub(cleanr, '', cleantext)
    return cleantext

def stemtext(text):
    # Remove any non-alphabetic character, split by space
    regex = re.compile('[^ÆØÅæøåa-zA-Z -]')
    list_to_stem = regex.sub('', text).split()

    # stem each word and join
    stem_set = set([stemmer.stem(word) for word in list_to_stem])
    return stem_set

def define_bloom_sets():
    bloom_E = set(['erhverv', 'forretning', 'handel', 'økonomi', 'økonomisk'])
    bloom_P = set(['politik', 'regulering', 'skat', 'udgift', 'underskud', 'nationalbank', 'folketing', 'regering'])
    bloom_U = set(['usik', 'usikker'])
    return bloom_E, bloom_P, bloom_U

def bloom_measure(text, bloomtuple=None):
    if bloomtuple is None:
        bloom_E, bloom_P, bloom_U = define_bloom_sets()
    else:
        bloom_E, bloom_P, bloom_U = bloomtuple
    stem_set = stemtext(text)

    return bool(bloom_E & stem_set) & bool(bloom_P & stem_set) & bool(bloom_U & stem_set)


def process_text(series):
    with Pool() as pool:
        list_of_bodies = pool.map(cleanhtml, series.values.tolist())
    
    return list_of_bodies

def load_raw_data(datafile='data/Nat_bank_articles.csv', nrows=None):
    """
    Loads the data from CSV and performs some basic cleaning. Essentially the
    cleaning removes corrupted lines.
    """
    # Load the data
    df = pd.read_csv(datafile, sep=';', encoding='UTF-16', error_bad_lines=False, nrows=nrows)
    print('Dropping articles with NaN content...')
    start_n = df.shape[0] 
    # df = df[df['Title']!='test']
    df = df[df['ArticleContents'].apply(type) == str]
    end_n = df.shape[0]
    print('Dropped {} articles with NaN content'.format(start_n-end_n))
    # print('Keeping only Børsen articles...')
    # start_n = df.shape[0] 
    # # df = df[df['Title']!='test']
    # df = df[df['Supplier'] == 'Børsen']
    # end_n = df.shape[0]
    # print('Dropped {} articles not in Børsen'.format(start_n-end_n))
    print('Dropping articles in pleasure section...')
    start_n = df.shape[0] 
    # df = df[df['Title']!='test']
    df = df[df['SectionWebSite'] != 'pleasure']
    end_n = df.shape[0]
    print('Dropped {} articles in pleasure website section'.format(start_n-end_n))
    print('Dropping articles with SectionName not being string...')
    start_n = df.shape[0] 
    # df = df[df['Title']!='test']
    df = df[df['SectionName'].apply(type) == str]
    df = df[df['SectionName'].str.isnumeric() == False]
    end_n = df.shape[0]
    print('Dropped {} articles with ID as SectionName'.format(start_n-end_n))

    return df

def plot_var_overtime(fig, ax, df, varname, frequency='W', smoothing = 3, aggr = 'mean', **kwargs):


    toplot = df[[varname, 'ArticleDateCreated']].groupby(
            [pd.Grouper(key='ArticleDateCreated', freq=frequency)]
            ).agg(['mean', 'count']).reset_index()

    # ax.scatter(toplot['ArticleDateCreated'], toplot[varname, aggr], s=0.5, **kwargs)
    ax.plot(toplot['ArticleDateCreated'].loc[smoothing-1:], smooth(toplot[varname, aggr], smoothing), lw=2, **kwargs)
    
    fig, ax = _format_time_ticks(fig, ax)
    
    return fig, ax

def plot_index(
    df, index_name, frequency = None, smoothing = None, plot_gdp = False, export_index = False, split=None, refactor = True
     ):
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

    toplot = smooth(plot[index_name, 'mean'], smoothing)

    if refactor:
        max_0 = toplot.max()
        min_0 = toplot.min()
        max_w = 3
        min_w = -2.5
        scale = (max_w - min_w)/(max_0 - min_0)
        toplot = toplot*scale - max_0*scale + max_w

    fig, ax = plt.subplots(figsize=(14,6))
    # ax.scatter(plot['ArticleDateCreated'], plot[index_name, 'mean'], s=plot[index_name, 'count']/800) 
    index_line, = ax.plot(plot['ArticleDateCreated'].loc[smoothing-1:], toplot, lw=2, label='Newspaper index')
    ax.legend([index_line], ['Newspaper index'], frameon=False)

    if plot_gdp:
        # Load gdp data
        gdp = pd.read_csv('data/gdp3.csv', usecols=[2,3], names=['quarter', 'growth'])
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
        return fig, ax, index
    else:
        return fig, ax

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

def smooth(y, smoothing_points):
    box = np.ones(smoothing_points)/smoothing_points
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth

def enrich_text(series):
    """
    Takes a series and returns lists of sentiment
    """
    with Pool() as pool:
        afinn_scores = pool.map(afinn.score, series.values.tolist())
        bloom_scores = pool.map(bloom_measure, series.values.tolist())
    
    return afinn_scores, bloom_scores