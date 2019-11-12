import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'D:\\projects\\FUI')
sys.path.insert(1, 'D:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
import pickle
import numpy as np
import json
import glob
import re
import gensim
import copy
import pandas as pd
import matplotlib.dates as mdates
#import matplotlib.patches as patches

from gensim.models import KeyedVectors
from cycler import cycler
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool
from functools import partial
from matplotlib import pyplot as plt
from fui.cluster import ClusterTree
from fui.utils import main_directory, dump_pickle, dump_csv, params

def validation(idx,start_year=2000,end_year=2019):
    if end_year == 2019:
        end_str = '-05-31'
    else:
        end_str = ''

    v1x, vix = load_vix(f='M',start_year=start_year,end_year=end_year,end_str=end_str)

    data = pd.read_csv(params().paths['input']+'validation.csv',header=0)
    data['date'] = pd.to_datetime(data['date'],format="%Ym%m") + pd.tseries.offsets.MonthEnd(1)
    data = data.set_index('date')

    idx = idx.filter(regex='_norm',axis=1)
    
    corr = {}
    for var in data.columns:
        data[var] = normalize(data[var])
        corr[var] = (_calc_corr(idx,data[var]),
            _calc_corr(v1x,data[var]),
            _calc_corr(vix,data[var]))
    return corr

def extend_dict_w2v(dict_name, n_words=10):
    """
    Extends bloom dictionary with similar words using a pre-trained
    embedding. Default model: https://fasttext.cc/docs/en/crawl-vectors.html
    args:
    params: input_params.json
    dict_name: name of Bloom dict in params
    n_words: include n_nearest words to subject word.
    """
    
    model = KeyedVectors.load_word2vec_format(params().paths['w2v_model'], binary=False)
    print("Model loaded")
    dict_out = copy.deepcopy(params().dicts[dict_name])
    for k, v in params().dicts[dict_name].items():
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

def _stemtext(text, min_len=2, max_len=25):
    # Remove any non-alphabetic character, split by space
    stemmer = SnowballStemmer("danish")
    pat = re.compile('(((?![\d])\w)+)', re.UNICODE)

    text = text.lower()
    list_to_stem = []
    list_to_stem = [match.group() for match in pat.finditer(text)]
    
    stemmed_list = [stemmer.stem(word) for word in list_to_stem if len(word) >= min_len and len(word) <= max_len]
    return stemmed_list

def parse_topic_labels(num_topics):
    """
    reads hand labeled topics from json file.
    
    """
    label_path = os.path.join(params().paths['topic_labels'], 
                        'epu'+str(num_topics)+'.json')
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels  

def load_vix(f='M',start_year=2000,end_year=2019,end_str=''):
    v1x = pd.read_csv(params().paths['input']+'v1x_monthly.csv', 
                      names=['date','v1x'], header=0)

    v1x['date'] = pd.to_datetime(v1x['date'])
    v1x.set_index('date', inplace=True)
    v1x = v1x[str(start_year):str(end_year)+end_str]   
    v1x['v1x'] = normalize(v1x['v1x'])
    
    vix = pd.read_csv(params().paths['input']+'vixcurrent.csv', 
                      names=['date','vix'], header=0)
    vix['date'] = pd.to_datetime(vix['date'])
    vix.set_index('date', inplace=True)
    vix = vix.resample(f).last()
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[str(start_year):str(end_year)+end_str]   
    vix['vix'] = normalize(vix['vix'])
               
    return v1x, vix
            
def load_model(lda_instance, num_topics):
    try:
        folder_path = os.path.join(params().paths['root'],params().paths['lda'], 'lda_model_' + str(num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        lda_instance.lda_model = gensim.models.LdaMulticore.load(file_path)
        print("LDA-model with {} topics loaded".format(num_topics))
    except FileNotFoundError:
        print("Error: LDA-model not found")
        lda_instance.lda_model = None
          
def uncertainty_count(dict_name='uncertainty', extend=True):
    """
    Finds for articles containing words in bloom dictionary. Saves result to disk.
    args:
    dict_name: name of bloom dict in params
    logic: matching criteria in params
    """
 
    out_path = params().paths['parsed_news']
 
    if extend:
        U_set = set(list(extend_dict_w2v(dict_name, n_words=10).values())[0])
        filename = 'u_count_extend'
    else:
        U_set = set(list(params().dicts[dict_name].values())[0])
        filename = 'u_count'

    #get parsed articles
    filelist = glob.glob(params().paths['parsed_news']+'boersen*.pkl') 
    yearlist = [f[-8:-4] for f in filelist]

    for (i,f) in enumerate(filelist):
        print(f"Processing year {yearlist[i]}.")
        with open(f, 'rb') as data:
            try:
                df = pickle.load(data)
            except TypeError:
                print("Parsed news is not a valid pickle!")
        
        #stem articles
        with Pool(4) as pool:
            df['body_stemmed'] = pool.map(_stemtext,
                                          df['ArticleContents'].values.tolist())
        
        #compare to dictionary
        with Pool(4) as pool:
            df['u_count'] = pool.map(partial(_count, 
                                          word_set=U_set), 
                                          df['body_stemmed'].values.tolist())
        
        #save to disk
        dump_pickle(out_path,filename+yearlist[i]+'.pkl',df[['article_id','u_count','ArticleDateCreated','word_count']])

def _count(word_list, word_set):
    count = 0
    for word in word_list:
        if word in word_set:
            count += 1
    return count
    
def load_u_count(sample_size=0,year='all',extend=True):
    if extend:
        filename='u_count_extend'
    else:
        filename='u_count'
    if year is not 'all':
            file_path = params().paths['parsed_news']+filename+str(year)+'.pkl'
            with open(file_path, 'rb') as f_in:
                df = pickle.load(f_in)
            return df
    else:
        filelist = glob.glob(params().paths['parsed_news']+filename+'*.pkl') 
        df = pd.DataFrame()
        for f in filelist:    
            with open(f, 'rb') as f_in:
                df_n = pickle.load(f_in)
                df = df.append(df_n)    
        if sample_size > 0:
            return df.sample(sample_size)
        else:
            return df
    
def merge_lda_u(extend=True):
    if extend:
        suffix='u_count_extend'
    else:
        suffix='u_count'
    try:
        with open(params().paths['doc_topics']+'doc_topics_'+suffix+'.pkl', 'rb') as f:
            df = pickle.load(f)
        return df
    except FileNotFoundError:
        df_u = load_u_count(extend=extend)
        with open(params().paths['doc_topics']+'document_topics.pkl', 'rb') as f:
            df = pickle.load(f)
    
        df = df.merge(df_u, 'inner', 'article_id')
        df.drop(columns='ArticleDateCreated_y',inplace=True)
        df.rename({'ArticleDateCreated_x':'ArticleDateCreated'},axis=1,inplace=True)
        dump_pickle(params().paths['doc_topics'],'doc_topics_'+suffix+'.pkl',df)
        return df

def load_doc_topics():
    with open(params().paths['doc_topics']+'document_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    return df
    
def _aggregate(df, name, col='idx', aggregation=['M'], norm=True,
               start_year=2000, end_year=2019, write_csv=True):
    """
    aggregates to means within 
    each aggregation frequency
    """
    if end_year == 2019:
        end_str = '-05-31'
    else:
        end_str = ''

    agg_dict = {}
    for f in aggregation:
        idx = df[[col, 'ArticleDateCreated']].groupby(
            [pd.Grouper(key='ArticleDateCreated', freq=f)]
        ).agg(['mean']).reset_index()
        
        idx.set_index('ArticleDateCreated', inplace=True)
        idx.index = idx.index.rename('date')
        idx = idx[str(start_year):str(end_year)+end_str]   

        if norm:
            #normalize to mean = 0, std = 1
            #idx.columns = idx.columns.get_level_values(0)
            idx[(col+'_norm')]=normalize(idx[col])
            
        #dump_csv(folder_path,var+'_score_'+f+'.csv',idx)
        agg_dict[f] = idx
        if write_csv:
            dump_csv(params().paths['indices'], name+'_'+f, idx, verbose=False)
    if len(aggregation) == 1:
        return agg_dict[aggregation[0]]
    else:
        return agg_dict
    
def plot_index(
    out_path, idx, name, plot_vix=True, 
    start_year=2000, end_year=2019, annotate=True):
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
    idx_col = 'idx_norm'
    
    v1x, vix = load_vix(f='M',start_year=start_year,end_year=end_year,end_str=end_str)
    
    idx = idx[str(start_year):str(end_year)+end_str]
    
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(idx.index, idx[idx_col], label='Børsen Uncertainty Index')
    if plot_vix:
        ax.plot(v1x.index, v1x.v1x, label='VDAX-NEW')
        ax.plot(vix.index, vix.vix, label='VIX')
    ax.legend(frameon=False, loc='upper left')    

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    
    if annotate:
        ax.axvspan(xmin=datetime(2000,1,31), xmax=datetime(2000,5,31), color=(102/255, 102/255, 102/255), alpha=0.3)
        ax.annotate("Dot com \n crash", xy=(datetime(2000,3,31), 0.8),  xycoords=('data', 'axes fraction'), fontsize='x-small', ha='center')
        ax.axvspan(xmin=datetime(2011,3,1), xmax=datetime(2012,11,30), color=(102/255, 102/255, 102/255), alpha=0.3)
        ax.annotate("Debt crisis", xy=(datetime(2012,2,15), 0.97),  xycoords=('data', 'axes fraction'), fontsize='x-small', ha='center')
        ax.axvspan(xmin=datetime(2018,3,1), xmax=datetime(2019,5,31), color=(102/255, 102/255, 102/255), alpha=0.3)
        ax.annotate("Trade war", xy=(datetime(2018,11,15), 0.97),  xycoords=('data', 'axes fraction'), fontsize='x-small', ha='center')

        dates_dict = {'9/11':'2001-09-11', 
                      '2001\n Election': '2001-11-20',
                      'Invasion of Iraq': '2003-03-19',
                      '2003\nElection': '2005-02-08',     
                      'Northern Rock\n bank run': '2007-09-14',
                      '2007\n Election': '2007-11-13',
                      'Lehman Brothers': '2008-09-15', 
                      '2010 Flash Crash': '2010-05-06',
                      '"Whatever\n it takes"': '2012-07-26', 
                      '2013 US Gov\n shutdown': '2013-10-15', 
                      'DKK pressure\n crisis': '2015-02-15',
                      '2015\n Election': '2015-06-18',
                      'Migrant\n crisis': '2015-09-15',
                      'Brexit': '2016-06-23',
                      'US\n Election': '2016-11-08',
                      'Labor parties\n agreement': '2018-04-15',
                      'Danke Bank\n money laundering': '2018-09-15',
                      '2018 US Gov\n shutdown': '2018-12-10'}
        
        heights = [0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 
                   0.97, 0.9, 0.7, 0.9, 0.7, 0.95, 0.8,
                   0.97, 0.9, 0.7, 0.9, 0.8]

        for l, d, h in zip(dates_dict.keys(), dates_dict.values(), heights):
            d = datetime.strptime(d, "%Y-%m-%d")
            ax.axvline(x=d, color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate(l, xy=(d, h),  xycoords=('data', 'axes fraction'), 
                        fontsize='x-small', ha='center')
        #corr = _calc_corr(vix,idx[idx_name])
        #ax.text(0.80, 0.95, 'Correlation with VIX: %.2f' % round(corr,2) , transform=ax.transAxes)
    
    plt.show()
    fig.savefig(f'{out_path}{name}_plot.png', dpi=300)
    return fig, ax

def ECB_index(name, cat=['P','F'], num_topics=80, use_weights=False, extend_u=True,
              threshold=0.0, req_all=False, start_year=2000, end_year=2019):
    
    df = merge_lda_u(extend_u)
    
    if threshold < 0.0:
        print("No negative thresholds.")
        return 0
    
    label_dict = parse_topic_labels(num_topics)
    labels = pd.DataFrame.from_dict(label_dict,orient='index',columns=['cat','region'])
    labels['topic'] = labels.index.astype('int64')
    labels.fillna(value='N/A', inplace=True)
    
    if not use_weights:
        df['max_topic'] = df['topics'].apply(lambda x: np.argmax(x))
        df['max_topic_cat'] = df['max_topic'].apply(lambda x: labels.cat[labels['topic'] == x].values[0])
        #df = df.merge(right=labels, how='left', left_on='max_topic',right_on='topic')
        if not req_all:
            df['idx'] = df['max_topic_cat'].apply(lambda x : bool(set(x).intersection(set(cat)))*1)
        else:
            df['idx'] = df['max_topic_cat'].apply(lambda x : bool(set(x)==set(cat))*1)
        idx = _aggregate(df,'ECB',name=name,start_year=start_year,end_year=end_year)
        return idx

    else:
        if not req_all:
            topic_idx = labels[labels['cat'].apply(
                    lambda x : bool(set([x]).intersection(set(cat))))
                    ].index.tolist()
        else:
            topic_idx = labels[labels['cat'].apply(
                    lambda x : bool(set([x])==(set(cat))))
                    ].index.tolist()
        df['idx'] = df['topics'].apply(_topic_weights, args=(topic_idx,threshold))
        idx = _aggregate(df,name=name,start_year=start_year,end_year=end_year)
        return idx

def intersection_index(name, cat=['P','F'], num_topics=80, threshold=0.0, extend_u=True,
                       exclude_dk=False, start_year=2000, end_year=2019, u_weight=False):
    df = merge_lda_u(extend_u)
    
    if threshold < 0.0:
        print("No negative thresholds.")
        return 0
    
    label_dict = parse_topic_labels(num_topics)
    labels = pd.DataFrame.from_dict(label_dict,orient='index',columns=['cat','region'])
    labels['topic'] = labels.index.astype('int64')
    labels.fillna(value='N/A', inplace=True)
    
    for c in cat:
        topic_idx = labels[labels['cat'].apply(
                    lambda x : bool(set([x]).intersection(set([c]))))
                    ].index.tolist()
        if exclude_dk:
            region_idx = labels.index.values[labels['region'] != 'DK'].tolist()
        else:
            region_idx = topic_idx
            
        df[c] = df['topics'].apply(_topic_weights, args=(topic_idx,region_idx,0.0))
    #df['in_idx'] = (df[cat] > threshold).all(1).astype(int)

    df['idx'] = df.loc[:, cat].prod(axis=1)*(df[cat] > 0.001).all(1).astype(int) 
    
    if u_weight:
        df['idx'] = df['in_idx']*np.log(df['u_count']+1)
    
    idx = _aggregate(df,name=name,start_year=start_year,end_year=end_year)
    return idx

def bloom_index(bloom_dict, name, logic, start_year=2000, end_year=2019, 
                weighted=False, extend=True):
    """
    Finds for articles containing words in bloom dictionary. Saves result to disk.
    args:
    params: dict of input_params
    dict_name: name of bloom dict in params
    logic: matching criteria in params
    """
    out_path = params().paths['indices']+name+'\\'+logic
    if not os.path.exists(out_path):
        os.makedirs(out_path)        
    
    # check if pickles exist
    pickles = glob.glob(out_path+'\\*.pkl') 
    if len(pickles) is not end_year-start_year+1:
        print("Pickles not found, creating index files...")
        if extend:
            bloom_dict = extend_dict_w2v(bloom_dict, n_words=10)
        else:
            bloom_dict = params().dicts[bloom_dict]
        
        b_E, b_P, b_U = _get_bloom_sets(bloom_dict)
        print('\n\nEconomic words: ' + repr(b_E) +
              '\n\n Political words: ' + repr(b_P) +
              '\n\n Uncertainty words: ' + repr(b_U))
        
        #get parsed articles
        filelist = glob.glob(params().paths['parsed_news']+'boersen*.pkl') 
        filelist = [(f,int(f[-8:-4])) for f in filelist 
                    if int(f[-8:-4]) >= start_year and int(f[-8:-4]) <= end_year]
        
        df_out = pd.DataFrame()
    
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
            if not weighted:
                logic_str = params().options['bloom_logic'][logic]
                print('\nLogic: '+logic_str)
                #compare to dictionary
                with Pool() as pool:
                    df['idx'] = pool.map(partial(_bloom_compare, 
                                                  logic=logic_str, 
                                                  bloom_E=b_E, 
                                                  bloom_P=b_P, 
                                                  bloom_U=b_U), 
                                                  df['body_stemmed'].values.tolist())
                
                #save to disk
                dump_pickle(out_path,'bloom'+str(f[1])+'.pkl',df[['article_id','idx','ArticleDateCreated']])
                df_out = df_out.append(df[['article_id','idx','ArticleDateCreated']])    
            else:
                logic_str = params().options['bloom_logic_weighted']
                df_u = load_u_count(year=f[1])
                df = df.merge(df_u[['u_count','article_id']], how='left', on='article_id')
                with Pool() as pool:
                    df['idx'] = pool.map(partial(_bloom_compare, 
                                                  logic=logic_str, 
                                                  bloom_E=b_E, 
                                                  bloom_P=b_P, 
                                                  bloom_U=b_U), 
                                                  df['body_stemmed'].values.tolist())
                df['idx'] = df['idx']*df['u_count']
                dump_pickle(out_path,'bloom'+str(f[1])+'_weighted.pkl',df[['article_id','idx','u_count','ArticleDateCreated']])
                df_out = df_out.append(df[['article_id','idx','u_count','ArticleDateCreated']])    
        
    else:
        print("Loading pickled index files...")
        df_out = pd.DataFrame()
        for f in pickles:
            with open(f, 'rb') as data:
                 df = pickle.load(data)
                 df_out = df_out.append(df)
                
    idx = _aggregate(df_out, name=name, aggregation=['M'],
                     start_year=start_year, end_year=end_year)
    return idx
    
def _bloom_compare(word_list, logic, bloom_E, bloom_P, bloom_U):  
    stem_set = set(word_list)        
    return eval(logic)

def _get_bloom_sets(bloom_dict):
    b_E = set(bloom_dict['economic'])
    b_P = set(bloom_dict['political'])
    b_U = set(bloom_dict['uncertainty'])
    return b_E, b_P, b_U
     
def _topic_weights(topic_weights,topic_idx,region_idx,threshold):
    if threshold > 0.0:
        psum = np.array(
                [topic_weights[int(i)] for i in topic_idx if topic_weights[int(i)] >= threshold and i in region_idx]).sum()
    else:
        psum = np.array([topic_weights[int(i)] for i in topic_idx if i in region_idx]).sum()
    return psum

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

def _calc_corr(df1,df2):
    df1 = df1.join(df2, how='inner', on='date')
    corr_mat = pd.np.corrcoef(df1.iloc[:,0].tolist(), df1.iloc[:,1].tolist())
    return corr_mat[0,1]
    
if __name__ == '__main__':
 
    uncertainty_count(extend=True)
    uncertainty_count(extend=False)
    
    #idx = ECB_index(params,df,use_weights=True)
    #idx2 = ECB_index(params,df,use_weights=True,threshold=2e-04)
    #test = labels.index[labels['cat'] == 'P'].tolist()
    #idx = ECB_index(params,df,use_weights=True,threshold=2e-04)
    #df2 = df.sample(10000)
    #df = load_doc_topics(params)
    #cl80 = ClusterTree(80,params)
    #cl80.dendrogram()

    idx = intersection_index(name='xsection_uw',extend=True,threshold=0.2,exclude_dk=False,u_weight=True)
    #idx = bloom_index('bloom_extended', name='bl_uw_ext', extend=True, weighted=True, logic='EandP', start_year=2000,end_year=2019)
    val = validation(idx)
    plot_index(params().paths['indices'], idx, 'bl_uw_ext', annotate=True)
#    NB_blue = '#007bd1'
#    fig = plt.figure(figsize=(5, 5))
#    ax = fig.add_subplot(111)
#    ax.scatter(df2['P'], df2['F'], marker='o', c='#007bd1')
#    rect = patches.Rectangle((0.2,0.2),0.5,0.5,linewidth=1,edgecolor='r',facecolor='none')
#    ax.add_patch(rect)
#    plt.show()