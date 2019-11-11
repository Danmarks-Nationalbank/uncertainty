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
import gensim
import pandas as pd
from multiprocessing import Pool
from functools import partial
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
#import matplotlib.patches as patches
from cycler import cycler

from fui.bloom import extend_dict_w2v, _stemtext
from fui.cluster import ClusterTree
from fui.utils import main_directory, dump_pickle, params

def validation(idx,start_year=2000,end_year=2019):
    if end_year == 2019:
        end_str = '-05-31'
    else:
        end_str = ''

    
    data = pd.read_csv(params().paths['input']+'validation.csv',header=0)
    data['date'] = pd.to_datetime(data['date'],format="%Ym%m") + pd.tseries.offsets.MonthEnd(1)
    data = data.set_index('date')
    
    vix = pd.read_csv(params().paths['input']+'v1x_monthly.csv', 
                      names=['date','vix'], header=0)
    vix['date'] = pd.to_datetime(vix['date'])
    vix.set_index('date', inplace=True)
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[str(start_year):str(end_year)+end_str]   
    vix['vix'] = normalize(vix['vix'])
    
    idx = idx.filter(regex='_norm',axis=1)
    
    corr = {}
    for var in data.columns:
        data[var] = normalize(data[var])
        corr[var] = (_calc_corr(idx,data[var]),_calc_corr(vix,data[var]))
    return corr

def parse_topic_labels(num_topics):
    """
    reads hand labeled topics from json file.
    
    """
    label_path = os.path.join(params().paths['topic_labels'], 
                        'epu'+str(num_topics)+'.json')
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels             
            
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
    else:
        U_set = set(list(params().dicts[dict_name].values())[0])

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
        dump_pickle(out_path,'u_count'+yearlist[i]+'.pkl',df[['article_id','u_count','ArticleDateCreated']])

def _count(word_list, word_set):
    count = 0
    for word in word_list:
        if word in word_set:
            count += 1
    return count
    
def load_u_count(sample_size=0):
    filelist = glob.glob(params().paths['parsed_news']+'u_count*.pkl') 
    df = pd.DataFrame()
    for f in filelist:    
        with open(f, 'rb') as f_in:
            df_n = pickle.load(f_in)
            df = df.append(df_n)
    if sample_size > 0:
        return df.sample(sample_size)
    else:
        return df
    
def merge_lda_u():
    df_u = load_u_count()
    with open(params().paths['doc_topics']+'document_topics.pkl', 'rb') as f:
        df = pickle.load(f)

    df = df.merge(df_u, 'inner', 'article_id')
    df.drop(columns='ArticleDateCreated_y',inplace=True)
    df.rename({'ArticleDateCreated_x':'ArticleDateCreated'},axis=1,inplace=True)

    return df

def load_doc_topics():
    with open(params().paths['doc_topics']+'document_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    return df
    
def _aggregate(df, col, aggregation=['M'], normalize=True,
               start_year=2000, end_year=2019):
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

        if normalize:
            #normalize to mean = 0, std = 1
            idx.columns = idx.columns.get_level_values(0)
            idx[(col+'_norm')]=normalize(idx[col])
            
        #dump_csv(folder_path,var+'_score_'+f+'.csv',idx)
        agg_dict[f] = idx
    if len(aggregation) == 1:
        return agg_dict[aggregation[0]]
    else:
        return agg_dict
    
def plot_index(
    out_path, idx, idx_name, plot_vix=True, start_year=2000, end_year=2019):
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
    
    vix = pd.read_csv(params().paths['input']+'v1x_monthly.csv', 
                      names=['date','vix'], header=0)
    vix['date'] = pd.to_datetime(vix['date'])
    vix.set_index('date', inplace=True)
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[str(start_year):str(end_year)+end_str]   
    vix['vix'] = normalize(vix['vix'])
    
    idx = idx[str(start_year):str(end_year)+end_str]
    
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(idx.index, idx[idx_name], label='Børsen Uncertainty Index')
    if plot_vix:
        ax.plot(vix.index, vix.vix, label='VDAX-NEW')
    ax.legend(frameon=False, loc='upper left')    

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    
    
    corr = _calc_corr(vix,idx[idx_name])
    ax.text(0.77, 0.95, 'Correlation: %.2f' % round(corr,2) , transform=ax.transAxes)
    
    plt.show()
    fig.savefig(f'{out_path}\\[idx_name]_plot.png', dpi=300)
    return corr, fig, ax

def ECB_index(df,cat=['P','F'],num_topics=80,use_weights=False,
              threshold=0.0,req_all=False):
    
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
            df['ECB'] = df['max_topic_cat'].apply(lambda x : bool(set(x).intersection(set(cat)))*1)
        else:
            df['ECB'] = df['max_topic_cat'].apply(lambda x : bool(set(x)==set(cat))*1)
        idx = _aggregate(df,'ECB')
        return idx

    else:
        if threshold > 0.0:
            idx_name = 'ECB_W_TH'
        else:
            idx_name = 'ECB_W'
        
        if not req_all:
            topic_idx = labels[labels['cat'].apply(
                    lambda x : bool(set([x]).intersection(set(cat))))
                    ].index.tolist()
        else:
            topic_idx = labels[labels['cat'].apply(
                    lambda x : bool(set([x])==(set(cat))))
                    ].index.tolist()
        df[idx_name] = df['topics'].apply(_topic_weights, args=(topic_idx,threshold))
        idx = _aggregate(df,idx_name)
        return idx

def intersection_index(df,cat=['P','F'],num_topics=80,threshold=0.0,exclude_dk=False, 
                       start_year=2000, end_year=2019, u_weight=False):
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

    df['in_idx'] = df.loc[:, cat].prod(axis=1)*1000*(df[cat] > 0.01).all(1).astype(int) 
    
    if u_weight:
        df['in_idx'] = df['in_idx']*df['u_count']
    
    idx = _aggregate(df,'in_idx',start_year=start_year,end_year=end_year)
    return idx
 
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
 
    #uncertainty_count()
    df = merge_lda_u()
    
    #idx = ECB_index(params,df,use_weights=True)
    #idx2 = ECB_index(params,df,use_weights=True,threshold=2e-04)
    #test = labels.index[labels['cat'] == 'P'].tolist()
    #idx = ECB_index(params,df,use_weights=True,threshold=2e-04)
    #df2 = df.sample(10000)
    #df = load_doc_topics(params)
    #cl80 = ClusterTree(80,params)
    #cl80.dendrogram()

    idx = intersection_index(df,threshold=0.3,exclude_dk=False,u_weight=False)
    val = validation(idx)
    plot_index(params().paths['lda'], idx, 'in_idx_norm')
#    NB_blue = '#007bd1'
#    fig = plt.figure(figsize=(5, 5))
#    ax = fig.add_subplot(111)
#    ax.scatter(df2['P'], df2['F'], marker='o', c='#007bd1')
#    rect = patches.Rectangle((0.2,0.2),0.5,0.5,linewidth=1,edgecolor='r',facecolor='none')
#    ax.add_patch(rect)
#    plt.show()