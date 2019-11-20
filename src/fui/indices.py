import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'D:\\projects\\FUI\\src')
sys.path.insert(1, 'D:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
import pickle
import json
import glob
import re
import copy
import codecs
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from gensim.models import KeyedVectors
from cycler import cycler
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool
from functools import partial
from matplotlib import pyplot as plt

#local imports
from fui.cluster import ClusterTree
from fui.utils import dump_pickle, dump_csv, params

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator((1, 4, 7, 10))
years_fmt = mdates.DateFormatter('%Y')

class BaseIndexer():
#    def __init__(self, name, start_year=2000, end_year=2019, f='M'):
#        self.name = name
#        self.start_year = start_year
#        self.end_year = end_year
#        self.f = f
#        self.corr = {}
#        if end_year == 2019:
#            self.end_str = '-05-31'
#        else:
#            self.end_str = ''
    def __init__(self,name):
        self.name = name
        
    def validate(self):
        v1x, vix = self.load_vix(f='M')
        data = pd.read_csv(params().paths['input']+'validation.csv', header=0)
        data['date'] = pd.to_datetime(data['date'], format="%Ym%m") + pd.tseries.offsets.MonthEnd(1)
        data = data.set_index('date')

        idx = self.idx.filter(regex='_norm', axis=1)
        
        for var in data.columns:
            data[var] = _normalize(data[var])
            self.corr[var] = \
                (_calc_corr(idx, data[var]),
                 _calc_corr(v1x, data[var]),
                 _calc_corr(vix, data[var]))
        return self.corr
    
    def load_vix(self,f='M'):
        v1x = pd.read_csv(params().paths['input']+'v1x_monthly.csv', 
                          names=['date','v1x'], header=0)
    
        v1x['date'] = pd.to_datetime(v1x['date'])
        v1x.set_index('date', inplace=True)
        v1x = v1x[str(self.start_year):str(self.end_year)+self.end_str]
        v1x['v1x'] = _normalize(v1x['v1x'])
        
        vix = pd.read_csv(params().paths['input']+'vixcurrent.csv',
                          names=['date','vix'], header=0)
        vix['date'] = pd.to_datetime(vix['date'])
        vix.set_index('date', inplace=True)
        vix = vix.resample(f).last()
        vix.columns = vix.columns.get_level_values(0)
        vix = vix[str(self.start_year):str(self.end_year)+self.end_str]
        vix['vix'] = _normalize(vix['vix'])
                   
        return v1x, vix

    def parse_topic_labels(self,name):
        """
        reads hand labeled topics from json file.
        
        """
        label_path = os.path.join(params().paths['topic_labels'], 
                                  name+str(self.num_topics)+'.json')
          
        with codecs.open(label_path, 'r', encoding='utf-8-sig') as f:
            self.labels = json.load(f)
        return self.labels
          
    def uncertainty_count(self, dict_name='uncertainty', extend=True):
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
            dump_pickle(out_path,filename+yearlist[i]+'.pkl',
                        df[['article_id','u_count','ArticleDateCreated','word_count']])

    def aggregate(self, df, col='idx', norm=True, write_csv=True, method='mean'):
        """
        aggregates to means within 
        each aggregation frequency
        """    

        idx = df[[col, 'ArticleDateCreated']].groupby(
            [pd.Grouper(key='ArticleDateCreated', freq=self.f)]
        ).agg([method]).reset_index()
        
        idx.set_index('ArticleDateCreated', inplace=True)
        idx.index = idx.index.rename('date')
        idx = idx[str(self.start_year):str(self.end_year)+self.end_str]   

        if norm:
            #normalize to mean = 0, std = 1
            #idx.columns = idx.columns.get_level_values(0)
            idx[(col+'_norm')] = _normalize(idx[col])
            
        #dump_csv(folder_path,var+'_score_'+f+'.csv',idx)
        if write_csv:
            dump_csv(params().paths['indices'], self.name+'_'+self.f, idx, verbose=False)
        return idx

    def plot_index(self, plot_vix=True, annotate=True, title=None):
        """
        """
        out_path = params().paths['indices']
        idx_col = 'idx_norm'
        
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
        
        v1x, vix = self.load_vix(self.f)
                
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(self.idx.index, self.idx[idx_col], label='Børsen Uncertainty Index')
        if title:
            ax.title.set_text(title)
        if plot_vix:
            ax.plot(v1x.index, v1x.v1x, label='VDAX-NEW')
            ax.plot(vix.index, vix.vix, label='VIX')
        ax.legend(frameon=False, loc='upper left')    
    
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        
        if annotate:
            ax.axvspan(xmin=datetime(2000,1,31), xmax=datetime(2000,5,31), 
                       color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate("Dot com \n crash", xy=(datetime(2000,3,31), 0.8),  
                        xycoords=('data', 'axes fraction'), fontsize='x-small', ha='center')
            ax.axvspan(xmin=datetime(2011,3,1), xmax=datetime(2012,11,30), 
                       color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate("Debt crisis", xy=(datetime(2012,2,15), 0.97),  
                        xycoords=('data', 'axes fraction'), fontsize='x-small', ha='center')
            ax.axvspan(xmin=datetime(2018,3,1), xmax=datetime(2019,5,31), 
                       color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate("Trade war", xy=(datetime(2018,11,15), 0.97), 
                        xycoords=('data', 'axes fraction'), fontsize='x-small', ha='center')
    
            dates_dict = {'Euro \nreferendum': '2000-09-28',
                          '9/11':'2001-09-11', 
                          '2001\n Election': '2001-11-20',
                          'Invasion of Iraq': '2003-03-19',
                          '2005\nElection': '2005-02-08',     
                          'Northern Rock\n bank run': '2007-09-14',
                          '2007\n Election': '2007-11-13',
                          'Lehman Brothers': '2008-09-15', 
                          '2010 Flash Crash': '2010-05-06',
                          '2011 Election': '2011-09-15',
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
            
            heights = [0.15,0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 
                       0.97, 0.9, 0.8, 0.7, 0.9, 0.7, 0.95, 0.8,
                       0.97, 0.9, 0.7, 0.9, 0.8]
    
            for l, d, h in zip(dates_dict.keys(), dates_dict.values(), heights):
                d = datetime.strptime(d, "%Y-%m-%d")
                ax.axvline(x=d, color=(102/255, 102/255, 102/255), alpha=0.3)
                ax.annotate(l, xy=(d, h),  xycoords=('data', 'axes fraction'), 
                            fontsize='x-small', ha='center')
            #corr = _calc_corr(vix,idx[idx_name])
            #ax.text(0.80, 0.95, 'Correlation with VIX: %.2f' % round(corr,2) , transform=ax.transAxes)
        
        plt.show()
        fig.savefig(f'{out_path}{self.name}_plot.png', dpi=300)
        return fig, ax
    
class LDAIndexer(BaseIndexer):    
    def build(self, labels=None, num_topics=80, start_year=2000, end_year=2019, f='M', 
              df=None, sample_size=0, topics=None, xsection=None, topic_thold=0.0, 
              xsection_thold=0.05, main_topic=True, extend_u_dict=True, u_weight=False):
        
        self.start_year = start_year
        self.end_year = end_year
        self.f = f
        self.num_topics = num_topics
        if labels:
            self.label_dict = self.parse_topic_labels(labels)
        if end_year == 2019:
            self.end_str = '-05-31'
        else:
            self.end_str = ''

        assert topics or xsection, "Must provide a list or dict of topics."
        assert topic_thold >= 0.0 and xsection_thold >= 0.0, "No negative thresholds."
        
        if df is None: 
            df = merge_lda_u(extend_u_dict,sample_size,self.num_topics)
            
        
        if topics:
            if isinstance(topics, int):
                topics = [topics]
            assert isinstance(topics, list) , "Keyword topics must be list or int."
            if isinstance(topics[0], str):
                topics = self.label_dict[topics[0]]
            
            self.topics = topics
            df['topics'] = df['topics'].apply(
                    lambda x : [j for i,j in enumerate(x) if i in topics])
             
        if xsection:
            assert isinstance(xsection, dict) or isinstance(xsection, list), \
            "intersection_dict must be dict or list of dict keys"
            if isinstance(xsection, list):
                xsection = dict((k, self.label_dict[k]) for k in xsection)
                
            cat = []
            for k, t in xsection.items():
                if isinstance(t, int):
                    t = [t]
                assert isinstance(t, list) , "Values in intersection dict must be int or list."
                df[str(k)] = df['topics'].apply(
                        lambda x : np.array([x[i] for i in t if x[i] >= topic_thold]).sum())
                cat.append(str(k))
            df['idx'] = df.loc[:, cat].prod(axis=1)*\
                        (df[cat] > xsection_thold).all(1).astype(int) 
        
            if u_weight:
                df['idx'] = df['idx']*(df['u_count']/df['word_count'])
        
            self.idx = self.aggregate(df)
            return self.idx
        
        if main_topic:
            df['idx'] = df['topics'].apply(
                lambda x : bool(np.argmax(x) in set(topics))*1)
        else:
            df['idx'] = df['topics'].apply(
                        lambda x : np.array([x[i] for i in topics if x[i] >= topic_thold]).sum())
        
        if u_weight:
            df['idx'] = df['idx']*(df['u_count']/df['word_count'])
        
        self.idx = self.aggregate(df)
        return self.idx

class BloomIndexer(BaseIndexer):
    def build(self, logic, bloom_dict_name,
              start_year=2000, end_year=2019, f='M', u_weight=False, extend=True):
        self.start_year = start_year
        self.end_year = end_year
        if self.end_year == 2019:
            self.end_str = '-05-31'
        else:
            self.end_str = ''
        
        """
        Finds for articles containing words in bloom dictionary. Saves result to disk.
        args:
        params: dict of input_params
        dict_name: name of bloom dict in params
        logic: matching criteria in params
        """
        out_path = params().paths['indices']+self.name+'\\'+logic
        if not os.path.exists(out_path):
            os.makedirs(out_path)        
        
        # check if pickles exist
        pickles = glob.glob(out_path+'\\*.pkl') 
        if len(pickles) is not end_year-start_year+1:
            print("Pickles not found, creating index files...")
            if extend:
                bloom_dict = extend_dict_w2v(bloom_dict_name, n_words=10)
            else:
                bloom_dict = params().dicts[bloom_dict_name]
            
            b_E, b_P, b_U = _get_bloom_sets(bloom_dict)
            print('\n\nEconomic words: ' + repr(b_E) +
                  '\n\n Political words: ' + repr(b_P) +
                  '\n\n Uncertainty words: ' + repr(b_U))
            
            #get parsed articles
            filelist = glob.glob(params().paths['parsed_news']+'boersen*.pkl') 
            filelist = [(fl,int(fl[-8:-4])) for fl in filelist 
                        if int(fl[-8:-4]) >= start_year and int(fl[-8:-4]) <= end_year]
            
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
                if not u_weight:
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
                    dump_pickle(out_path,'bloom'+str(f[1])+'.pkl', df[['article_id', 'idx', 'ArticleDateCreated']])
                    df_out = df_out.append(df[['article_id', 'idx', 'ArticleDateCreated']])    
                else:
                    logic_str = params().options['bloom_logic_weighted']
                    df_u = _load_u_count(year=f[1])
                    df = df.merge(df_u[['u_count','article_id']], how='left', on='article_id')
                    with Pool() as pool:
                        df['idx'] = pool.map(partial(_bloom_compare, 
                                                     logic=logic_str, 
                                                     bloom_E=b_E, 
                                                     bloom_P=b_P, 
                                                     bloom_U=b_U), 
                                                     df['body_stemmed'].values.tolist())
                    
                    df['idx'] = df['idx']*(df['u_count']/df['word_count'])
                    dump_pickle(out_path, 'bloom'+str(f[1])+'_weighted.pkl', df[['article_id','idx','u_count','ArticleDateCreated']])
                    df_out = df_out.append(df[['article_id', 'idx', 'u_count', 'ArticleDateCreated']])    
            
        else:
            print("Loading pickled index files...")
            df_out = pd.DataFrame()
            for fl in pickles:
                with open(fl, 'rb') as data:
                    df = pickle.load(data)
                    df_out = df_out.append(df)
                    
        self.idx = self.aggregate(df_out)
        return self.idx
    

def _calc_corr(df1,df2):
    df1 = df1.join(df2, how='inner', on='date')
    corr_mat = pd.np.corrcoef(df1.iloc[:,0].tolist(), df1.iloc[:,1].tolist())
    return corr_mat[0,1]

def _bloom_compare(word_list, logic, bloom_E, bloom_P, bloom_U):  
    stem_set = set(word_list)
    return eval(logic)

def _get_bloom_sets(bloom_dict):
    b_E = set(bloom_dict['economic'])
    b_P = set(bloom_dict['political'])
    b_U = set(bloom_dict['uncertainty'])
    return b_E, b_P, b_U

def _load_doc_topics():
    with open(params().paths['doc_topics']+'document_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    return df

def doc_topics_to_feather():
    df = _load_doc_topics()
    df2 = pd.DataFrame(df.topics.values.tolist(), index = df.index)
    df = df[['article_id', 'ArticleDateCreated']].merge(df2, left_index=True, right_index=True)
    del(df2)
    
def _normalize(series):
    return (series-series.mean())/series.std()

def _count(word_list, word_set):
    count = 0
    for word in word_list:
        if word in word_set:
            count += 1
    return count

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
    pat = re.compile(r'(((?![\d])\w)+)', re.UNICODE)

    text = text.lower()
    list_to_stem = []
    list_to_stem = [match.group() for match in pat.finditer(text)]
    
    stemmed_list = [stemmer.stem(word) for word in list_to_stem if len(word) >= min_len and len(word) <= max_len]
    return stemmed_list

def _load_u_count(sample_size=0,year='all',extend=True):
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

def merge_lda_u(extend=True,sample_size=0,num_topics=80):
    if extend:
        suffix='u_count_extend'
    else:
        suffix='u_count'
    try:
        df = pd.read_hdf(params().paths['doc_topics']+'doc_topics_'+suffix+'.h5', 'table')
        
        #convert columns to single col list
        df['topics']= df.iloc[:,0:num_topics].values.tolist()
        df.drop(df.columns[0:num_topics], axis=1, inplace=True)
        if sample_size > 0:
            return df.sample(sample_size) 
        return df
    except FileNotFoundError:
        print('File not found, merging lda topics and uncertainty counts...')
        df_u = _load_u_count(extend=extend,sample_size=sample_size)
        df = _load_doc_topics()
    
        df = df.merge(df_u, 'inner', 'article_id')
        df.drop(columns='ArticleDateCreated_y',inplace=True)
        df.rename({'ArticleDateCreated_x':'ArticleDateCreated'},axis=1,inplace=True)
        
        #convert list to columns for hdf compatibility
        df2 = pd.DataFrame(df.topics.values.tolist(), index = df.index)
        df = pd.concat([df2, df[['article_id', 'ArticleDateCreated', 'u_count', 'word_count']]], axis=1)
        del(df2)
        df.to_hdf(params().paths['doc_topics']+'doc_topics_'+suffix+'.h5', 'table', format='table', mode='w', append=False)
        if sample_size > 0:
            return df.sample(sample_size) 
        return df

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


if __name__ == '__main__':
        
    
    ecb = LDAIndexer(name='ecb')
    ecb.build(num_topics=80,labels='meta_topics',sample_size=20000,topics=['EP'],main_topic=True)
    
    xidx = LDAIndexer(name='xidx')
    xidx.build(num_topics=80,labels='meta_topics',sample_size=20000,xsection=['EP','F'],u_weight=True)

    bloom = BloomIndexer(name='bloom')
    bloom.build(logic='EandPandU',bloom_dict_name='bloom_extended',extend=True)