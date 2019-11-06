import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'D:\\projects\\FUI')
sys.path.insert(1, 'D:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
import pickle
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import copy
from gensim.models import KeyedVectors
import codecs
import json
import glob
import re
import gensim
import pandas as pd
from multiprocessing import Pool
from functools import partial

from src.fui.cluster import ClusterTree
from src.fui.utils import main_directory
from src.fui.utils import dump_pickle

def parse_topic_labels(num_topics,params):
    """
    reads hand labeled topics from json file.
    
    """
    label_path = os.path.join(params['paths']['root'],params['paths']['topic_labels'], 
                        'epu'+str(num_topics)+'.json')
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels             
            
def load_model(lda_instance, num_topics, params):
    try:
        folder_path = os.path.join(params['paths']['root'],params['paths']['lda'], 'lda_model_' + str(num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        lda_instance.lda_model = gensim.models.LdaMulticore.load(file_path)
        print("LDA-model with {} topics loaded".format(num_topics))
    except FileNotFoundError:
        print("Error: LDA-model not found")
        lda_instance.lda_model = None

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
    dict_out = copy.deepcopy(params[dict_name])
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

def _stemtext(text, min_len=2, max_len=25):
    # Remove any non-alphabetic character, split by space
    stemmer = SnowballStemmer("danish")
    pat = re.compile('(((?![\d])\w)+)', re.UNICODE)

    text = text.lower()
    list_to_stem = []
    list_to_stem = [match.group() for match in pat.finditer(text)]
    
    stemmed_list = [stemmer.stem(word) for word in list_to_stem if len(word) >= min_len and len(word) <= max_len]
    return stemmed_list

def uncertainty_count(params, dict_name='uncertainty', extend=True):
    """
    Finds for articles containing words in bloom dictionary. Saves result to disk.
    args:
    dict_name: name of bloom dict in params
    logic: matching criteria in params
    """
 
    out_path = params['paths']['root']+params['paths']['parsed_news']
    if not os.path.exists(out_path):
        os.makedirs(out_path)        
    
    if extend:
        U_set = set(list(extend_dict_w2v(dict_name, params, n_words=10).values())[0])
    else:
        U_set = set(list(params[dict_name].values())[0])

    #get parsed articles
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'boersen*.pkl') 
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
    
def load_parsed_data(params, sample_size=None):
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'u_count*.pkl') 
    df = pd.DataFrame()
    for f in filelist:    
        with open(f, 'rb') as f_in:
            df_n = pickle.load(f_in)
            df = df.append(df_n)
    if sample_size is not None:
        return df.sample(sample_size)
    else:
        return df
    
def merge_lda_u(params):
    df_u = load_parsed_data(params)
    with open(params['paths']['root']+
              params['paths']['lda']+'\\document_topics\\document_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    
    label_dict = parse_topic_labels(80,params)
    labels = pd.DataFrame.from_dict(label_dict,orient='index',columns=['cat','region'])
    labels['topic'] = labels.index.astype('int64')
        
    df = df.merge(df_u, 'inner', 'article_id')
    df.drop(columns='ArticleDateCreated_y',inplace=True)
    df.rename({'ArticleDateCreated_x':'ArticleDateCreated'},axis=1,inplace=True)

    return df

def _aggregate(df, col, aggregation=['M'], normalize=True):
    """
    aggregates to means within 
    each aggregation frequency
    """

    agg_dict = {}
    for f in aggregation:
        idx = df[[col, 'ArticleDateCreated']].groupby(
            [pd.Grouper(key='ArticleDateCreated', freq=f)]
        ).agg(['mean']).reset_index()
        
        idx.set_index('ArticleDateCreated', inplace=True)

        if normalize:
            #normalize to mean = 0, std = 1
            idx.columns = idx.columns.get_level_values(0)
            idx[(col+'_norm')]=(idx[col]-idx[col].mean())/idx[col].std()
            
        #dump_csv(folder_path,var+'_score_'+f+'.csv',idx)
        agg_dict[f] = idx
    return agg_dict

def ECB_index(params,df,cat=['P'],num_topics=80,use_weights=False):
    label_dict = parse_topic_labels(80,params)
    labels = pd.DataFrame.from_dict(label_dict,orient='index',columns=['cat','region'])
    labels['topic'] = labels.index.astype('int64')
    
    if not use_weights:
        df = df.merge(right=labels, how='left', left_on='max_topic',right_on='topic')
        df['max_topic'] = df['topics'].apply(lambda x: np.argmax(x))
        df['ECB'] = (df['cat'].isin(cat)) * 1
        idx = _aggregate(df,'ECB')
        return idx
    
    #TODO:
    else: 
        pass
        #df['topics'].apply()
    
    
    
if __name__ == '__main__':
 
    os.chdir(main_directory())
    
    PARAMS_PATH = 'scripts/input_params.json'
    with codecs.open(PARAMS_PATH, 'r', 'utf-8-sig') as json_file:  
        params = json.load(json_file)
    
    uncertainty_count(params)
    #U_set = set(dict_out)
#    df = merge_lda_u(params)
#    df2 = df.sample(1000)
    #idx = ECB_index(params,df)
    