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
from itertools import cycle, islice, repeat
import codecs
import json
import glob
import re
import gensim
import pandas as pd
from multiprocessing import Pool
from functools import partial
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist

from src.fui.utils import main_directory
from src.fui.utils import dump_pickle

class ClusterTree():
    """
    Build clusters from topic models using scipy.cluster.hierarchy.
    :num_topics: Used to load a pre-trained topic model.
    :metric: and :method: Used for HAC.
    """
    
    def __init__(self, num_topics, metric='jensenshannon', method='ward'):
        """
        Saves linkage matrix :Z: and :nodelist:
        """
        
        self.num_topics = num_topics
        self.metric = metric
        self.method = method
        
        folder_path = os.path.join(params['paths']['root'],params['paths']['lda'], 'lda_model_' + str(self.num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        self.lda_model = gensim.models.LdaMulticore.load(file_path)
        topics = self.lda_model.get_topics()
        y = pdist(topics, metric=self.metric)
        self.Z = hierarchy.linkage(y, method=self.method)
        rootnode, self.nodelist = hierarchy.to_tree(self.Z,rd=True)
    
    def _get_children(self, id):
        """
        Recursively get all children of parent node :id:
        """
        if not self.nodelist[id].is_leaf():
            for child in [self.nodelist[id].get_left(), self.nodelist[id].get_right()]:
                yield child
                for grandchild in self._get_children(child.id):
                    yield grandchild
                    
    def children(self):
        """
        Returns a dict with k, v: parent: [children]. Does not include leaf nodes.
        """
        self.children = {}
        for i in range(self.num_topics,len(self.nodelist)):
            self.children[i] = [child.id for child in self._get_children(i)]
        return self.children
    
    def _colorpicker(self,k):
        """
        Returns an NB color to visually group similar topics in dendrogram
        
        """
        NB_colors = [(0, 123, 209),
            (146, 34, 156),
            (196, 61, 33),
            (223, 147, 55),
            (176, 210, 71)] 
        
        # Get flat clusters for grouping
        self.flat_clusters(n=self.colors)
        clist = list(islice(cycle(NB_colors), len(self.L)))
        for c,i in enumerate(list(self.L)):
            if k in [child.id for child in self._get_children(i)]:
                color = clist[c]
                return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        # Gray is default
        return "#666666"
    
    def dendrogram(self,w=10,h=10,colors=10,no_plot=False):
        """
        Draws dendrogram
        :no_plot: Don's render figure. Use self.graph to render figure later
        :colors: Approx. no of color clusters in figure.
        """
        
        self.colors = colors
        plt.figure(figsize=(w, h))
        plt.title("Topic Dendrogram")
        
        #hierarchy.set_link_color_palette([colors.rgb2hex(rgb) for rgb in NB_colors])
        self.graph = hierarchy.dendrogram(self.Z,
                       orientation='right',
                       #labels=labelList,
                       distance_sort='descending',
                       show_leaf_counts=False,
                       no_plot=no_plot,
                       #color_threshold=2.0*np.max(self.Z[:,2])
                       link_color_func=self._colorpicker)

    
    def flat_clusters(self,n=8,init=1,criterion='maxclust'):
        """
        Returns flat clusters from the linkage matrix :Z:
        """
        if criterion is 'distance':
            self.T = hierarchy.fcluster(self.Z,init,criterion='distance')
            a = 0
            while a < 20:
                if self.T.max() < n:
                    init = init-0.02
                    a += 1
                elif self.T.max() > n:
                    init = init+0.02
                    a += 1
                else:
                    self.L, self.M = hierarchy.leaders(self.Z,self.T)
                    return self.T
                self.T = hierarchy.fcluster(self.Z,init,criterion='distance')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        elif criterion is 'inconsistent':
            self.T = hierarchy.fcluster(self.Z,criterion='inconsistent')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        elif criterion is 'maxclust':
            self.T = hierarchy.fcluster(self.Z,t=n,criterion='maxclust')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        else:
            print('Criteria not implemented')
            return 0
            
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
    print("stemming!")
    list_to_stem = [match.group() for match in pat.finditer(text)]
    
    stemmed_list = [stemmer.stem(word) for word in list_to_stem if len(word) >= min_len and len(word) <= max_len]
    return stemmed_list

def uncertainty_count(params, dict_name):
    """
    Finds for articles containing words in bloom dictionary. Saves result to disk.
    args:
    params: dict of input_params
    dict_name: name of bloom dict in params
    logic: matching criteria in params
    """
 
    out_path = params['paths']['root']+params['paths']['parsed_news']
    if not os.path.exists(out_path):
        os.makedirs(out_path)        
    
    U_set = set(params[dict_name]['uncertainty'])
    
    #get parsed articles
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'boersen*.pkl') 

    for (i,f) in enumerate(filelist):
        with open(f, 'rb') as data:
            try:
                df = pickle.load(data)
            except TypeError:
                print("Parsed news is not a valid pickle!")
        
        #stem articles
        print("Here!")
        with Pool(3) as pool:
            df['body_stemmed'] = pool.map(_stemtext,
                                          df['ArticleContents'].values.tolist())
        
        print("Here!")
        #compare to dictionary
        with Pool(3) as pool:
            df['u_count'] = pool.map(partial(_count, 
                                          word_set=U_set), 
                                          df['body_stemmed'].values.tolist())
        
        #save to disk
        df.rename({'ID2': 'article_id'}, axis=1, inplace=True)
        dump_pickle(out_path,'u_count'+str(i+1997)+'.pkl',df[['ID2','u_count','ArticleDateCreated']])


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
        df_lda = pickle.load(f)
    print(df_lda.columns)
    print(df_u.columns)
    
    df_lda = df_lda.merge(df_u, 'inner', 'article_id')
    return df_lda

def ECB_index(params):
    with open("data\\intermediate\\lda\\document_topics.pkl", 'rb') as f_in:
        df = pickle.load(f_in)
    df['max_topic'] = df['topics'].apply(lambda x: np.argmax(x))
    params['topic_dict']
    
if __name__ == '__main__':
 
    os.chdir(main_directory())
    
    PARAMS_PATH = 'scripts/input_params.json'
    with codecs.open(PARAMS_PATH, 'r', 'utf-8-sig') as json_file:  
        params = json.load(json_file)
    
    c80 = ClusterTree(80)
    children = c80.children()
    c80.dendrogram(10,15,colors=15)
    T = c80.flat_clusters(8,1)
    
    
    
#    params['uncertainty_extended_w2v'] = extend_dict_w2v('uncertainty', params, n_words=10)            
#    uncertainty_count(params, 'uncertainty_extended_w2v')
#    df = merge_lda_u(params)
    
    
    
#    topic_labels = {}
#    for l in range(80):
#        topic_labels[l] = ['test']
#    with open('topics_labels.json', 'w') as json_file:
#        json.dump(topic_labels, json_file)
    #df_u = load_parsed_data(params)
