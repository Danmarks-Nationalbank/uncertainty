import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\projects\\FUI\\src')
sys.path.insert(1, 'C:\\projects\\FUI\\env\\Lib\\site-packages')
#sys.path.remove('C:\\ProgramData\\Anaconda3\\lib\\site-packages')
import pandas as pd
import numpy as np
import codecs
import lemmy
import json
import gensim
import random
import pickle
import glob
from matplotlib import pyplot as plt

from fui.cluster import ClusterTree
from fui.lda import LDA
from fui.utils import main_directory, dump_pickle, dump_csv, params
from fui.ldatools import preprocess, optimize_topics, create_dictionary, 
from fui.ldatools import generate_wordclouds, merge_documents_and_topics,
from fui.ldatools import jsd_measure, create_corpus, save_models, load_model
from fui.ldatools import print_topics, parse_topic_labels
from fui.preprocessing import parse_raw_data, load_parsed_data

if __name__ == "__main__":

    NROWS = None
    
#    parse_raw_data(params, nrows=None)

    lemmatizer = lemmy.load("da")
    lda_instance = LDA(lemmatizer, test_share=0.02)
    
    create_dictionary(lda_instance, load_bigrams=False)
    create_corpus(lda_instance)

#    lda_instance.lda_models, coherence_scores = optimize_topics(lda_instance, topics, plot=False)
#    save_models(lda_instance, params)
    
    pd.set_option('max_colwidth', 100)

    load_model(lda_instance, 80)
    labels = parse_topic_labels('labels', 80)
    
    word_list = print_topics(lda_instance,topn=30,unique_sort=False)
    df = pd.DataFrame(word_list)
    for col in df.columns:
        df.rename(columns={col:labels[str(col)][0]}, inplace=True)
    
    dft = df.transpose()
    dft = dft.reset_index()
    dft['text'] = dft.iloc[:,1:10].apply(lambda x: ', '.join(x), axis=1)
    latex = dft.to_latex(columns=['index','text'])

#    jsd = []
#    for topic in topics:
#        load_model(lda_instance,topic,params)
#        jsd_ = jsd_measure(lda_instance,params)*1000
#        print(f"Model with {topic} topics has jsd {jsd_:.6f}")
#        jsd.append(jsd_)

    
    #generate_wordclouds(lda_instance,shade=True,title='Monetary policy',num_words=20,topics=69)





