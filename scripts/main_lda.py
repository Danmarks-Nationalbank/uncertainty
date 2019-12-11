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

from fui.lda import LDA
from fui.utils import main_directory, dump_pickle, dump_csv, params
from fui.ldatools import preprocess, optimize_topics, create_dictionary, create_corpus, save_models, load_model, load_models
from fui.ldatools import generate_wordclouds, merge_documents_and_topics, get_unique_words
from fui.ldatools import get_unique_words, jsd_measure
from fui.ldatools import print_topics, parse_topic_labels
from fui.preprocessing import parse_raw_data, load_parsed_data

if __name__ == "__main__":

    NROWS = None
    topics = [65,68,70,72,75,78,80,85,88]
    
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
    
    word_list = print_topics(lda_instance,topn=8,unique_sort=False)
    df = pd.DataFrame(word_list)
    for col in df.columns:
        df.rename(columns={col:labels[str(col)][0]}, inplace=True)
    
    dft = df.transpose()
    dft = dft.reset_index()
    dft['text'] = dft.iloc[:,1:10].apply(lambda x: ', '.join(x), axis=1)
    latex = dft.to_latex(columns=['index','text'])
#    
#    jsd = []
#    for topic in topics:
#        load_model(lda_instance,topic,params)
#        jsd_ = jsd_measure(lda_instance,params)*1000
#        print(f"Model with {topic} topics has jsd {jsd_:.6f}")
#        jsd.append(jsd_)
  
#    #merge_documents_and_topics(lda_instance)
#    df = get_unique_words(lda_instance.lda_model, lda_instance, 15, sort=False)
#    df.index = pd.MultiIndex.from_arrays(
    
    
#    dfu = get_unique_words(lda_instance, topn=15, sort=True)
    #generate_wordclouds(lda_instance,shade=True,title='Monetary policy',num_words=20,topics=69)


#    define_NB_colors()
#
#    fig, ax1 = plt.subplots(figsize=(14,6))
#    
#    ax1.set_xlabel('Topics')
#    ax1.set_ylabel('Coherence', color=(0/255, 123/255, 209/255))
#    ax1.plot(df_co['topics'], df_co['coherence'], color=(0/255, 123/255, 209/255))
#    ax1.tick_params(axis='y', labelcolor=(0/255, 123/255, 209/255))
#    
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    
#    ax2.set_ylabel('JS distance', color=(146/255, 34/255, 156/255))  # we already handled the x-label with ax1
#    ax2.plot(df_co['topics'], df_co['jsd'], color=(146/255, 34/255, 156/255))
#    ax2.tick_params(axis='y', labelcolor=(146/255, 34/255, 156/255))
#
#    plt.xticks(df_co['topics'])
#    fig.tight_layout() 
#    fig.savefig(os.path.join(params['paths']['lda'], 'topic_selection.pdf'), dpi=300)
#    plt.show()


