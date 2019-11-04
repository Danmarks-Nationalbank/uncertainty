import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'D:\\projects\\FUI')
sys.path.insert(1, 'D:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
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

from src.fui.lda import LDA
from src.fui.utils import main_directory, dump_pickle, dump_csv, define_NB_colors
from src.fui.ldatools import preprocess, optimize_topics, create_dictionary, create_corpus, save_models, load_model, load_models
from src.fui.ldatools import generate_wordclouds, merge_documents_and_topics
from src.fui.ldatools import get_top_words, jsd_measure, get_perplexity
from src.fui.ldatools import print_topics, cluster_topics
from src.fui.preprocessing import parse_raw_data, load_parsed_data

if __name__ == "__main__":
    os.chdir(main_directory())
    PARAMS_PATH = 'scripts/input_params.json'
    NROWS = None
    topics = [65,68,70,72,75,78,80,85,88]
    
#    with open(PARAMS_PATH, encoding='utf8') as json_params:
#        params = json.load(json_params)
    with codecs.open(PARAMS_PATH, 'r', 'utf-8-sig') as json_file:  
        params = json.load(json_file)
    
    #parse_raw_data(params, nrows=None)
    
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'boersen*.pkl') 

    lemmatizer = lemmy.load("da")
    lda_instance = LDA(filelist, params, lemmatizer, test_share=0.02)
    
    create_dictionary(lda_instance, params, load_bigrams=False)
    create_corpus(lda_instance, params)

#    lda_instance.lda_models, coherence_scores = optimize_topics(lda_instance, topics, plot=False)
#    save_models(lda_instance, params)
##    
 
    load_model(lda_instance, 80, params)
    topics = lda_instance.lda_model.get_topics()
    y = pdist(topics, metric='jensenshannon')
    Z = hac.linkage(y, method='ward')
    rootnode, nodelist = hac.to_tree(Z,rd=True)
    children = {}
    for i in range(79,len(nodelist)):
        children[i] = [child.id for child in _children(nodelist,i)]
        
#    print_topics(lda_instance.lda_model,params)
#    
#    #df = get_top_words(lda_instance.lda_model, lda_instance, params, topn=30)
#    jsd = []
#    for topic in topics:
#        load_model(lda_instance,topic,params)
#        jsd_ = jsd_measure(lda_instance,params)*1000
#        print(f"Model with {topic} topics has jsd {jsd_:.6f}")
#        jsd.append(jsd_)

    #merge_documents_and_topics(filelist, lda_instance, params)
#
#     
#    df_co = pd.read_csv('coherence.csv',header=None,names=['topics','coherence'])
#    df_co['jsd'] = np.asarray(jsd)[df_co.index]
#
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
    

#    df_top = get_top_words(lda_instance.lda_models[1], lda_instance, params, topn=30)

    # Choose desired model
    
    
    #generate_wordclouds(lda_instance, params)


    # These functions all refer to the model chosen above, i.e., lda_instance.lda_model
    # LDATools.plt_weight_words(lda_instance)
    # LDATools.merge_documents_and_topics(lda_instance)
    # LDATools.weight_of_top_words(lda_instance)
    # LDATools.dominating_sentence_per_topic(lda_instance, lda_model, SerializedCorpus)

    # plot_firmshare_per_topic()
    # plot_negativity_trend()

    # iv = IVconstruction(params)
    # iv.load_topic_enriched_documents()
    # iv.construct_iv()
    #
    # folder_path = os.path.join(params.iv['folder'])
    # file_path = os.path.join(folder_path, params.iv['iv_file'])
    # with open(file_path, 'rb') as f_in:
    #     iv = pickle.load(f_in)


