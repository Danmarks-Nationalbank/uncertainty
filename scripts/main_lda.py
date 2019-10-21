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
#import pandas as pd
#import pickle
import glob
from src.fui.lda import LDA
from src.fui.utils import main_directory, dump_pickle, dump_csv
from src.fui.ldatools import preprocess, optimize_topics, create_dictionary, create_corpus, save_models, load_model, load_models
from src.fui.ldatools import merge_documents_and_topics, dominating_sentence_per_topic, term_frequency, get_perplexity
from src.fui.ldatools import get_top_words, _get_scaled_significance, docs2bow

from src.fui.preprocessing import parse_raw_data, load_parsed_data

if __name__ == "__main__":
    os.chdir(main_directory())
    PARAMS_PATH = 'scripts/input_params.json'
    NROWS = None
    topics = [160,120,80,40,20]
    
#    with open(PARAMS_PATH, encoding='utf8') as json_params:
#        params = json.load(json_params)
    with codecs.open(PARAMS_PATH, 'r', 'utf-8-sig') as json_file:  
        params = json.load(json_file)
    
    #parse_raw_data(params, nrows=None)
    
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'boersen*.pkl') 

    lemmatizer = lemmy.load("da")
    lda_instance = LDA(filelist, params, lemmatizer, test=False)
    #phrases = add_bigrams(lda_instance,params)
    
    create_dictionary(lda_instance, params)
    create_corpus(lda_instance, params)

    lda_instance.lda_models, coherence_scores = optimize_topics(lda_instance, topics, plot=False)
    save_models(lda_instance, params)
    #load_models(lda_instance, topics, params)
    
     #lda_instance.lda_models[3].show_topic(2,50)
#    df = get_top_words(lda_instance.lda_models[0], lda_instance, params, topn=10)
#    
#    for model in models:
#        num_topics = len(model.print_topics(num_topics=-1, num_words=1))
#        perp, test_corpus = get_perplexity(model,params,chunksize=1000)
#        print(f"Model with {num_topics} has perplexity score {perp:.1f}")

    #df = load_parsed_data(params,1000)
    
    
    #lda_instance.lda_models, coherence_scores = optimize_topics(lda_instance, topics, plot=False)
    #save_models(lda_instance, params)
    
    #lda_instance.lda_models, coherence_scores = load_models(lda_instance, topics, params)
    #print(lda_instance.coherence_scores)
    
    # Choose desired model
    #load_model(lda_instance, 80, params)
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


