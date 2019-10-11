import os
#hacky spyder crap
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\projects\\FUI\\src')
sys.path.insert(1, 'C:\\projects\\FUI\\hackenv')
import json
import pandas as pd
import pickle
import glob
from src.fui.lda import LDA
from src.fui.utils import main_directory, dump_pickle, dump_csv
from src.fui.ldatools import preprocess, optimize_topics, create_dictionary, create_corpus, save_models, load_model
from src.fui.ldatools import merge_documents_and_topics, generate_wordclouds, dominating_sentence_per_topic, term_frequency
from src.fui.preprocessing import parse_raw_data
from src.fui.bloom import bloom_measure, plot_index


if __name__ == "__main__":
    os.chdir(main_directory())
    PARAMS_PATH = 'scripts/input_params.json'
    NROWS = None
    num_topics = [80,60,40,20]
    
    with open(PARAMS_PATH, encoding='utf8') as json_params:
        params = json.load(json_params)
        
    filelist = glob.glob(params['paths']['root']+
                         params['paths']['parsed_news']+'boersen*.pkl') 
    filelist = [(f,int(f[-8:-4])) for f in filelist 
                if int(f[-8:-4]) >= start_year and int(f[-8:-4]) <= end_year]
        
    lda_instance = LDA(filelist, params, test=False)
    create_dictionary(lda_instance, params)
    create_corpus(lda_instance, params)

    # if False:
    #     a, b = LDATools.term_frequency(lda_instance.SerializedCorpus, lda_instance.dictionary)

    lda_instance.lda_models, coherence_scores = optimize_topics(lda_instance, num_topics, plot=False)
    save_models(lda_instance, params)
    
    
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


