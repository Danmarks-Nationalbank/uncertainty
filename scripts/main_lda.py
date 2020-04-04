import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
#sys.path.insert(1, 'C:\\projects\\FUI\\src')
#sys.path.insert(1, 'C:\\projects\\FUI\\env\\Lib\\site-packages')


from src.fui.cluster import ClusterTree
import lemmy
from src.fui.lda import LDA
from src.fui.utils import main_directory, dump_pickle, dump_csv, params
from src.fui.ldatools import preprocess, optimize_topics, create_dictionary
from src.fui.ldatools import jsd_measure, create_corpus, save_models, load_model
from src.fui.preprocessing import parse_for_lda, load_parsed_data

if __name__ == "__main__":

    NROWS = None
    
    parse_for_lda(nrows=None)

    lemmatizer = lemmy.load("da")
    lda_instance = LDA(lemmatizer, test_share=0.0)
    
    create_dictionary(lda_instance, load_bigrams=True)
    create_corpus(lda_instance)

    lda_instance.lda_models, coherence_scores = optimize_topics(lda_instance, 80, plot=False)
    save_models(lda_instance, params)

    #load_model(lda_instance, 80)
    #labels = parse_topic_labels('labels', 80)

    # word_list = print_topics(lda_instance,topn=30,unique_sort=False)
    # df = pd.DataFrame(word_list)
    # for col in df.columns:
    #     df.rename(columns={col:labels[str(col)][0]}, inplace=True)
    #
    # dft = df.transpose()
    # dft = dft.reset_index()
    # dft['text'] = dft.iloc[:,1:10].apply(lambda x: ', '.join(x), axis=1)
    # latex = dft.to_latex(columns=['index','text'])

#    jsd = []
#    for topic in topics:
#        load_model(lda_instance,topic,params)
#        jsd_ = jsd_measure(lda_instance,params)*1000
#        print(f"Model with {topic} topics has jsd {jsd_:.6f}")
#        jsd.append(jsd_)

    
    #generate_wordclouds(lda_instance,shade=True,title='Monetary policy',num_words=20,topics=69)





