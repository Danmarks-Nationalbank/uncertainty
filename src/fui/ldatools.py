import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gensim
import numpy as np
import os
import pandas as pd
import random
import csv
import json
import copy
import codecs

from collections import Counter
from matplotlib import pyplot as plt

#from wordcloud import WordCloud
#from langdetect import detect

from src.fui.utils import timestamp, params
from src.fui.sql import update_article_topics


def __remove_stopwords(word_list, stopfile):
    if not os.path.exists(stopfile):
        raise Exception('No stopword file in directory')
    stopwords_file = open(stopfile, "r")
    stopwords = stopwords_file.read().splitlines()
    word_list = [word for word in word_list if word not in stopwords]
    return word_list

def print_topics(lda_instance, topn=30, unique_sort=True):
    lda_model = lda_instance.lda_model
    
    csv_path = os.path.join(params().paths['lda'], 
                            'topic_words'+str(lda_model.num_topics)+'.csv') 
    header = ['topic_'+str(x) for x in range(lda_model.num_topics)]
    
    if not unique_sort:
        word_lists = []
        for t in range(lda_model.num_topics):
            word_list = lda_model.show_topic(t,topn)
            if not len(word_lists):
                word_list = [[w[0]] for w in word_list]
                word_lists = word_list
            else:
                word_list = [w[0] for w in word_list]
                for i in range(topn):
                    word_lists[i].append(word_list[i])
        with open(csv_path, mode='w', newline='\n', encoding='utf-8-sig') as csv_out:
            csvwriter = csv.writer(csv_out, delimiter=',')
            csvwriter.writerow(header)
            for i in range(topn):
                csvwriter.writerow(word_lists[i])
        return word_lists
        
    else: 
        df = get_unique_words(lda_instance, topn)

        df = df[['word']]
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(1), df.groupby(level=1).cumcount()],
            names=['token', 'topic'])
        df = df.unstack(level=0)
        df.to_csv(csv_path,header=header,encoding='utf-8-sig',index=False)
        return df
        
def optimize_topics(lda_instance, topics_to_optimize, plot=False, plot_title=""):
    coherence_scores = []
    lda_models = []

    if not hasattr(lda_instance, 'Corpus'):
        lda_instance.create_corpus()

    print("Finding coherence-scores for the list {}:".format(topics_to_optimize))
    for num_topics in topics_to_optimize:
        print("\t{} topics... {}".format(num_topics, timestamp()))

        lda_model_n = gensim.models.LdaMulticore(corpus=lda_instance.Corpus,
                                                 num_topics=num_topics,
                                                 id2word=lda_instance.dictionary,
                                                 passes=20, per_word_topics=False,
                                                 alpha='asymmetric',
                                                 eval_every=100,
                                                 minimum_probability=0.0,
                                                 chunksize=10000, workers=16)

        coherence_model_n = gensim.models.CoherenceModel(model=lda_model_n,
                                                         texts=(articles for articles in lda_instance),
                                                         dictionary=lda_instance.dictionary,
                                                         coherence='c_v',
                                                         processes=16)
        lda_models.append(lda_model_n)
        coherence_scores.append(coherence_model_n.get_coherence())

        try:
            folder_path = os.path.join(params().paths['lda'], 'lda_model_' + str(num_topics))
            file_path = os.path.join(folder_path, 'trained_lda')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lda_model_n.save(file_path)
            print("LDA-model saved ({} topics)".format(num_topics))
        except FileNotFoundError:
            print("Error: LDA-file not found")

        with open('coherence.csv', 'a+', newline='') as csvout:
            wr = csv.writer(csvout, delimiter=',', lineterminator='\n')
            wr.writerow([num_topics, coherence_model_n.get_coherence()])

    for n, cv in zip(topics_to_optimize, coherence_scores):
        print("LDA with {} topics has a coherence-score {}".format(n, round(cv, 2)))

    if plot:
        plt.plot(topics_to_optimize, coherence_scores)
        plt.xlabel('Number of topics')
        plt.ylabel('Coherence score')
        #plot_title += str(params().options['lda']['tf-idf'])
        #plt.title(plot_title)
        plt.show()

    return lda_models, coherence_scores


def save_models(lda_instance):

    # Save all models in their respective folder
    for i, lda_model in enumerate(lda_instance.lda_models):
        try:
            folder_path = os.path.join(params().paths['lda'], 'lda_model_' + str(lda_model.num_topics))
            file_path = os.path.join(folder_path, 'trained_lda')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lda_model.save(file_path)
            print("LDA-model #{} saved ({} topics)".format(i, lda_model.num_topics))
        except FileNotFoundError:
            print("Error: LDA-file not found")
        except IndexError:
            print("Error: List index out of range")
        
def load_models(lda_instance, topics, plot=False):
    lda_models = []
    file_list = []
    for t in topics: 
        print(t)
        file_list.append(os.path.join(params().paths['root'],params().paths['lda'], 'lda_model_'+str(t)+'\\trained_lda'))


    for f in file_list:
        print(f)
        try:
            lda_model_n = gensim.models.LdaMulticore.load(f)
            lda_models.append(lda_model_n)

        except FileNotFoundError:
            print(f"Error: LDA-model at {f} not found")

    lda_instance.lda_models = lda_models

    
def docs2bow(sample_size=2000):
    file_path = os.path.join(params().paths['lda'], 'corpus.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path)  # `mm` document stream now has random access
    if sample_size is not None:
        sample = [random.randint(0,mm.num_docs) for i in range(sample_size)]
        corpus_bow = []
        for doc in sample:
            corpus_bow.append(mm[doc])
    else:
        corpus_bow = []
        for doc in range(0,mm.num_docs,1):
            corpus_bow.append(mm[doc])
    word_ids = [item for sublist in corpus_bow for item in sublist]
    df = pd.DataFrame(word_ids, columns=['word','count'], dtype='int')
    df = df.groupby(['word'])['count'].sum().reset_index()
    bow = [tuple(x) for x in df.values]
    return bow

def get_word_proba(bow,lda_instance):
    """Returns probability matrix of same format as lda_model.get_topics() for test corpus,
    corrects for missing probabilities due to missing words in test corpus by adding zero padding.
    """
    test_topics, test_word_topics, test_word_proba = lda_instance.lda_model.get_document_topics(bow, 
                                                                                                minimum_probability=0.0, 
                                                                                                minimum_phi_value=0.0, 
                                                                                                per_word_topics=True)
    
    placeholder = [(j,0.0) for j in range(lda_instance.lda_model.num_topics)]
    for i,t in enumerate(test_word_proba):
        if t[1] is None:
            test_word_proba.append((i,placeholder))
        elif not len(t[1]):
            #print(test_word_proba[i]) 
            test_word_proba[i] = (i,placeholder)
            test_word_proba.sort()
        elif len(t[1]) is not lda_instance.lda_model.num_topics:
            #print(test_word_proba[i]) 
            dict_ = dict(t[1])
            for j in range(lda_instance.lda_model.num_topics):
                try:
                   dict_[j]
                except KeyError:
                   dict_[j] = 0.0
            dict_list = list(dict_.items())
            dict_list.sort()
            test_word_proba[i] = (i,dict_list)
            test_word_proba.sort()

    test_word_proba = [[i[1] for i in g[1]] for g in test_word_proba]
    #transform to probabilities
    test_word_proba = np.transpose(np.apply_along_axis(lambda x: x/x.sum(),0,np.array(test_word_proba)))
    return test_topics, test_word_topics, test_word_proba
    
def get_perplexity(lda_model, lda_instance, chunksize=2000):
    file_path_test = os.path.join(params().paths['lda'], 'corpus_test.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path_test)  # `mm` document stream now has random access
    sample = [random.randint(0,mm.num_docs) for i in range(chunksize)]
    test_corpus = []
    for doc in sample:
        test_corpus.append(mm[doc])
    perplexity = np.exp2(-lda_model.log_perplexity(test_corpus,len(lda_instance.articles)))
    return perplexity

def _get_scaled_significance(lda_model, n_words=200):
    _cols = ['token_id', 'weight', 'topic']
    df = pd.DataFrame(data=None, columns=_cols)
    num_topics = lda_model.num_topics
    for i in range(0,num_topics,1):
        _list = lda_model.get_topic_terms(i,n_words)
        df_n = pd.DataFrame(_list, columns=_cols[0:2])
        df_n['topic'] = i
        df = df.append(df_n)
        
    df = df.set_index(['token_id','topic'])
    df = df.join(df.groupby(level=0).sum(), how='inner', rsuffix='_sum').groupby(level=[0,1]).first()
    df['scaled_weight'] = df['weight']/df['weight_sum']
    return df['scaled_weight']

def get_unique_words(lda_instance, topn=10):
    """Builds df with topic words sorted by scaled uniqueness. 
    args:
        lda_instance (obj): Instance of LDA
        topn (int): top words to consider when sorting
    returns:
        sorted DataFrame
    """
    df_out = pd.DataFrame(data=None, columns=['scaled_weight','word'])

    df = _get_scaled_significance(lda_instance.lda_model, topn)
    for i in range(0,lda_instance.lda_model.num_topics,1):
        tokens = []
        df_topic = df[df.index.get_level_values('topic') == i]
        df_topic = df_topic[0:topn]
        for t in range(0,topn,1):
            tokens.append(lda_instance.dictionary[df_topic.index.values[t][0]])
        tokens = pd.DataFrame(tokens, index=df_topic.index, columns=['word'])
        df_topic = pd.concat([df_topic,tokens], axis=1, sort=True)

        df_out = df_out.append(df_topic)
    df_out.index = pd.MultiIndex.from_tuples(df_out.index)
    df_out = df_out.rename_axis(['token_id','topic'])
    df_out = df_out.sort_values(by = ['topic', 'scaled_weight'], ascending = [True, False])

    return df_out

def upload_topics(res):
    res = [item for sublist in res for item in sublist]
    df = pd.DataFrame.from_records(res)
    df.columns = ['DNid', 'topic', 'probability']
    topic_file = os.path.join(params().paths['area060'], "article_topics.csv")
    df.to_csv(topic_file, index=False)
    print("Uploading topics to sql server")
    update_article_topics(topic_file)
    del (df)

def get_topics(lda_instance):
    res = []
    c = 0
    for line in lda_instance.get_doc_topics():
        topics = [list(i) for i in line[0]]
        for t in topics:
            t.insert(0,line[1])
        res.append(topics)
        c += 1
        if len(res) == 100000:
            upload_topics(res)
            res = []
    upload_topics(res)

def term_frequency(corpus_bow, dictionary, terms=30):
    corpus_iter = iter(corpus_bow)
    counter = Counter()
    while True:
        try:
            document = next(corpus_iter)
            for token_id, document_count in document:
                counter[dictionary.get(token_id)] += document_count
        except StopIteration:
            print("Done counting term frequencies")
            break
    return counter.most_common(terms), counter.most_common()[:-terms-1:-1]

def _return_array(array, num_topics=80):
    """
    Utility function transforming the projections as returned by the LDA
    module into a 1xT numpy array, where T is the number of topics.
    """
    output = np.array(
        [[sum([el[1] for el in row if el[0] == topic]) for topic in range(1, n_topics + 1)] for row in array])
    return output

def parse_topic_labels(name,num_topics):
    """
    reads hand labeled topics from json file.
    
    """
    label_path = os.path.join(params().paths['topic_labels'], 
                              name+str(num_topics)+'.json')
      
    with codecs.open(label_path, 'r', encoding='utf-8-sig') as f:
        labels = json.load(f)
    return labels
