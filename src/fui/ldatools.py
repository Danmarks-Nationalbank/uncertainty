import gensim
import h5py
import numpy as np
import os
import pandas as pd
import pickle
import random
import csv
import copy

from collections import Counter
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
#from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
import lemmy

from src.fui.utils import timestamp


def __remove_stopwords(word_list, stopfile):
    if not os.path.exists(stopfile):
        raise Exception('No stopword file in directory')
    stopwords_file = open(stopfile, "r")
    stopwords = stopwords_file.read().splitlines()
    word_list = [word for word in word_list if word not in stopwords]
    return word_list   

def preprocess(text, lemmatizer, stopfile='data/stopwords.txt'):
    """
    - simple_preprocess (convert all words to lowercase, remove punctuations, floats, newlines (\n),
    tabs (\t) and split to words)
    - remove all stopwords
    - remove words with a length < threshold
    - lemmatize
    """
    text = gensim.utils.simple_preprocess(text, deacc=True, max_len=25)
    list_to_stem = __remove_stopwords(text, stopfile)
        
    lemmed_list = [lemmatizer.lemmatize("", word)[0] for word in list_to_stem]
    text = [word for word in lemmed_list if len(word) >= 3]
    
    return ' '.join([word for word in lemmed_list])

def optimize_topics(lda_instance, topics_to_optimize, plot=True, plot_title=""):
    coherence_scores = []
    lda_models = []

    print("Finding coherence-scores for the list {}:".format(topics_to_optimize))
    for num_topics in topics_to_optimize:
        print("\t{} topics... {}".format(num_topics, timestamp()))

        lda_model_n = gensim.models.LdaMulticore(corpus=lda_instance.SerializedCorpus,
                                                 num_topics=num_topics,
                                                 id2word=lda_instance.dictionary,
                                                 passes=1, per_word_topics=False,
                                                 # alpha=50/num_topics,
                                                 # eta=0.005,
                                                 workers=15)

        coherence_model_n = gensim.models.CoherenceModel(model=lda_model_n,
                                                         texts=(articles for articles in lda_instance),
                                                         dictionary=lda_instance.dictionary,
                                                         coherence='c_v',
                                                         processes=15)
        lda_models.append(lda_model_n)
        coherence_scores.append(coherence_model_n.get_coherence())

    for n, cv in zip(topics_to_optimize, coherence_scores):
        print("LDA with {} topics has a coherence-score {}".format(n, round(cv, 2)))
        
    with open('coherence.csv', 'w', newline='') as csvout:
        wr = csv.writer(csvout, delimiter=',')
        for cv, n in zip(coherence_scores, topics_to_optimize):
            wr.writerow([n,cv])

    if plot:
        plt.plot(topics_to_optimize, coherence_scores)
        plt.xlabel('Number of topics')
        plt.ylabel('Coherence score')
        #plot_title += str(lda_instance.params.lda['tf-idf'])
        #plt.title(plot_title)
        plt.show()

    return lda_models, coherence_scores


def create_dictionary(lda_instance, params, unwanted_words=None, keep_words=None):
    
    # Clean and write texts to HDF
    if not lda_instance.load_processed_text():
        lda_instance.load_and_clean_body_text()
        
    # Create dictionary (id2word)
    file_path = os.path.join(params['paths']['lda'], params['filenames']['lda_dictionary'])
    
    # Load bigram phraser
    if not hasattr(lda_instance, 'bigram_phraser'):
        lda_instance.load_bigrams()
            
    
    try:
        lda_instance.dictionary = gensim.corpora.Dictionary.load(file_path)
        print("Loaded pre-existing dictionary")
    except FileNotFoundError:
        print("Dictionary not found, creating from scratch")

        lda_instance.dictionary = gensim.corpora.Dictionary(articles for articles in lda_instance)

        lda_instance.dictionary.filter_extremes(no_below=params['options']['lda']['no_below'],
                                                no_above=params['options']['lda']['no_above'],
                                                keep_n=params['options']['lda']['keep_n'],
                                                keep_tokens=keep_words)
        if unwanted_words is None:
            unwanted_words = []
        unwanted_ids = [k for k, v in lda_instance.dictionary.items() if v in unwanted_words]
        lda_instance.dictionary.filter_tokens(bad_ids=unwanted_ids)
        lda_instance.dictionary.compactify()
        lda_instance.dictionary.save(file_path)
    print("\t{}".format(lda_instance.dictionary))


def create_corpus(lda_instance, params):

    # Helper-class to create BoW-corpus "lazily"
    class MyCorpus:
        def __iter__(self):
            for line in lda_instance.articles:
                yield lda_instance.dictionary.doc2bow(lda_instance.bigram_phraser[line.split()])

    # Serialize corpus using either BoW of tf-idf
    corpus_bow = MyCorpus()

    file_path = os.path.join(params['paths']['lda'], 'corpus.mm')
    try:
        lda_instance.SerializedCorpus = gensim.corpora.MmCorpus(file_path)
        print("Loaded pre-existing corpus")
    except FileNotFoundError:
        print("Corpus not found, creating from scratch")
        if not hasattr(lda_instance, 'bigram_phraser'):
            lda_instance.load_bigrams()

        # Serialize corpus (either BoW or tf-idf)
        if not params['options']['lda']['tf-idf']:
            print("\tSerializing corpus, BoW")
            gensim.corpora.MmCorpus.serialize(file_path, corpus_bow)
        else:
            print("\tSerializing corpus, tf-idf")
            tfidf = gensim.models.TfidfModel(corpus_bow)
            corpus_tfidf = tfidf[corpus_bow]
            gensim.corpora.MmCorpus.serialize(file_path, corpus_tfidf)

        lda_instance.SerializedCorpus = gensim.corpora.MmCorpus(file_path)


def save_models(lda_instance, params):

    # Save all models in their respective folder
    for i, lda_model in enumerate(lda_instance.lda_models):
        try:
            folder_path = os.path.join(params['paths']['lda'], 'lda_model_' + str(lda_model.num_topics))
            file_path = os.path.join(folder_path, 'trained_lda')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lda_model.save(file_path)
            print("LDA-model #{} saved ({} topics)".format(i, lda_model.num_topics))
        except FileNotFoundError:
            print("Error: LDA-file not found")
        except IndexError:
            print("Error: List index out of range")


def load_model(lda_instance, num_topics, params):
    try:
        folder_path = os.path.join(params['paths']['root'],params['paths']['lda'], 'lda_model_' + str(num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        lda_instance.lda_models = gensim.models.LdaMulticore.load(file_path)
        print("LDA-model with {} topics loaded".format(num_topics))
    except FileNotFoundError:
        print("Error: LDA-model not found")
        lda_instance.lda_models = None
        
def load_models(lda_instance, topics, params, plot=False):
    lda_models = []
    coherence_scores = []
    file_list = []
    for t in topics: 
        print(t)
        file_list.append(os.path.join(params['paths']['root'],params['paths']['lda'], 'lda_model_'+str(t)+'\\trained_lda'))


    for f in file_list:
        print(f)
        try:
            lda_model_n = gensim.models.LdaMulticore.load(f)
#            print(f"LDA-model at {f} loaded, getting coherence score")
#            coherence_model_n = gensim.models.CoherenceModel(model=lda_model_n,
#                                                         texts=(articles for articles in lda_instance),
#                                                         dictionary=lda_instance.dictionary,
#                                                         coherence='c_v',
#                                                         processes=10)
    
            lda_models.append(lda_model_n)
#            cv = coherence_model_n.get_coherence()
#            coherence_scores.append(cv)
#            print("LDA at %s topics has a coherence-score %8.2f" % (f, cv))

        except FileNotFoundError:
            print(f"Error: LDA-model at {f} not found")
    
#    with open('coherence.csv', 'w', newline='') as csvout:
#        wr = csv.writer(csvout, delimiter=',')
#        for cv, n in zip(coherence_scores, topics):
#            wr.writerow([n,cv])
#    
#    if plot:
#        plt.plot(topics, coherence_scores)
#        plt.xlabel('Number of topics')
#        plt.ylabel('Coherence score')
#        #plot_title += str(lda_instance.params.lda['tf-idf'])
#        #plt.title(plot_title)
#        plt.show()

    lda_instance.lda_models = lda_models
#, coherence_scores
    
def docs2bow(params, sample_size=2000):
    file_path = os.path.join(params['paths']['lda'], 'corpus.mm')
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

def corpus2bow(lda_instance, params):
    """
    returns corpus in a bag of words list of (word_id,word_count)
    """
    bow_dict = copy.deepcopy(lda_instance.dictionary.id2token)
    bow_dict = {k: 0 for (k,v) in bow_dict}
    file_path = os.path.join(params['paths']['lda'], 'corpus.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path)  # `mm` document stream now has random access
    for doc in range(0,mm.num_docs,1):
        doc_dict = dict(mm[doc])
        for k, v in doc_dict.items():
            bow_dict[k] = bow_dict[k] + v
    bow_list = [(k, v) for k, v in bow_dict.items()] 
    return bow_list

def get_perplexity(lda_model, params, chunksize=2000):
    file_path = os.path.join(params['paths']['lda'], 'corpus.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path)  # `mm` document stream now has random access
    sample = [random.randint(0,mm.num_docs) for i in range(chunksize)]
    test_corpus = []
    for doc in sample:
        test_corpus.append(mm[doc])
    perplexity = np.exp2(-lda_model.log_perplexity(test_corpus,mm.num_docs))
    return perplexity, test_corpus

def _get_scaled_significance(lda_model, params, n_words=200):
    _cols = ['token_id', 'weight', 'topic']
    df = pd.DataFrame(data=None, columns=_cols)
    num_topics = lda_model.num_topics
    for i in range(0,num_topics,1):
        _list = lda_model.get_topic_terms(i,n_words)
        df_n = pd.DataFrame(_list, columns=_cols[0:2])
        df_n['topic'] = i
        df = df.append(df_n)
        
    df = df.set_index(['token_id','topic'])
    df = df.join(df.groupby(level=0).sum(),how='inner',rsuffix='_sum').groupby(level=[0,1]).first()
    df['scaled_weight'] = df['weight']/df['weight_sum']
    df.sort_values(['topic','scaled_weight'],inplace=True,ascending=False)
    return df['scaled_weight']

def get_top_words(lda_model, lda_instance, params, topn=10):
    df_out = pd.DataFrame(data=None, columns=['scaled_weight','word'])

    df = _get_scaled_significance(lda_model, params, 20)
    for i in range(0,lda_model.num_topics,1):
        tokens = []
        df_topic = df[df.index.get_level_values('topic') == i]
        df_topic = df_topic[0:topn]
        for t in range(0,topn,1):
            tokens.append(lda_instance.dictionary[df_topic.index.values[t][0]])
        tokens = pd.DataFrame(tokens, index=df_topic.index, columns=['word'])
        print(tokens)
        df_topic = pd.concat([df_topic,tokens], axis=1, sort=True)
        #print(tokens)
        #print(df_topic)
        df_out = df_out.append(df_topic)
    return df_out

def merge_documents_and_topics(lda_instance, params):

    print("Merging documents and LDA-topics")

    # Load enriched articles
    articles = pd.read_csv(os.path.join(params['paths']['enriched_news'],
                                        params['filenames']['CENSOR_events_enriched_data']),
                           sep=None, engine='python')
    articles.rename({'id': 'article_id'}, axis=1, inplace=True)
    print("\tLoaded {} enriched documents ({} unique)... {}".format(len(articles),
                                                                    len(articles['article_id'].unique()),
                                                                    timestamp()))

    # Find LDA-document indices that match the ids of the enriched articles
    enriched_article_id = set(articles['article_id'])
    lda_indices = [i for (i, j) in enumerate(lda_instance.article_id) if j in enriched_article_id]
    print("\t{} common article-ids... {}".format(len(lda_indices), timestamp()))

    # Find document-topics for the document-intersection above
    with Pool(6) as pool:
        document_topics = pool.map(partial(LDA.get_topics,
                                           lda_instance.lda_model,
                                           lda_instance.dictionary),
                                   [lda_instance.articles[i] for i in lda_indices])

    df_lda = pd.DataFrame({'article_id': [lda_instance.article_id[i] for i in lda_indices],
                           'topics': [[x[1] for x in document_topics[i]] for i in range(len(document_topics))]})

    # Merge the enriched data onto LDA-projections
    df_enriched_lda = pd.merge(df_lda, articles[params['options']['lda']['features'] +
                                                ['dates', 'own_firm_id', 'article_id']],
                               how='inner',
                               on='article_id')

    print("\tJoin between LDA-topics and enriched documents gave {} documents... {}".format(len(df_enriched_lda),
                                                                                            timestamp()))

    # Datetime conversion and weekdate
    df_enriched_lda['dates'] = pd.to_datetime(df_enriched_lda['dates'])
    df_enriched_lda['weekdate'] = (
                df_enriched_lda['dates'] - df_enriched_lda['dates'].dt.weekday * timedelta(days=1)).dt.date

    # Save enriched documents with their topics
    folder_path = os.path.join(params['paths']['lda'], 'document_topics')
    topics_path = os.path.join(folder_path, params['filenames']['lda_merge_doc_topics_file'])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(topics_path, 'wb') as f_out:
        print("\tWriting file to disc... {}".format(timestamp()))
        pickle.dump(df_enriched_lda, f_out)


def generate_wordclouds(lda_instance, params, num_words=15):
    colors = ["#000000", "#111111", "#101010", "#121212", "#212121", "#222222"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    cloud = WordCloud(background_color='white', font_path='C:/WINDOWS/FONTS/TAHOMA.TTF', stopwords=[],
                      collocations=False, colormap=cmap, max_words=200, width=1000, height=600)

    print("Generating wordclouds... {}:".format(timestamp()))
    for lda_model in lda_instance.lda_models:
        print("\t{} topics...".format(lda_model.num_topics))

        folder_path = os.path.join(params['paths']['lda'], 'wordclouds_' + str(lda_model.num_topics))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        topics = lda_model.show_topics(formatted=False, num_topics=-1, num_words=num_words)
        for topic_num in range(0, lda_model.num_topics):
            topic_words = dict(topics[topic_num][1])
            cloud.generate_from_frequencies(topic_words)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic {}'.format(topic_num), fontdict=dict(size=12))
            plt.gca().axis('off')

            file_path = os.path.join(folder_path, '{}.png'.format(topic_num))
            plt.savefig(file_path, bbox_inches='tight')


def dominating_sentence_per_topic(lda_instance, lda_model, corpus):
    """
    Construct a dataframe that, for each topic, contains
        - The (un-processed) document that loads the most on it
        - The document-weight
        - The document (article)-id
        - The fraction of documents that load the most on the topic
    """

    df_dom_sentences = pd.DataFrame()
    subset_idx = random.sample(range(0, len(lda_model[corpus])), int(len(lda_model[corpus])))

    for i, row in enumerate([lda_model[corpus][i] for i in subset_idx]):
        # Pick the topic that the document loads the most on (the first element after sorting)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        (topic_num, prop_topic) = row[0]

        # Extract keywords of topic_num
        topic_keywords = lda_model.show_topic(topic_num)
        topic_keywords = ', '.join([word for word, prop in topic_keywords])

        # Gather topic_num, loading, and keywords of topic_num for this document
        df_dom_sentences = df_dom_sentences.append(pd.Series([int(topic_num),
                                                              round(prop_topic, 3),
                                                              topic_keywords]), ignore_index=True)

    # Rename column and add article-id
    df_dom_sentences.columns = ['dominating_topic', 'projection', 'topic_keywords']
    df_dom_sentences['article_id'] = [lda_instance.article_id[i] for i in subset_idx]

    # Group by topic and pick document with largest projection per group
    df_dom_sentence_sort = pd.DataFrame()
    for i, grp in df_dom_sentences.groupby('dominating_topic'):
        df_grp = grp.sort_values('projection', ascending=False).head(1)
        df_grp['fraction_documents'] = round(len(grp)/len(df_dom_sentences), 3)

        df_dom_sentence_sort = df_dom_sentence_sort.append(df_grp)

    df_dom_sentence_sort.reset_index(drop=True, inplace=True)

    # Append un-processed data to df
    files_list = get_files_list(params['paths']['parsed_news'])
    for f in files_list:
        with open(os.path.join(params['paths']['parsed_news'], f), 'rb') as f_in:
            df_year = pickle.load(f_in)

            df_dom_sentence_sort.set_index('article_id', inplace=True)
            df_dom_sentence_sort['body'] = None
            df_dom_sentence_sort.update(df_year.set_index('id')['body'])

    return df_dom_sentence_sort


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


def visualize_lda(lda_instance, params, lambda_step=0.05):
    from pyLDAvis import gensim
    print("Creating vis_data")
    vis_data = gensim.prepare(lda_instance.lda_model,
                              lda_instance.SerializedCorpus,
                              lda_instance.dictionary,
                              sort_topics=False, lambda_step=lambda_step)
    pyLDAvis.save_html(vis_data, os.path.join(params['paths']['lda'], 'pyLDAvis.html'))


def plot_descending_topic_size(lda_instance, thresholds):
    # flattened_documents = [word for document in lda_instance.articles for word in document.split()]
    # counter = Counter(flattened_documents)
    topics = lda_instance.lda_model.show_topics(formatted=False, num_topics=-1, num_words=100)

    df_keywords = []
    for topic_number, topic in topics:
        for keyword, weight in topic:
            # df_keywords.append([topic_number, keyword, weight, counter[keyword]])
            df_keywords.append([topic_number, keyword, weight])

    for threshold in thresholds:
        # df_threshold = pd.DataFrame(df_keywords, columns=['topic', 'keyword', 'keyword_weight', 'word_count'])
        df_threshold = pd.DataFrame(df_keywords, columns=['topic', 'keyword', 'keyword_weight'])
        df_threshold = pd.DataFrame([(i, len(df_threshold[(df_threshold['topic'] == i) &
                                                          (df_threshold['keyword_weight'] > threshold)]))
                                     for i in range(0, len(topics))],
                                    columns=['topic_number', 'number_keywords'])

        df_threshold.sort_values('number_keywords', ascending=False, inplace=True)

        plt.plot([i for i in range(len(df_threshold))], df_threshold['number_keywords'], label=threshold)
        plt.legend(loc='upper left')
        plt.xlabel('Cluster number')
        plt.ylabel('Number of keywords')
        plt.show()


def weight_of_top_words(lda_instance, top_n_words=10):

    top_weights = []
    topics = lda_instance.lda_model.show_topics(formatted=False, num_topics=-1, num_words=top_n_words)
    for _, topic in topics:
        weight = sum([weight for _, weight in topic])
        top_weights.append(weight)

    indices = np.arange(len(topics))
    width = 0.5

    fig, ax = plt.subplots()
    ax.bar(indices - width/2, top_weights, width, color = 'lightcoral')
    ax.set_xticks(indices)

    plt.xlabel('Topic')
    plt.ylabel('Sum of {} largest weights'.format(top_n_words))
    plt.title('Total probability of top {} words in each topic'.format(top_n_words))
    plt.xlim(-0.5, top_n_words-0.5)
    plt.ylim(0, 0.75)
    plt.xticks(indices[::2])
    plt.show()


def plt_weight_words(lda_instance, params, top_n_words=20):

    def split_array_in_chunks(l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    array_topic_numbers = split_array_in_chunks(range(lda_instance.lda_model.num_topics), 8)
    topics = lda_instance.lda_model.show_topics(num_topics=-1, num_words=top_n_words, formatted=False)

    for i, topic_numbers in enumerate(array_topic_numbers):

        plt.figure()
        for j, word_weight in [topics[k] for k in topic_numbers]:
            weights = [weight for _, weight in word_weight]
            plt.plot(range(top_n_words), weights, label=j)
            plt.legend()

        plt.xlabel('Word rank')
        plt.ylabel('Weight')
        plt.title('Weights of top {} words in each topic'.format(top_n_words))
        plt.xticks(range(top_n_words+1)[::2])
        plt.xlim(-1, top_n_words)

        folder_path = os.path.join(params['paths']['lda'], 'topic_sizes_' +
                                   str(lda_instance.lda_model.num_topics))
        file_path = os.path.join(folder_path, '{}.png'.format(i))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
def _return_array(array, n_topics=80):
    """
    Utility function transforming the projections as returned by the LDA
    module into a 1xT numpy array, where T is the number of topics.
    """
    output = np.array(
        [[sum([el[1] for el in row if el[0] == topic]) for topic in range(1, n_topics + 1)] for row in array])
    return output


def _smooth(y, smoothing_points):
    box = np.ones(smoothing_points) / smoothing_points
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth
