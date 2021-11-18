# -*- coding: utf-8 -*-

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gensim
import pandas as pd
import os
import warnings
from sqlalchemy import create_engine



from src.fui.ldatools import get_topics
from src.fui.indices import import_word_counts
from src.fui.utils import params

class LDA:
    def __init__(self, num_topics=None):
        self.dictionary = None
        self.articles = []
        self.article_ids = []
        self.lda_model = None
        if num_topics is not None:
            self.load_model(num_topics)
                
        #if params().options['lda']['log']:
        import logging
        try:
            os.remove(params().paths['lda']+'lda_log.txt')
        except (FileNotFoundError, PermissionError):
            pass
        logging.basicConfig(filename=params().paths['lda']+'lda_log.txt',
                            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    def __iter__(self):
        for line in self.articles:
            yield self.bigram_phraser[line.split()]

    def load_articles(self, chunksize=100000):
        print("Loading preprocessed data from SRV9DNBDBM078")
        df = import_word_counts(chunksize=chunksize)
        for c in df:
            self.article_ids.extend(c['DNid'].values.tolist())
            self.articles.extend(c['body'].values.tolist())
        print("\t{} documents loaded".format(len(self.articles)))

    def _get_unprocessed_articles(self, chunksize=None):
        sql = """
            select a.DNid
    		,a.body
    		from area060.article_word_counts a
    		LEFT JOIN area060.article_topics b on a.DNid = b.DNid
    		where b.DNid IS NULL
    		"""
        engine = create_engine(
            'mssql+pyodbc://SRV9DNBDBM078/workspace01?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
        df = pd.read_sql(sql, con=engine, chunksize=chunksize)
        return df

    def load_new_articles(self, chunksize=None):
        print("Loading unprocessed articles from sql")
        df = self._get_unprocessed_articles(chunksize=chunksize)
        if chunksize is None:
            self.article_ids = df['DNid'].values.tolist()
            self.articles = df['body'].values.tolist()
        else:
            for c in df:
                self.article_ids.extend(c['DNid'].values.tolist())
                self.articles.extend(c['body'].values.tolist())
        if len(self.articles) == 0:
            print("No new articles to process for LDA.")
        else:
            print("\t{} new articles found, processing for LDA...".format(len(self.articles)))
            get_topics(self)

    def create_dictionary(self, load_bigrams=True, unwanted_words=None, keep_words=None):
        # Create dictionary (id2word)
        file_path = os.path.join(params().paths['lda'], params().filenames['lda_dictionary'])

        # Load bigram phraser
        if load_bigrams:
            self.load_bigrams()

        try:
            self.dictionary = gensim.corpora.Dictionary.load(file_path)
            print("Loaded pre-existing dictionary")
        except FileNotFoundError:
            print("Dictionary not found, creating from scratch")

            self.dictionary = gensim.corpora.Dictionary(articles for articles in self)

            self.dictionary.filter_extremes(no_below=params().options['lda']['no_below'],
                                                    no_above=params().options['lda']['no_above'],
                                                    keep_n=params().options['lda']['keep_n'],
                                                    keep_tokens=keep_words)
            if unwanted_words is None:
                unwanted_words = []
            unwanted_ids = [k for k, v in self.dictionary.items() if v in unwanted_words]
            self.dictionary.filter_tokens(bad_ids=unwanted_ids)
            self.dictionary.compactify()
            self.dictionary.save(file_path)
        print("\t{}".format(self.dictionary))

    def create_corpus(self):
        #Helper-class to create BoW-corpus "lazily"
        corpus_bow = LdaIterator(self)

        file_path = os.path.join(params().paths['lda'], 'corpus.mm')
        try:
            self.Corpus = gensim.corpora.MmCorpus(file_path)
            print("Loaded pre-existing corpus")
        except FileNotFoundError:
            print("Corpus not found, creating from scratch")
            if not hasattr(self, 'bigram_phraser'):
                self.load_bigrams()

            # Serialize corpus (either BoW or tf-idf)
            if not params().options['lda']['tf-idf']:
                print("\tSerializing corpus, BoW")
                gensim.corpora.MmCorpus.serialize(file_path, corpus_bow)
            else:
                print("\tSerializing corpus, tf-idf")
                tfidf = gensim.models.TfidfModel(corpus_bow)
                train_corpus_tfidf = tfidf[corpus_bow]
                gensim.corpora.MmCorpus.serialize(file_path, train_corpus_tfidf)

            self.Corpus = gensim.corpora.MmCorpus(file_path)
    
    def load_bigrams(self):
        if os.path.isfile(os.path.join(params().paths['lda'],'phrases.pkl')):
            phrases = gensim.utils.SaveLoad.load(os.path.join(params().paths['lda'],'phrases.pkl'))
            self.bigram_phraser = gensim.models.phrases.Phraser(phrases)
            print("Bigram phraser loaded")
        else:
            print("Bigram phraser not found, training")
            self.load_processed_text()
            phrases = gensim.models.phrases.Phrases(self.articles, params().options['lda']['no_below'], threshold=100)
            frozen_model = phrases.freeze()
            frozen_model.save(os.path.join(params().paths['lda'],'phrases.pkl'), separately=None, sep_limit=10485760, ignore=frozenset([]), pickle_protocol=2)
            self.bigram_phraser = gensim.models.phrases.Phraser(frozen_model)
            print("Bigram phraser loaded")

    def get_doc_topics(self):
        self.Corpus = gensim.corpora.mmcorpus.MmCorpus(os.path.join(params().paths['lda'], 'corpus.mm'))
        for bow, id in zip(self.Corpus, self.article_ids):
            # try:
            #     bow = self.dictionary.doc2bow(self.bigram_phraser[article.split()])
            # except AttributeError:
            #     print(article)
            #     continue
            yield [self.lda_model.get_document_topics(bow, minimum_probability=0.01), id]

    def _worker(self, pair):
        topics = self.lda_model.get_document_topics(pair[0], minimum_probability=0.0)
        line = [pair[1]]
        line.extend([i[1] for i in topics])
        return line

    def get_topics_list(self):
        return list(self.get_topics())
        #bow = self.dictionary.doc2bow(self.bigram_phraser[text.split()])
        #return self.lda_model.get_document_topics(bow, minimum_probability=0.0)

    def load_model(self, num_topics):
        try:
            folder_path = os.path.join(params().paths['root'], params().paths['lda'], 'lda_model_' + str(num_topics))
            file_path = os.path.join(folder_path, 'trained_lda')
            self.lda_model = gensim.models.LdaMulticore.load(file_path)
            print("LDA-model with {} topics loaded".format(num_topics))
        except FileNotFoundError:
            print("Error: LDA-model not found")

class LdaIterator:
    def __init__(self, lda_instance):
        self.lda = lda_instance

    def __iter__(self):
        for l in self.lda.articles:
            yield self.lda.dictionary.doc2bow(self.lda.bigram_phraser[l.split()])
