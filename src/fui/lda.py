# -*- coding: utf-8 -*-

import gensim
import h5py
import numpy as np
import os
import re
import pandas as pd
import pickle
import random
import warnings
import json

from src.fui.ldatools import preprocess
from src.fui.utils import timestamp

from collections import Counter
from datetime import timedelta, datetime
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer

class LDA:
    def __init__(self, files_list, params, test=False):
        self.dictionary = None
        self.articles = []
        self.article_id = []
        self.SerializedCorpus = None
        self.test = test        
        self.files_list = files_list
        self.params = params
                
        if self.params['options']['lda']['log']:
            import logging
            logging.basicConfig(filename=self.params['paths']['lda']+'lda_log.txt',
                                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            warnings.filterwarnings('ignore', category=DeprecationWarning)

    def __iter__(self):
        for line in self.articles:
            yield line.split()

    def load_and_clean_body_text(self):
        print("No existing pre-processed data found. Loading {} file(s) for preprocessing".format(len(self.files_list)))
        for f in self.files_list:
            with open(f, 'rb') as f_in:
                df = pickle.load(f_in)

                try:
                    self.articles.extend(list(df['ArticleContents'].values))
                    self.article_id.extend(list(df['ID'].values))
                except KeyError:
                    print("Skipped {}, file doesn't contain any body-text".format(f))

        # Perform LDA on smaller sample, just for efficiency in case of testing...
        if self.test is True:
            random.seed(1)
            test_idx = random.sample(range(0, len(self.articles)), self.params['options']['lda']['test_size'])
            self.articles = [self.articles[i] for i in test_idx]
            self.article_id = [self.article_id[i] for i in test_idx]

        # Pre-process LDA-docs
        if len(self.articles):
            print("\tProcessing {} documents for LDA".format(len(self.articles)))
            with Pool(self.params['options']['threads']) as pool:
                self.articles = pool.map(preprocess, self.articles)

            print("\tSaving cleaned documents")
            folder_path = self.params['paths']['lda']
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, self.params['filenames']['lda_cleaned_text'])
            with h5py.File(file_path, 'w') as hf:
                data = np.array(list(zip(self.article_id, self.articles)), dtype=object)
                string_dt = h5py.special_dtype(vlen=str)
                hf.create_dataset('parsed_strings', data=data, dtype=string_dt)

    def load_processed_text(self):
        try:
            with h5py.File(os.path.join(self.params['paths']['lda'], self.params['filenames']['lda_cleaned_text']), 'r') as hf:
                print("Loading processed data from HDF-file")
                hf = hf['parsed_strings'][:]
                self.article_id = list(zip(*hf))[0]
                self.articles = list(zip(*hf))[1]
                print("\t{} documents loaded".format(len(self.articles)))
                return 1
        except OSError:
            return 0

    @staticmethod
    def get_topics(lda_model, dictionary, text):
        bow = dictionary.doc2bow(text.split())
        return lda_model.get_document_topics(bow, minimum_probability=0.0)

