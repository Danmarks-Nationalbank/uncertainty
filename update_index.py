import sys, getopt
from src.fui.preprocessing import parse_for_lda
from src.fui.borsen_api import upload_new_articles
from src.fui.ldatools import get_topics
from src.fui.indices import LDAIndexer, BloomIndexer
from src.fui.lda import LDA
from pathlib import Path

#key = 'GMuZMqJne9fra7FJNhBxH5BqcWuJoRjBfrbknnG0EF/UNKyGVCr6pA=='

def main(argv):
    key = ''
    try:
        opts, args = getopt.getopt(argv, "k:")
    except getopt.GetoptError:
        print('test.py -k <api_key>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-k':
            key = arg
    print(f'API key is is {key}')

    #Download new articles and upload to area060
    upload_new_articles(key=key)

    #Preprocess new articles
    parse_for_lda(only_new=True)

    #Load LDA model for preprocessing
    mylda = LDA(num_topics=90)
    mylda.load_new_articles(chunksize=100000)
    mylda.create_dictionary()

    # Build main index
    main_idx = LDAIndexer(name='ep_all')
    main_idx.build(num_topics=90,topics=['EP'],topic_thold=0.5,frq='Q')

    bloom_idx = BloomIndexer(name='bloom')
    bloom_idx.build(logic='EandPandU', bloom_dict_name='bloom')

    main_idx.plot_index(plot_bloom=True, plot_vix=False)

    main_idx.build(num_topics=90,topics=['EP'],topic_thold=0.5,frq='M')


if __name__ == "__main__":
    print(str(Path(__file__).parent))
    main(sys.argv[1:])