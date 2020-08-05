import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\projects\\FUI\\src')
sys.path.insert(1, 'C:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
from src.fui.utils import params
from src.fui.indices import LDAIndexer, BloomIndexer
from src.fui.indices import extend_dict_w2v, uncertainty_count
import os
import codecs
import json

if __name__ == '__main__':
    
    #df = parse_raw_data()

    #U_set = set(list(extend_dict_w2v("uncertainty", n_words=20).values())[0])
    #print(params().paths['enriched_news'])
    #uncertainty_count()
    #uncertainty_count(extend=False)
    num_topics = 90
    #
    # label_path = os.path.join(params().paths['topic_labels'],
    #                           'labels' + str(num_topics) + '.json')
    # with codecs.open(label_path, 'r', encoding='utf-8-sig') as f:
    #     labels = json.load(f)

    # for t in range(0,num_topics,1):
    #     topic = LDAIndexer(name='topic_'+str(t))
    #     idx = topic.build(num_topics=num_topics, topics=t, labels=None, topic_thold=0.0, frq='Q', u_weight=True)
    #     topic.plot_index(title=labels[str(t)])
    #     print(idx.head())

    # international = LDAIndexer(name='ep_int')
    # idx = international.build(num_topics=num_topics, topics=['EP_int'], topic_thold=0.5, frq='M', u_weight=True)
    # international.plot_index(title='Economic policy uncertainty, international, monthly')
    # print(idx.head())
    #
    # international = LDAIndexer(name='ep_int')
    # international.build(num_topics=num_topics, topics=['EP_int'], topic_thold=0.5, frq='Q')
    # international.plot_index(title='Economic policy uncertainty, international, quarterly')
    #
    # domestic = LDAIndexer(name='ep_dk')
    # domestic.build(num_topics=num_topics, topics=['EP_dk'],topic_thold=0.5,frq='M')
    # domestic.plot_index(title='Economic policy uncertainty, domestic, monthly')
    #
    # domestic = LDAIndexer(name='ep_dk')
    # domestic.build(num_topics=num_topics, topics=['EP_dk'], topic_thold=0.5, frq='Q')
    # domestic.plot_index(title='Economic policy uncertainty, domestic, quarterly')
    #
    # agg = LDAIndexer(name='ep_all')
    # agg.build(num_topics=num_topics, topics=['EP'], topic_thold=0.5, frq='M')
    # agg.plot_index(title='Economic policy uncertainty, monthly')
    #
    bloom = BloomIndexer(name='bloom')
    bloom.build(logic='EandPandU', bloom_dict_name='bloom', extend=False)

    agg = LDAIndexer(name='ep_all')
    agg.build(num_topics=num_topics,topics=['EP'],topic_thold=0.5,frq='Q')
    agg.plot_index(title='Economic policy uncertainty, quarterly', plot_bloom=True, plot_vix=True)