import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'C:\\projects\\FUI\\src')
sys.path.insert(1, 'C:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
from fui.utils import params
from fui.indices import LDAIndexer, BloomIndexer
from fui.indices import extend_dict_w2v, uncertainty_count

if __name__ == '__main__':
    
    #df = parse_raw_data()
    #U_set = set(list(extend_dict_w2v("uncertainty", n_words=20).values())[0])
    #print(params().paths['enriched_news'])
    #uncertainty_count()
    #uncertainty_count(extend=False)
    
    international = LDAIndexer(name='ep_int')
    international.build(num_topics=80,sample_size=0,topics=['EP_int'],topic_thold=0.0,frq='M')
    international.plot_index(title='Economic policy uncertainty, international', plot_vix=False, plot_hh=True)
#    
#    international = LDAIndexer(name='ep_int')
#    international.build(num_topics=80,sample_size=0,topics=['EP_int'],topic_thold=0.0,frq='Q')
#    
#    domestic = LDAIndexer(name='ep_dk')
#    domestic.build(num_topics=80,sample_size=0,topics=['EP_dk'],topic_thold=0.0,frq='M')
#    domestic.plot_index(title='Economic policy uncertainty, domestic')
#    
#    domestic = LDAIndexer(name='ep_dk')
#    domestic.build(num_topics=80,sample_size=0,topics=['EP_dk'],topic_thold=0.0,frq='Q')
#    
#    agg = LDAIndexer(name='ep_all')
#    agg.build(num_topics=80,sample_size=0,topics=['EP'],topic_thold=0.0,frq='M')
#    agg.plot_index(title='Economic policy uncertainty, all')
##    
#    agg = LDAIndexer(name='ep_all')
#    agg.build(num_topics=80,sample_size=0,topics=['EP'],topic_thold=0.02,frq='Q')
    
    #xidx = LDAIndexer(name='xidx_int_f')
    #xidx.build(num_topics=80,labels='meta_topics',sample_size=0,xsection=['International F','International politics'],xsection_thold=0.1,u_weight=True)
    #xidx.plot_index(title='International uncertainty')

    
    #bloom = BloomIndexer(name='bloom')
    #bloom.build(logic='EandPandU',bloom_dict_name='bloom_extended',extend=True)