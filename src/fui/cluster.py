import os
import sys
#hacky spyder crap
#sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
sys.path.insert(1, 'D:\\projects\\FUI')
sys.path.insert(1, 'D:\\projects\\FUI\\env\\Lib\\site-packages')
sys.path.insert(1, 'C:\\Users\\EGR\\AppData\\Roaming\\Python\\Python37\\site-packages')
from itertools import cycle, islice
import json
import gensim
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist

class ClusterTree():
    """
    Build clusters from topic models using scipy.cluster.hierarchy.
    :num_topics: Used to load a pre-trained topic model.
    :metric: and :method: Used for HAC.
    """
    
    def __init__(self, num_topics, params, metric='jensenshannon', method='ward'):
        """
        Saves linkage matrix :Z: and :nodelist:
        """
        
        self.num_topics = num_topics
        self.metric = metric
        self.method = method
        self.params = params
        
        folder_path = os.path.join(params['paths']['root'],params['paths']['lda'], 
                                   'lda_model_' + str(self.num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        self.lda_model = gensim.models.LdaMulticore.load(file_path)
        topics = self.lda_model.get_topics()
        y = pdist(topics, metric=self.metric)
        self.Z = hierarchy.linkage(y, method=self.method)
        rootnode, self.nodelist = hierarchy.to_tree(self.Z,rd=True)
    
    def _get_children(self, id):
        """
        Recursively get all children of parent node :id:
        """
        if not self.nodelist[id].is_leaf():
            for child in [self.nodelist[id].get_left(), self.nodelist[id].get_right()]:
                yield child
                for grandchild in self._get_children(child.id):
                    yield grandchild
                    
    def children(self):
        """
        Returns a dict with k, v: parent: [children]. Does not include leaf nodes.
        """
        self.children = {}
        for i in range(self.num_topics,len(self.nodelist)):
            self.children[i] = [child.id for child in self._get_children(i)]
        return self.children
    
    def _colorpicker(self,k):
        """
        Returns an NB color to visually group similar topics in dendrogram
        
        """
        NB_colors = [(0, 123, 209),
            (146, 34, 156),
            (196, 61, 33),
            (223, 147, 55),
            (176, 210, 71)] 
        
        # Get flat clusters for grouping
        self.flat_clusters(n=self.colors)
        clist = list(islice(cycle(NB_colors), len(self.L)))
        for c,i in enumerate(list(self.L)):
            if k in [child.id for child in self._get_children(i)]:
                color = clist[c]
                return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    
        # Gray is default color
        return "#666666"
    
    def _labelpicker(self,k):
        labels = parse_topic_labels(self.num_topics,self.params)
        return ', '.join(labels[str(k)])
        
    def dendrogram(self,w=10,h=10,colors=10,color_labels=True):
        """
        Draws dendrogram
        :no_plot: Don's render figure. Use self.graph to render figure later
        :colors: Approx. no of color clusters in figure.
        """
        
        self.colors = colors
        fig = plt.figure(figsize=(w,h)) 
        plt.title("Topic Dendrogram")
        plt.xlabel("Distance")
        plt.ylabel("Topic")
        
        R = hierarchy.dendrogram(self.Z,
                       orientation='right',
                       #labels=labelList,
                       distance_sort='descending',
                       show_leaf_counts=False,
                       no_plot=False,
                       leaf_label_func=self._labelpicker,
                       #color_threshold=2.0*np.max(self.Z[:,2])
                       link_color_func=self._colorpicker)
        
        if color_labels:
            ax = plt.gca()
            self.cluster_idxs = {}
            for c, pi in zip(R['color_list'], R['icoord']):
                for leg in pi[1:3]:
                    i = (leg - 5.0) / 10.0
                    if abs(i - int(i)) < 1e-5:
                        self.cluster_idxs[int(i)] = c
            
            ylbls = ax.get_ymajorticklabels()
            for c,y in enumerate(ylbls):
                y.set_color(self.cluster_idxs[c])
       
        plt.tight_layout()
        fig.savefig(os.path.join(self.params['paths']['root'],self.params['paths']['lda'], 
                                   'dendrogram'+str(self.num_topics)+'.pdf'), dpi=300)

        plt.show()

    
    def flat_clusters(self,n=8,init=1,criterion='maxclust'):
        """
        Returns flat clusters from the linkage matrix :Z:
        """
        if criterion is 'distance':
            self.T = hierarchy.fcluster(self.Z,init,criterion='distance')
            a = 0
            while a < 20:
                if self.T.max() < n:
                    init = init-0.02
                    a += 1
                elif self.T.max() > n:
                    init = init+0.02
                    a += 1
                else:
                    self.L, self.M = hierarchy.leaders(self.Z,self.T)
                    return self.T
                self.T = hierarchy.fcluster(self.Z,init,criterion='distance')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        elif criterion is 'inconsistent':
            self.T = hierarchy.fcluster(self.Z,criterion='inconsistent')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        elif criterion is 'maxclust':
            self.T = hierarchy.fcluster(self.Z,t=n,criterion='maxclust')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        else:
            print('Criteria not implemented')
            return 0

def parse_topic_labels(num_topics,params):
    """
    reads hand labeled topics from json file.
    
    """
    label_path = os.path.join(params['paths']['root'],params['paths']['topic_labels'], 
                        'labels'+str(num_topics)+'.json')
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels             
            
def load_model(lda_instance, num_topics, params):
    try:
        folder_path = os.path.join(params['paths']['root'],params['paths']['lda'], 'lda_model_' + str(num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        lda_instance.lda_model = gensim.models.LdaMulticore.load(file_path)
        print("LDA-model with {} topics loaded".format(num_topics))
    except FileNotFoundError:
        print("Error: LDA-model not found")
        lda_instance.lda_model = None