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
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from cycler import cycler


def plot_index(idx1, idx2, plot_vix=False, plot_bloom=False, annotate=True, title=None):
    """
    Plot index from df column named "idx_norm".
    Args:
    plot_vix (bool): Add series of vix and vdax to plot.
    plot_hh (bool): Plot household equity transactions (not ready).
    annotate (bool): Add event annotation to plot.
    title (str): Plot title.
    Returns: plt objects (figure,axes).
    """

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator((1, 4, 7, 10))
    years_fmt = mdates.DateFormatter('%Y')

    out_path = params().paths['indices']
    c = cycler(
        'color',
        [
            (0 / 255, 123 / 255, 209 / 255),
            (146 / 255, 34 / 255, 156 / 255),
            (196 / 255, 61 / 255, 33 / 255),
            (223 / 255, 147 / 255, 55 / 255),
            (176 / 255, 210 / 255, 71 / 255),
            (102 / 255, 102 / 255, 102 / 255)
        ])
    plt.rcParams["axes.prop_cycle"] = c

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(idx1.index, idx1['idx'], label='Børsen Uncertainty Index (curated)')
    ax.plot(idx2.index, idx2['idx'], label='Børsen Uncertainty Index (broad)')
    if title:
        ax.title.set_text(title)
    if plot_vix:
        vix = self.load_vix(self.frq)
        # ax.plot(v1x.index, v1x.v1x, label='VDAX-NEW')
        ax.plot(vix.index, vix.vix, label='VIX')
    if plot_bloom:
        bloom = pd.read_csv(params().paths['indices'] + 'bloom_' + self.frq + '.csv',
                            names=['date', 'bloom'], header=0)
        bloom['date'] = pd.to_datetime(bloom['date'])
        bloom.set_index('date', inplace=True)
        ax.plot(bloom.index, bloom.bloom, label='Baker & Bloom')

    ax.legend(frameon=False, loc='upper left')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    if annotate:
        ax.axvspan(xmin=datetime(2000, 1, 31), xmax=datetime(2000, 5, 31),
                   color=(102 / 255, 102 / 255, 102 / 255), alpha=0.3)
        ax.annotate("Dot-com \n crash", xy=(datetime(2000, 3, 31), 0.8),
                    xycoords=('data', 'axes fraction'), fontsize='large', ha='center')
        ax.axvspan(xmin=datetime(2011, 3, 1), xmax=datetime(2012, 11, 30),
                   color=(102 / 255, 102 / 255, 102 / 255), alpha=0.3)
        ax.annotate("Debt crisis", xy=(datetime(2012, 2, 15), 0.96),
                    xycoords=('data', 'axes fraction'), fontsize='large', ha='center')
        # ax.axvspan(xmin=datetime(2018,3,1), xmax=datetime(2019,12,1),
        #           color=(102/255, 102/255, 102/255), alpha=0.3)
        # ax.annotate("Trade war", xy=(datetime(2019,2,15), 0.97),
        #            xycoords=('data', 'axes fraction'), fontsize='large', ha='center')
        ax.axvspan(xmin=datetime(2020, 2, 1), xmax=datetime(2020, 7, 1),
                   color=(102 / 255, 102 / 255, 102 / 255), alpha=0.3)
        ax.annotate("COVID-19", xy=(datetime(2020, 3, 15), 0.96),
                    xycoords=('data', 'axes fraction'), fontsize='large', ha='center')

        dates_dict = {'Euro \nreferendum': ('2000-09-28', 0.5),
                      '9/11': ('2001-09-11', 0.7),
                      '2001\n election': ('2001-11-20', 0.8),
                      'Invasion of Iraq': ('2003-03-19', 0.9),
                      '2005\nelection': ('2005-02-08', 0.8),
                      'Northern Rock\n bank run': ('2007-09-14', 0.9),
                      '2007\n election': ('2007-11-13', 0.8),
                      'Lehman Brothers': ('2008-09-15', 0.97),
                      '2010 Flash Crash': ('2010-05-06', 0.9),
                      '2011 election': ('2011-09-15', 0.8),
                      '"Whatever\n it takes"': ('2012-07-26', 0.7),
                      '2013 US Gov\n shutdown': ('2013-10-15', 0.9),
                      # "'DKK pressure\n crisis': ('2015-02-15', 0.7),
                      '2015\n election': ('2015-06-18', 0.9),
                      'Migrant\n crisis': ('2015-09-15', 0.8),
                      'Brexit': ('2016-06-23', 0.75),
                      'US\n election': ('2016-11-08', 0.9),
                      # 'Labor parties\n agreement': ('2018-04-15', 0.7),
                      'Danske Bank\n money laundering': ('2018-09-15', 0.9),
                      '2018 US gov\n shutdown': ('2018-12-10', 0.8)}

        for l, d in zip(dates_dict.keys(), dates_dict.values()):
            date = datetime.strptime(d[0], "%Y-%m-%d")
            ax.axvline(x=date, color=(102 / 255, 102 / 255, 102 / 255), alpha=0.3, linewidth=2)
            ax.annotate(l, xy=(date, d[1]), xycoords=('data', 'axes fraction'),
                        fontsize='medium', ha='center')
        # corr = _calc_corr(vix,idx[idx_name])
        # ax.text(0.80, 0.95, 'Correlation with VIX: %.2f' % round(corr,2) , transform=ax.transAxes)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Standard deviations", fontsize='large')
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{out_path}double_Q_plot.png', dpi=300)
    return fig, ax


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
    #bloom = BloomIndexer(name='bloom')
    #bloom.build(logic='EandPandU', bloom_dict_name='bloom', extend=False)

    # agg = LDAIndexer(name='ep_all')
    # agg.build(num_topics=num_topics,topics=['EP'],topic_thold=0.5,frq='Q')
    # agg.plot_index(title='Economic policy uncertainty, quarterly', plot_bloom=True, plot_vix=True)
    #
    agg = LDAIndexer(name='ep_all')
    agg.build(num_topics=num_topics,topics=['EP'],topic_thold=0.5,frq='Q')
    agg.plot_index(plot_bloom=True, plot_vix=True)

    # mort = LDAIndexer(name='mortgage')
    # mort.build(num_topics=num_topics,topics=['mortgage'],topic_thold=0.5,frq='Q')
    # mort.plot_index(plot_bloom=False, plot_vix=False, annotate=False)
    #
    broad = LDAIndexer(name='broad')
    broad.build(num_topics=num_topics,topics=['broad'],topic_thold=0.5,frq='Q')
    #broad.plot_index(title='Economic uncertainty, quarterly', plot_bloom=True, plot_vix=True, annotate=True)

    #fin = LDAIndexer(name='financial')
    #fin.build(num_topics=num_topics,topics=['financial'],topic_thold=0.5,frq='Q')
    #fin.plot_index(title='Financial uncertainty, quarterly', plot_bloom=True, plot_vix=True, annotate=True)

    #nonfin = LDAIndexer(name='nonfin')
    #nonfin.build(num_topics=num_topics,topics=['non-financial'],topic_thold=0.5,frq='Q')
    #nonfin.plot_index(title='Non-financial uncertainty, quarterly', plot_bloom=True, plot_vix=True, annotate=True)

    fig, ax = plot_index(agg.idx, broad.idx)

 