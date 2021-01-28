# Uncertainty and the Real Economy: Evidence from Denmark

This repo contains the code underlying the main results in [Uncertainty and the Real Economy: Evidence from Denmark](https://www.nationalbanken.dk/en/publications/Pages/2020/11/Working-Paper-Uncertainty-and-the-real-economy-Evidence-from-Denmark.aspx). The notebook `scripts/working_paper.ipynb` can be used to create the uncertainty indices and accompanying graphs and tables. The authors would like to thank Rastin Matin for helping to set up the LDA model. Data access is restricted, please contact the authors for more information. If you use this code, please cite:

Bess, M., Grenestam, E., Pedersen, J. and Tang-Andersen Martinello, A. (2020). [Uncertainty and the real economy: Evidence from Denmark.](https://www.nationalbanken.dk/en/publications/Pages/2020/11/Working-Paper-Uncertainty-and-the-real-economy-Evidence-from-Denmark.aspx) Working paper 165, Danmarks Nationalbank.

## Replicating the environment
The python environment can be replicated via the `requirements.txt` file. For example, using pip in a linux machine, from the root folder:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Replicating the indexes paper
The data has been provided by the Danish newspaper Børsen for the development of this index. However, provided a database of (danish) articles is obtained, the indexes described in the paper can be obtained by running the code in the [working_paper](https://github.com/Danmarks-Nationalbank/uncertainty/blob/8-publication/scripts/working_paper.ipynb) notebook.

The figures relative to the text analysis part of the paper can also be replicated through the same notebook  

## Matlab part
Add info on how to replicate the part by Mikkel
