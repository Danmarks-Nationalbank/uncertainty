# Uncertainty and the Real Economy: Evidence from Denmark

This repo contains the code underlying the main results in [Uncertainty and the Real Economy: Evidence from Denmark](https://www.nationalbanken.dk/en/publications/Pages/2020/11/Working-Paper-Uncertainty-and-the-real-economy-Evidence-from-Denmark.aspx). The authors would like to thank Rastin Matin for helping to set up the LDA model. Data access is restricted, please contact the authors for more information. If you use this code, please cite:

Bess, M., [Grenestam, E.](http://erikgrenestam.se/research/), [Pedersen, J.](https://www.nationalbanken.dk/en/research/economists/Pages/Jesper-Pedersen.aspx) and [Tang-Andersen Martinello, A.](https://alemartinello.com/) (2020). [Uncertainty and the real economy: Evidence from Denmark.](https://www.nationalbanken.dk/en/publications/Pages/2020/11/Working-Paper-Uncertainty-and-the-real-economy-Evidence-from-Denmark.aspx) Working paper 165, Danmarks Nationalbank.

## Replicating the environment
The python environment can be replicated via the `requirements.txt` file. For example, using pip on a linux machine, from the root folder:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Replicating the paper
The notebook `scripts/working_paper.ipynb` can be used to create the uncertainty indices and accompanying graphs and tables. 

The SVAR model is estimated using Matlab and the BEAR 4.0 toolbox, which can be downloaded from the [ECB website](https://www.ecb.europa.eu/pub/research/working-papers/html/bear-toolbox.en.html). The data is available in `data/` and replication files are available in `scripts/`: 

* `bear_settings.m` estimates the SVAR model.  
* `manyirfs.m` reproduced the robustness checks presented by figure 5.
* `hist_decomp.m` produces the historical decomposition presented by figures 6 and A8.

## Indices
Our index and our Danish version of the Baker, Bloom and Davis (2016) index are available as csv files in `data/`. Last update: February 2021.
