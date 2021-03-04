# Uncertainty and the Real Economy: Evidence from Denmark

This repo contains the code underlying the main results in [Uncertainty and the Real Economy: Evidence from Denmark](https://www.nationalbanken.dk/en/publications/Pages/2020/11/Working-Paper-Uncertainty-and-the-real-economy-Evidence-from-Denmark.aspx). The authors would like to thank Rastin Matin for helping to set up the LDA model. Data access is restricted, please contact the authors for more information. If you use this code, please cite:

Bess, M., Grenestam, E., Pedersen, J. and Tang-Andersen Martinello, A. (2020). [Uncertainty and the real economy: Evidence from Denmark.](https://www.nationalbanken.dk/en/publications/Pages/2020/11/Working-Paper-Uncertainty-and-the-real-economy-Evidence-from-Denmark.aspx) Working paper 165, Danmarks Nationalbank.

## Replicating the environment
The python environment can be replicated via the `requirements.txt` file. For example, using pip on a linux machine, from the root folder:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Replicating the paper
The notebook `scripts/working_paper.ipynb` can be used to create the uncertainty indices and accompanying graphs and tables. The SVAR model is estimated using Matlab and the BEAR 4.0 toolbox, which can be downloaded from the [ECB website](https://www.ecb.europa.eu/pub/research/working-papers/html/bear-toolbox.en.html). The data is available in `data/` and replication files are available in `scripts/`. `BearSettings.m` estimates the model and `HistDecomp.m` produces the historical decomposition presented by figures 6 and A8 in the paper.
