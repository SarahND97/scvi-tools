# This is a fork of the original SCVI-TOOLS repo made for the thesis "Hybrid Variational Autoencoder Clustering of Single-Cell RNA-seq Data"
# Written by Sarah Narrowe Danielsson

The hybrid model created for the thesis can be found in `scvi-tools/scvi/model/_hybridvi.py` and `scvi-tools/scvi/module/_hybridvae.py`.
The results of the thesis can be found in `scvi-tools/output/` and `scvi-tools/cross_valid_results/`. 

# Rapid development of novel probabilistic models

scvi-tools contains the building blocks to develop and deploy novel probablistic
models. These building blocks are powered by popular probabilistic and
machine learning frameworks such as [PyTorch
Lightning](https://www.pytorchlightning.ai/) and
[Pyro](https://pyro.ai/). For an overview of how the scvi-tools package
is structured, you may refer to [this](https://docs.scvi-tools.org/en/stable/user_guide/background/codebase_overview.html) page.

We recommend checking out the [skeleton
repository](https://github.com/YosefLab/scvi-tools-skeleton) as a
starting point for developing new models into scvi-tools.

# Basic installation

For conda,
```
conda install scvi-tools -c bioconda -c conda-forge
```
and for pip,
```
pip install scvi-tools
```
Please be sure to install a version of [PyTorch](https://pytorch.org/) that is compatible with your GPU (if applicable).

# Resources

-   Tutorials, API reference, and installation guides are available in
    the [documentation](https://docs.scvi-tools.org/).
-   For discussion of usage, check out our
    [forum](https://discourse.scvi-tools.org).
-   Please use the [issues](https://github.com/YosefLab/scvi-tools/issues) to submit bug reports.
-   If you\'d like to contribute, check out our [contributing
    guide](https://docs.scvi-tools.org/en/stable/contributing/index.html).
-   If you find a model useful for your research, please consider citing
    the corresponding publication (linked above).

# Reference

```
@article{Gayoso2021scvitools,
	author = {Gayoso, Adam and Lopez, Romain and Xing, Galen and Boyeau, Pierre and Wu, Katherine and Jayasuriya, Michael and Mehlman, Edouard and Langevin, Maxime and Liu, Yining and Samaran, Jules and Misrachi, Gabriel and Nazaret, Achille and Clivio, Oscar and Xu, Chenling and Ashuach, Tal and Lotfollahi, Mohammad and Svensson, Valentine and da Veiga Beltrame, Eduardo and Talavera-Lopez, Carlos and Pachter, Lior and Theis, Fabian J and Streets, Aaron and Jordan, Michael I and Regier, Jeffrey and Yosef, Nir},
	title = {scvi-tools: a library for deep probabilistic analysis of single-cell omics data},
	year = {2021},
	doi = {10.1101/2021.04.28.441833},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/04/29/2021.04.28.441833},
	eprint = {https://www.biorxiv.org/content/early/2021/04/29/2021.04.28.441833.full.pdf},
	journal = {bioRxiv}
}
```
