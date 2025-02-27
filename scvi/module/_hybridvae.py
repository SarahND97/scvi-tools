# -*- coding: utf-8 -*-
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi import _CONSTANTS
from scvi._compat import Literal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial, HypersphericalUniform, VonMisesFisher
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, EncoderHYBRIDVI, one_hot

torch.backends.cudnn.benchmark = True

class HYBRIDVAE(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is an implementation of the scVI model described in [Lopez18]_

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        gene_indexes: Iterable[int],
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent_normal: int = 10,
        n_latent_von_mises: int = 2,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        latent_distribution: str = "hybrid",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent_von_mises = n_latent_von_mises
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.gene_indexes = gene_indexes

        self.use_observed_lib_size = use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        n_input_encoder_normal = (n_input - len(self.gene_indexes)) + n_continuous_cov * encode_covariates
        n_input_encoder_von_mises =  len(self.gene_indexes) + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        
        self.z_encoder_normal = Encoder(
            n_input_encoder_normal,
            n_latent_normal,
            n_cat_list=encoder_cat_list,
            # Set to the chosen hyperparameters of scVI
            n_layers=1,
            n_hidden=128,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )

        self.z_encoder_von_mises = EncoderHYBRIDVI(
            n_input_encoder_von_mises,
            n_latent_von_mises,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        # print("len(n_input_encoder) .... ", encoder_cat_list)
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = (n_latent_normal+n_latent_von_mises) + n_continuous_cov
        # print("len(n_input_decoder) .... ", len(n_input_encoder))
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            # Set to the chosen hyperparameters of scVI
            n_layers=1,
            n_hidden=128,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]    
        library = inference_outputs["library"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x        
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        encoder_input_von_mises = encoder_input[:,self.gene_indexes]
        encoder_input_normal = torch.Tensor(np.delete(np.array(encoder_input), self.gene_indexes, axis=1))
        qz_m, qz_v, z = self.z_encoder_normal(encoder_input_normal, batch_index, *categorical_input)
        qz_m_vM, qz_v_vM, z_vM = self.z_encoder_von_mises(encoder_input_von_mises, batch_index, *categorical_input)
        qz_m = [qz_m, qz_m_vM]  
        qz_v = [qz_v, qz_v_vM]
        z = torch.cat((z, z_vM), dim=-1)
        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            qz_m_normal = qz_m[0].unsqueeze(0).expand((n_samples, qz_m[0].size(0), qz_m[0].size(1)))
            qz_v_normal = qz_v[0].unsqueeze(0).expand((n_samples, qz_v[0].size(0), qz_v[0].size(1)))
            qz_m_von_mises = qz_m[1].unsqueeze(0).expand((n_samples, qz_m[1].size(0), qz_m[1].size(1)))
            qz_v_von_mises = qz_v[1].unsqueeze(0).expand((n_samples, qz_v[1].size(0), qz_v[1].size(1)))
            # when z is normal, untran_z == z
            untran_z_normal = Normal(qz_m_normal, qz_v_normal.sqrt()).sample()
            untran_z_von_mises = VonMisesFisher(qz_m_von_mises, qz_v_von_mises.sqrt()).sample()
            z_normal = self.z_encoder_normal.z_transformation(untran_z_normal)
            z_von_mises = self.z_encoder_von_mises.z_transformation(untran_z_von_mises)
            z = torch.cat((z_normal, z_von_mises), -1)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, decoder_input, library, batch_index, *categorical_input, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m[0])
        scale = torch.ones_like(qz_v[0])

        kl_divergence_z_normal = kl(Normal(qz_m[0], qz_v[0].sqrt()), Normal(mean, scale)).sum(dim=1)
        kl_divergence_z_von_mises = kl(VonMisesFisher(qz_m[1], qz_v[1]), HypersphericalUniform(self.n_latent_von_mises - 1)).mean()
        kl_divergence_z = kl_divergence_z_normal +  kl_divergence_z_von_mises 
        if not self.use_observed_lib_size:
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                Normal(ql_m, ql_v.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=True,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        if self.gene_likelihood == "poisson":
            l_train = px_rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
        else:
            raise ValueError(
                "{} reconstruction error not handled right now".format(
                    self.module.gene_likelihood
                )
            )
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)
        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)

            p_z = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            log_prob_sum += p_z + p_x_zl

            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            log_prob_sum -= q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )

                ql_m = inference_outputs["ql_m"]
                ql_v = inference_outputs["ql_v"]
                q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl
