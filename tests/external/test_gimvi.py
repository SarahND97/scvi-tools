import os
import pickle

import numpy as np
import pytest
import torch

import scvi
from scvi.data import synthetic_iid
from scvi.external import GIMVI


def test_saving_and_loading(save_path):
    def legacy_save(
        model,
        dir_path,
        prefix=None,
        overwrite=False,
        save_anndata=False,
        **anndata_write_kwargs,
    ):
        # get all the user attributes
        user_attributes = model._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        file_name_prefix = prefix or ""

        if save_anndata:
            dataset_names = ["seq", "spatial"]
            for i in range(len(model.adatas)):
                dataset_name = dataset_names[i]
                save_path = os.path.join(
                    dir_path, f"{file_name_prefix}adata_{dataset_name}.h5ad"
                )
                model.adatas[i].write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, f"{file_name_prefix}var_names_{dataset_name}.csv"
                )

                var_names = model.adatas[i].var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}model_params.pt")
        attr_save_path = os.path.join(dir_path, f"{file_name_prefix}attr.pkl")

        torch.save(model.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    def test_save_and_load(save_path, legacy=False):
        prefix = "GIMVI_"
        adata = synthetic_iid(run_setup_anndata=False)
        GIMVI.setup_anndata(
            adata,
            batch_key="batch",
        )

        # GIMVI
        model = GIMVI(adata, adata)
        model.train(3, train_size=0.5)
        z1 = model.get_latent_representation([adata])
        z2 = model.get_latent_representation([adata])
        np.testing.assert_array_equal(z1, z2)
        if legacy:
            legacy_save(
                model, save_path, overwrite=True, save_anndata=True, prefix=prefix
            )
        else:
            model.save(save_path, overwrite=True, save_anndata=True, prefix=prefix)
        model = GIMVI.load(save_path, prefix=prefix)
        model.get_latent_representation()
        tmp_adata = scvi.data.synthetic_iid(n_genes=200)
        with pytest.raises(ValueError):
            GIMVI.load(
                save_path, adata_seq=tmp_adata, adata_spatial=tmp_adata, prefix=prefix
            )
        model = GIMVI.load(
            save_path, adata_seq=adata, adata_spatial=adata, prefix=prefix
        )
        z2 = model.get_latent_representation([adata])
        np.testing.assert_array_equal(z1, z2)
        model = GIMVI.load(
            save_path,
            adata_seq=adata,
            adata_spatial=adata,
            use_gpu=False,
            prefix=prefix,
        )
        z2 = model.get_latent_representation([adata])
        np.testing.assert_almost_equal(z1, z2, decimal=3)
        assert model.is_trained is True

    test_save_and_load(save_path, legacy=True)
    test_save_and_load(save_path)
    # Test load prioritizes newer save paradigm and thus mismatches legacy save.
    with pytest.raises(AssertionError):
        test_save_and_load(save_path, legacy=True)


def test_gimvi():
    adata_seq = synthetic_iid()
    adata_spatial = synthetic_iid()
    model = GIMVI(adata_seq, adata_spatial, n_latent=10)
    assert hasattr(model.module, "library_log_means_0") and not hasattr(
        model.module, "library_log_means_1"
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_latent_representation()
    model.get_imputed_values()

    adata_spatial.var_names += "asdf"
    with pytest.raises(ValueError):
        model = GIMVI(adata_seq, adata_spatial)


def test_gimvi_model_library_size():
    adata_seq = synthetic_iid()
    adata_spatial = synthetic_iid()
    model = GIMVI(
        adata_seq, adata_spatial, model_library_size=[True, True], n_latent=10
    )
    assert hasattr(model.module, "library_log_means_0") and hasattr(
        model.module, "library_log_means_1"
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_latent_representation()
    model.get_imputed_values()
