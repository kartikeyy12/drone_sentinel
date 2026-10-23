"""SVDPCA and UncenteredSVD against scikit-learn, plus the linear algebra facts they rely on."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA, TruncatedSVD

from sentinel.svdpca import SVDPCA, UncenteredSVD


def _data(n=300, d=20, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((d, d)) * np.linspace(3.0, 0.1, d)  # distinct singular values
    return rng.standard_normal((n, d)) @ mix + offset


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flip each row (component) or column (scores) of `a` to the sign of the matching row of `b`."""
    return a * np.sign(np.sum(a * b, axis=1, keepdims=True))


@pytest.mark.parametrize("k", [1, 5, 20])
def test_matches_sklearn_pca(k):
    X = _data()
    mine, ref = SVDPCA(k).fit(X), PCA(k, svd_solver="full").fit(X)
    np.testing.assert_allclose(mine.mean_, ref.mean_, atol=1e-12)
    np.testing.assert_allclose(_align(mine.components_, ref.components_), ref.components_, atol=1e-8)
    np.testing.assert_allclose(mine.singular_values_, ref.singular_values_, rtol=1e-10)
    np.testing.assert_allclose(mine.explained_variance_, ref.explained_variance_, rtol=1e-10)
    np.testing.assert_allclose(mine.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-10)


def test_transform_matches_sklearn_up_to_sign():
    X = _data()
    mine, ref = SVDPCA(6).fit(X), PCA(6, svd_solver="full").fit(X)
    Zm, Zr = mine.transform(X), ref.transform(X)
    signs = np.sign(np.sum(Zm * Zr, axis=0))
    np.testing.assert_allclose(Zm * signs, Zr, atol=1e-8)


def test_inverse_transform_roundtrip_and_rank_k_projection():
    X = _data()
    full = SVDPCA().fit(X)
    np.testing.assert_allclose(full.inverse_transform(full.transform(X)), X, atol=1e-8)
    mine, ref = SVDPCA(4).fit(X), PCA(4, svd_solver="full").fit(X)
    np.testing.assert_allclose(mine.inverse_transform(mine.transform(X)), ref.inverse_transform(ref.transform(X)),
                               atol=1e-8)


def test_float_n_components_matches_sklearn():
    X = _data()
    for target in (0.5, 0.9, 0.99):
        mine, ref = SVDPCA(target).fit(X), PCA(target, svd_solver="full").fit(X)
        assert mine.n_components_ == ref.n_components_
        assert mine.explained_variance_ratio_.sum() > target


def test_whiten_matches_sklearn_and_has_unit_variance():
    X = _data()
    mine = SVDPCA(5, whiten=True).fit(X)
    ref = PCA(5, whiten=True, svd_solver="full").fit(X)
    Zm, Zr = mine.transform(X), ref.transform(X)
    np.testing.assert_allclose(Zm * np.sign(np.sum(Zm * Zr, axis=0)), Zr, atol=1e-8)
    np.testing.assert_allclose(Zm.var(axis=0, ddof=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(mine.inverse_transform(Zm), SVDPCA(5).fit(X).inverse_transform(SVDPCA(5).fit(X).transform(X)),
                               atol=1e-8)


def test_components_orthonormal_and_scores_uncorrelated():
    X = _data()
    m = SVDPCA(8).fit(X)
    np.testing.assert_allclose(m.components_ @ m.components_.T, np.eye(8), atol=1e-10)
    cov = np.cov(m.transform(X), rowvar=False)
    np.testing.assert_allclose(cov - np.diag(np.diag(cov)), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.diag(cov), m.explained_variance_, rtol=1e-8)
    assert np.all(np.diff(m.explained_variance_) <= 0)


def test_ratio_sums_to_one_and_full_spectrum_is_kept():
    m = SVDPCA(3).fit(_data())
    assert m.explained_variance_ratio_all_.sum() == pytest.approx(1.0)
    assert len(m.singular_values_all_) == 20 and len(m.components_) == 3


def test_eckart_young_frobenius_error_equals_discarded_singular_values():
    X = _data()
    m = SVDPCA(5).fit(X)
    err = np.linalg.norm((X - m.mean_) - m.transform(X) @ m.components_)
    assert err == pytest.approx(np.sqrt((m.singular_values_all_[5:] ** 2).sum()), rel=1e-10)
    assert m.reconstruction_mse(X) == pytest.approx(err ** 2 / X.size, rel=1e-10)


def test_wide_matrix_fewer_samples_than_features():
    X = _data(n=10, d=30)
    mine, ref = SVDPCA().fit(X), PCA(svd_solver="full").fit(X)
    assert mine.n_components_ == ref.n_components_ == 10
    np.testing.assert_allclose(mine.explained_variance_, ref.explained_variance_, atol=1e-10)


def test_sign_convention_is_deterministic():
    X = _data()
    c = SVDPCA(5).fit(X).components_
    assert np.all(c[np.arange(5), np.argmax(np.abs(c), axis=1)] > 0)
    np.testing.assert_array_equal(c, SVDPCA(5).fit(X + 0.0).components_)


def test_constant_column_and_constant_matrix_do_not_crash():
    X = _data()
    X[:, 3] = 7.0
    m = SVDPCA(5).fit(X)
    assert np.isfinite(m.components_).all() and abs(m.components_[:, 3]).max() < 1e-8
    flat = SVDPCA(2).fit(np.ones((10, 4)))
    assert np.all(flat.explained_variance_ratio_ == 0) and np.isfinite(flat.transform(np.ones((3, 4)))).all()


def test_input_validation():
    m = SVDPCA(2)
    with pytest.raises(RuntimeError):
        m.transform(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        m.fit(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        m.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        SVDPCA(50).fit(_data())
    with pytest.raises(ValueError):
        SVDPCA(1.5).fit(_data())
    m = SVDPCA(2).fit(_data())
    with pytest.raises(ValueError):
        m.transform(np.zeros((3, 7)))


# ---- uncentred SVD -----------------------------------------------------------------------------

def test_uncentered_svd_matches_sklearn_truncated_svd():
    X = _data(offset=5.0)
    mine = UncenteredSVD(5).fit(X)
    ref = TruncatedSVD(5, algorithm="arpack", random_state=0).fit(X)
    np.testing.assert_allclose(_align(mine.components_, ref.components_), ref.components_, atol=1e-6)
    np.testing.assert_allclose(mine.singular_values_, ref.singular_values_, rtol=1e-6)
    np.testing.assert_allclose(mine.explained_variance_, ref.explained_variance_, rtol=1e-6)
    np.testing.assert_allclose(mine.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-6)
    Zm, Zr = mine.transform(X), ref.transform(X)
    np.testing.assert_allclose(Zm * np.sign(np.sum(Zm * Zr, axis=0)), Zr, atol=1e-5)


def test_uncentered_first_component_follows_the_mean_when_the_offset_is_large():
    X = _data(offset=20.0)
    mean_dir = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    unc, pca = UncenteredSVD(3).fit(X), SVDPCA(3).fit(X)
    assert abs(unc.components_[0] @ mean_dir) > 0.99
    # Rayleigh quotient: the top singular direction holds at least the energy along the mean direction,
    # and that share is the bulk of the raw energy (n * ||mean||^2 / ||X||_F^2 is about 0.88 for this data)
    mean_share = float(np.sum((X @ mean_dir) ** 2) / np.sum(X ** 2))
    assert mean_share > 0.8
    assert unc.energy_ratio_[0] >= mean_share - 1e-12
    assert pca.explained_variance_ratio_[0] < 0.4  # PCA removes it and sees the real structure


def test_uncentered_equals_pca_subspace_when_data_is_centred():
    X = _data()
    X = X - X.mean(axis=0)
    unc, pca = UncenteredSVD(4).fit(X), SVDPCA(4).fit(X)
    np.testing.assert_allclose(unc.singular_values_, pca.singular_values_, rtol=1e-10)
    np.testing.assert_allclose(_align(unc.components_, pca.components_), pca.components_, atol=1e-8)


def test_uncentered_roundtrip_and_validation():
    X = _data(offset=2.0)
    m = UncenteredSVD(20).fit(X)
    np.testing.assert_allclose(m.inverse_transform(m.transform(X)), X, atol=1e-8)
    with pytest.raises(RuntimeError):
        UncenteredSVD(2).transform(X)
    with pytest.raises(ValueError):
        UncenteredSVD(0).fit(X)
