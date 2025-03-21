from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD  # type: ignore[import-untyped]
from sklearn.manifold import Isomap  # type: ignore[import-untyped]

REDUCTION_METHODS = {
    "pca": PCA(n_components=2, random_state=0),
    "truncated_svd": TruncatedSVD(n_components=2, random_state=0),
    "isomap": Isomap(n_components=2),
    "kernel_pca": KernelPCA(n_components=2, kernel="rbf", random_state=0),
    }
