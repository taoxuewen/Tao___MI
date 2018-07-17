'''author:taoxuewen'''
import numpy as np
from scipy import linalg
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def _ajd_pham(X, eps=1e-6, max_iter=15):
    #矩阵联合近似对角化，来源于一篇对角化的文献，属于数学的东西，非常深奥，不要深究。
    #我是从mne库里抄的，后来发现mne那帮人也是在pyRiemann里抄的，pyRiemann是一帮大神写的
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_channels)
        A set of covariance matrices to diagonalize.
    eps : float, defaults to 1e-6
        The tolerance for stoping criterion.
    max_iter : int, defaults to 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_epochs, n_channels, n_channels)
        The set of quasi diagonal matrices.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
           definite Hermitian matrices." SIAM Journal on Matrix Analysis and
           Applications 22, no. 4 (2001): 1136-1152.

    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(max_iter):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D

def _regularized_covariance(data, reg=None):
    #这个是正则化协方差函数（主要是为了预防空间滤波后出现的奇异矩阵情况），来源于mne库，改写的sklearn的原函数，直接用了得了,life is short.
    """Compute a regularized covariance from data using sklearn.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Data for covariance estimation.
    reg : float | str | None (default None)
        If not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        The covariance matrix.
    """
    if reg is None:
        # compute empirical covariance
        cov = np.cov(data)
    else:
        no_sklearn_err = ('the scikit-learn package is missing and '
                          'required for covariance regularization.')
        # use sklearn covariance estimators
        if isinstance(reg, float):
            if (reg < 0) or (reg > 1):
                raise ValueError('0 <= shrinkage <= 1 for '
                                 'covariance regularization.')
            try:
                import sklearn
                from sklearn.covariance import ShrunkCovariance
            except ImportError:
                raise Exception(no_sklearn_err)
        elif isinstance(reg, str):
            if reg == 'ledoit_wolf':
                try:
                    from sklearn.covariance import LedoitWolf
                except ImportError:
                    raise Exception(no_sklearn_err)
                # init sklearn.covariance.LedoitWolf estimator
                skl_cov = LedoitWolf(store_precision=False,
                                     assume_centered=True)
            elif reg == 'oas':
                try:
                    from sklearn.covariance import OAS
                except ImportError:
                    raise Exception(no_sklearn_err)
                # init sklearn.covariance.OAS estimator
                skl_cov = OAS(store_precision=False,
                              assume_centered=True)
            else:
                raise ValueError("regularization parameter should be "
                                 "'ledoit_wolf' or 'oas'")
        else:
            raise ValueError("regularization parameter should be "
                             "of type str or int (got %s)." % type(reg))

        # compute regularized covariance using sklearn
        cov = skl_cov.fit(data.T).covariance_

    return cov




class _ITFECSP_(ClassifierMixin,BaseEstimator):
    def __init__(self, n_components=4, reg=None):
        self.log=None
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        #reg只有三种可选,None,oas,ledit_wolf
        if (
                (reg is not None) and
                (reg not in ['oas', 'ledoit_wolf']) and
                ((not isinstance(reg, (float, int))) or
                 (not ((reg <= 1.) and (reg >= 0.))))
        ):
            raise ValueError('reg must be None, "oas", "ledoit_wolf" or a '
                             'float in between 0. and 1.')
        self.reg = reg

    def fit(self, X, y):
        n_channels = X.shape[1]
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs = np.zeros((n_classes, n_channels, n_channels))
        sample_weights = list()
        for class_idx, this_class in enumerate(self._classes):
            class_ = X[y == this_class]
            cov = np.zeros((n_channels, n_channels))
            for this_X in class_:
                cov += _regularized_covariance(this_X, reg=self.reg)
            cov /= len(class_)
            weight = len(class_)

            covs[class_idx] = cov

            # covs[class_idx] /= np.trace(cov)# norm trace
            sample_weights.append(weight)

        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
            # sort eigenvectors
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = _ajd_pham(covs)

            # Here we apply an euclidean mean. See pyRiemann for other metrics
            mean_cov = np.average(covs, axis=0, weights=sample_weights)
            eigen_vectors = eigen_vectors.T

            # normalize
            for ii in range(eigen_vectors.shape[1]):
                tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov),
                             eigen_vectors[:, ii])
                eigen_vectors[:, ii] /= np.sqrt(tmp)

            # class probability
            class_probas = [np.mean(y == _class) for _class in self._classes]

            # mutual information
            mutual_info = []
            for jj in range(eigen_vectors.shape[1]):
                aa, bb = 0, 0
                for (cov, prob) in zip(covs, class_probas):
                    tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov),
                                 eigen_vectors[:, jj])
                    aa += prob * np.log(np.sqrt(tmp))
                    bb += prob * (tmp ** 2 - 1)
                mi = - (aa + (3.0 / 16) * (bb ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]

        # sort eigenvectors
        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        self.patterns_ = linalg.pinv(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=2)
        log = True if self.log is None else self.log
        if log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X

class ITFECSP(ClassifierMixin,BaseEstimator):
    def __init__(self, n_components=4, reg=None,MLclf=LogisticRegression()):
        self.n_components=n_components
        self.reg=reg
        self.MLclf=MLclf
        self.itfe=_ITFECSP_(n_components=self.n_components,reg=self.reg)
        self.clf=make_pipeline(self.itfe,self.MLclf)

    def fit(self,X,y):
        return self.clf.fit(X,y)

    def transform(self,X):
        return self.clf.transform(X)

    def predict(self,X):
        return self.clf.predict(X)

if __name__=='__main__':
    a=ITFECSP(n_components=10)


