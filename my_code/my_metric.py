# AUC metric
import numpy as np

def auc_metric_(solution, prediction, task='binary.classification'):
    ''' Normarlized Area under ROC curve (AUC).
    Return Gini index = 2*AUC-1 for  binary classification problems.
    Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
    for the predictions. If solution and prediction are not vectors, the AUC
    of the columns of the matrices are computed and averaged (with no weight).
    The same for all classification problems (in fact it treats well only the
    binary and multilabel classification problems).'''
    # auc = metrics.roc_auc_score(solution, prediction, average=None)
    # There is a bug in metrics.roc_auc_score: auc([1,0,0],[1e-10,0,0]) incorrect
    label_num = 1
    if solution.ndim >1:
        label_num = solution.shape[1]
    auc = np.empty(label_num)
    for k in range(label_num):
        if solution.ndim == 1:
            r_ = tiedrank(prediction)
            s_ = solution
        else:
            r_ = tiedrank(prediction[:, k])
            s_ = solution[:, k]
        if sum(s_) == 0: print('WARNING: no positive class example in class {}'.format(k + 1))
        npos = sum(s_ == 1)
        nneg = sum(s_ < 1)
        auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
        # print('AUC[%d]=' % k + '%5.2f' % auc[k])
    return 2 * mvmean(auc) - 1
    
def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties 
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties 
        oldval = sa[0]
        newval = sa[0]
        k0 = 0
        for k in range(1, m):
            newval = sa[k]
            if newval == oldval:
                # moving average
                R[k0:k + 1] = R[k - 1] * (k - k0) / (k - k0 + 1) + R[k] / (k - k0 + 1)
            else:
                k0 = k;
                oldval = newval
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S

def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0: return R
    average = lambda x: reduce(lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]), enumerate(x))[
        1]
    R = np.array(R)
    if len(R.shape) == 1: return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))