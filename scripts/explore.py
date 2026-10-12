import numpy as np, json
d = np.load('reports/devdata.npz', allow_pickle=False)
q, y, qmin, scale = d['q'], d['y'], d['qmin'], d['scale']
head, head_y = d['head'], d['head_y']
X = q.astype(np.float32) / np.where(scale > 0, scale, 1.0) + qmin
print('dev X', X.shape, 'head', head.shape)
print('labels', {k: int((y == k).sum()) for k in np.unique(y)})
np.save('/tmp/dev_X.npy', X)  # convenience
# how similar are the two halves?
A, B = X[:, :1024], X[:, 1024:]
cos = (A * B).sum(1)
print('cos(half1, half2) mean %.4f std %.4f min %.3f max %.3f' % (cos.mean(), cos.std(), cos.min(), cos.max()))
# within-class centroids of each half
for k in np.unique(y):
    m = y == k
    cA = A[m].mean(0); cB = B[m].mean(0)
    print(f'{k:9s} n={m.sum():5d} |cA|={np.linalg.norm(cA):.3f} |cB|={np.linalg.norm(cB):.3f} cos(cA,cB)={float(cA@cB/np.linalg.norm(cA)/np.linalg.norm(cB)):.3f}')
# PCA on halves
for name, Z in [('A', A), ('B', B), ('AB', X)]:
    Zc = Z - Z.mean(0)
    u, s, vt = np.linalg.svd(Zc[:4000], full_matrices=False)
    var = s**2 / (s**2).sum()
    print(name, 'top10 var', np.round(var[:10], 3), 'cum var@10 %.3f @50 %.3f @100 %.3f' % (var[:10].sum(), var[:50].sum(), var[:100].sum()))
# float32 vs quantised check
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict
    Xh = head; yh = head_y
    clf = LogisticRegression(max_iter=200, C=1.0)
    pred = cross_val_predict(clf, Xh, yh, cv=3, n_jobs=2)
    print('float32 head logreg macroF1 (3-fold on 1500):', round(f1_score(yh, pred, average='macro'), 4))
except Exception as e:
    print('sklearn err', e)
