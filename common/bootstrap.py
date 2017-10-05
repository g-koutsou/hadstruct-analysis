import numpy as np
rint = np.random.randint

def bootstrap(data, N_bs, seed=None):
    np.random.seed(seed)
    shape = list(data.shape)
    Nstat = shape[0]
    bs = list(map(lambda x: data[rint(0,Nstat,Nstat),:].mean(axis=0), range(N_bs)))
    return np.array(bs)
