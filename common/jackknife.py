import numpy as np

def jackknife(data, n):
    old_shape = data.shape
    data = data.reshape([data.shape[0], -1])
    N = int(data.shape[0]/n)*n
    data = data[:N,:].reshape([int(N/n),n,-1])
    data = (data.sum(axis=0).sum(axis=0) - data.sum(axis=1))/(N-n)
    data = data.reshape((data.shape[0],) + old_shape[1:])
    return data

def _superjackknife(*args):
    data = args
    shapes = [x.shape[1:] for x in data]
    if len(set(shapes)) != 1:
        msg = " Error in superjackknife: arrays with unequal secondary shapes"
        raise ValueError(msg)
    inner_shape = tuple(list(set(shapes))[0])
    nens = len(data)
    nn = [x.shape[0] for x in data]
    data = [x.reshape([nn[i],-1]) for i,x in enumerate(data)]
    nsuper = sum(nn)
    newshape = (nens,nsuper) + data[0].shape[1:]
    superdata = np.zeros(newshape)
    for ens in range(nens):
        superdata[ens] = np.array([data[ens].mean(axis=0)]*nsuper)
    i = 0
    for ens in range(nens):
        for n in range(nn[ens]):
            superdata[ens][i,:] = data[ens][n,:]
            i += 1
    superdata = superdata.transpose((1,0,2))
    superdata = superdata.reshape((nsuper,nens) + inner_shape)
    return superdata

def superjackknife(*args):
    data = args
    n = len(data)
    nn = [x.shape[0] for x in data]
    sh = [x.shape for x in data]
    data = [x.reshape([sh[i][0],-1]) for i,x in enumerate(data)]
    N = sum([x.shape[0] for x in data])
    superdata = [list([0]*n) for i in range(N)]    
    for i in range(N):
        for j in range(n):
            superdata[i][j] = data[j].mean(axis=0).reshape(sh[j][1:])
    k = 0
    for j in range(n):
        for i in range(nn[j]):
            superdata[k][j] = data[j][i,:].reshape(sh[j][1:])
            k += 1
    return superdata
