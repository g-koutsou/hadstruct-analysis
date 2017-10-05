import numpy as np
import h5py
import scipy.optimize
from common import common as co
from matplotlib import pyplot as plt
lsq = scipy.optimize.leastsq

def errors_bs(bins):
    return bins.std(axis=0)

def errors_jk(bins):
    return bins.std(axis=0)*np.sqrt(bins.shape[0]-1)
        
#### INPUTS ####
#
nsrc = 16
binning_kind = "jackknife"
nbin = 16
seed = None
T = 128
#
################

if binning_kind == "bootstrap":
    def errors(x): return errors_bs(x)
elif binning_kind == "jackknife":
    def errors(x): return errors_jk(x)    

#
# Get trajectory numbers
#
with open("trajs.list", "r") as fp:
    trajs = list(map(str.strip, fp.readlines()))

#
# Read two-point and three-point functions
#
twop = co.binning(co.get_2pt(trajs, nsrc).real, nbin,
                  seed=seed, kind=binning_kind)

#
# Nucleon effective mass
#
twp = {'ave': twop['ave'][:,0],
       'err': twop['err'][:,0],
       'bin': twop['bin'][:,:,0]}
meff = co.baryon_meff(twp, errfunc=errors)
mfit = {1: {}, 2: {}}
for fi in range(1,26):
    for ff in [24,]:#range(fi+4, 26):
        if ff-fi < 3:
            continue
        mfit[1][fi,ff] = co.fit_meff(meff, [fi,ff], errfunc=errors)

for fi in range(1,15):
    for ff in [24,]:#range(fi+10, 26):
        if ff-fi < 5:
            continue
        x = co.fit_meff(meff, [fi,ff], errfunc=errors, two_state=True, T=T)
        if x is not None:
            mfit[2][fi,ff] = x

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cuts = [0, 0.5, 2.5, 3]
n = max(cuts)
for i,cut in enumerate(cuts[1:],1):
    subs = {}
    for st in 1,2:
        mm,we = np.array([(x['ave'],x['chi']) for x in mfit[st].values()]).T
        subs[st] = {k: (x['ave'],x['err'],x['chi']) for k,x in mfit[st].items() if np.abs(x['chi']-1) < cut}
        subs[st] = {k: x for k,x in subs[st].items() if np.abs(x[2]-1) >= cuts[i-1]}
    for fr,va in subs[1].items():
        ax.errorbar(fr[0], va[0], va[1], color='r', ls="", marker='s', alpha=(n-cut+.1)/n)
    for fr,va in subs[2].items():
        ax.errorbar(fr[0], va[0][0], va[1][0], color='b', ls="", marker='o', alpha=(n-cut+.1)/n)
ax.set_xticks(range(24))
ax.set_ylim(0.1,0.5)
fig.canvas.draw()
fig.show()
