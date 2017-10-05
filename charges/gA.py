import numpy as np
import h5py
import time
from common import common as co

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
currents = ["=loc:g5gx=", "=loc:g5gy=", "=loc:g5gz="]
proj_opts = ("Pk",)
dts = 12,14,16
sumr_fit_params = {"dts": [(12,14,16),]}
exci_fit_params = {"dts": [(12,14,16),(14,16),(12,14)],
                   "tin": (1,2,3,),
                   "twp": [(i,j) for i in range(1,10) for j in [15,20,25,]]}
projs = "P4","P5","P6"
flavs = "up","dn"
Z_A = 0.791
DZ_A = 0.797 - 0.791
outfile = "gA.h5"
ow_outfile = False
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

thrp = {}
for proj in projs:
    for dt in dts:
        for flav in flavs:
            chns,arr = co.get_3pt(currents, trajs, nsrc, dt, proj, flav, factor=complex(0,1))
            ### Get bare matrix elements            
            thrp[proj, dt, flav]= {'arr': arr.real}

#
# Get the P3 projector, from P4 + P5 + P6
#
for dt in dts:
    for flav in flavs:
        a4 = thrp["P4", dt, flav]["arr"]
        a5 = thrp["P5", dt, flav]["arr"]
        a6 = thrp["P6", dt, flav]["arr"]
        a3 = a4 + a5 + a6
        thrp["P3", dt, flav] = {"arr": a3}

### Append P6 to projector list
projs = projs + ("P3",)

#
# Create the isov=up-dn and isos=up+dn combinations
#
for proj in projs:
    for dt in dts:
        au = thrp[proj, dt, "up"]["arr"]
        ad = thrp[proj, dt, "dn"]["arr"]
        isv = Z_A*(au - ad)
        iss = (Z_A + DZ_A)*(au + ad)
        thrp[proj, dt, "isov"] = co.binning(isv, nbin, seed=seed, kind=binning_kind)
        thrp[proj, dt, "isov"]["arr"] = isv
        thrp[proj, dt, "isos"] = co.binning(iss, nbin, seed=seed, kind=binning_kind)
        thrp[proj, dt, "isos"]["arr"] = iss

#
# Renormalize up and dn taking into account the mixing with the
# isoscalar component
#
for proj in projs:
    for dt in dts:
        up = thrp[proj, dt, "up"]["arr"]
        dn = thrp[proj, dt, "dn"]["arr"]
        #..
        upr = Z_A*up + DZ_A*(up + dn)/2.0
        thrp[proj, dt, "up"] = co.binning(upr, nbin, seed=seed, kind=binning_kind)
        thrp[proj, dt, "up"]["arr"] = upr
        #..
        dnr = Z_A*dn + DZ_A*(up + dn)/2.0
        thrp[proj, dt, "dn"] = co.binning(dnr, nbin, seed=seed, kind=binning_kind)
        thrp[proj, dt, "dn"]["arr"] = dnr
        
### Append isov and isos to flavor list
flavs = "isos","isov","up","dn"

#
# Create the ratios
#
ratio = {}
for flav in flavs:
    for dt in dts:
        ### twop[:,0] is the forward-propagating proton
        twopave = twop['ave'][dt,0]
        twopbin = twop['bin'][:,dt,0]
        for po in proj_opts:
            if po == "P3":
                ### Summed projector
                thrpave = thrp["P3",dt,flav]["ave"].mean(axis=1)        
                thrpbin = thrp["P3",dt,flav]["bin"].mean(axis=2)
                ratibin = (thrpbin.T/twopbin).T
                #. ratiave = thrpave/twopave
                ratiave = ratibin.mean(axis=0)
                ratio[flav,dt,"P3"] = {"num": {"ave": thrpave,
                                               "bin": thrpbin,
                                               "err": errors(thrpbin)},
                                       "den": {"ave": twopave,
                                               "bin": twopbin,
                                               "err": errors(twopbin)},
                                       "ave": ratiave,
                                       "bin": ratibin,
                                       "err": errors(ratibin)}
            if po == "Pk":
                ### Unsummed projector
                pr = ["P4", "P5", "P6"]
                thrpave = np.array([thrp[p,dt,flav]["ave"][:,i] for i,p in enumerate(pr)])
                thrpbin = np.array([thrp[p,dt,flav]["bin"][:,:,i] for i,p in enumerate(pr)])
                thrpave = thrpave.mean(axis=0)
                thrpbin = thrpbin.mean(axis=0)            
                ratibin = (thrpbin.T/twopbin).T
                #. ratiave = thrpave/twopave
                ratiave = ratibin.mean(axis=0)
                ratio[flav,dt,"Pk"] = {"num": {"ave": thrpave,
                                               "bin": thrpbin,
                                               "err": errors(thrpbin)},
                                       "den": {"ave": twopave,
                                               "bin": twopbin,
                                               "err": errors(twopbin)},
                                       "ave": ratiave,
                                       "bin": ratibin,
                                       "err": errors(ratibin)}
        
#
# Fit ratios to constant
#
ratiofit = {}
for flav in flavs:
    for dt in dts:
        for po in proj_opts:
            for i in range(1,int(dt/2)):
                fr = (i,dt-i)
                x = co.fit_plat(ratio[flav,dt,po], dt, fr, errfunc=errors)
                if x is None:
                    continue
                else:
                    ratiofit[flav,dt,po,fr] = x
                    
#
# Fit ratios to an exponential form
#    
twp = {'ave': twop['ave'][:,0],
       'err': twop['err'][:,0],
       'bin': twop['bin'][:,:,0]}

excifit = {}
for flav in flavs:
    for po in proj_opts:
        for dt in exci_fit_params["dts"]:                
            for ifit in exci_fit_params["tin"]:
                for twfit in exci_fit_params["twp"]:
                    print(flav, po, dt, ifit, twfit)
                    thp = [ratio[flav, dt, po]["num"] for dt in dt]
                    x = co.fit_exci(dt, twp, thp, twfit, (ifit, ifit),
                                    errfunc=errors, ratio_fit=False)
                    if x is not None:
                        excifit[flav, po, twfit, dt, (ifit,-ifit)] = x

#
# Get the summed ratio
#
sumrati = {}
for flav in flavs:
    for po in proj_opts:
        ### Skip 1st and last tins
        sl = slice(1,-1)
        sra = np.array([ratio[flav, dt, po]['ave'][sl].sum() for dt in dts])
        srb = np.array([ratio[flav, dt, po]['bin'][:,sl].sum(axis=1) for dt in dts]).T
        sre = errors(srb)
        sumrati[flav, po] = {"ts": np.array(dts),
                             "ave": sra,
                             "err": errors(srb),
                             "bin": srb,}
                             
#
# Fit the summed ratio
#
sumratifit = {}
for flav in flavs:
    for po in proj_opts:
        for sdts in sumr_fit_params["dts"]:
            sumratifit[flav, po, sdts] = co.fit_sumr(sdts, sumrati[flav, po],
                                                     errfunc=errors)

mode = "w" if ow_outfile else "a"
with h5py.File(outfile, mode) as fp:
    s = filter(lambda x: 'set' in x, fp.keys())
    n = len(list(s))
    grp = fp.create_group("set%d" % n)        
    grp.attrs['trajs'] = ",".join(trajs)
    grp.attrs['nconf'] = len(trajs)
    grp.attrs['renorm'] = Z_A,DZ_A
    grp.attrs['resampling'] = binning_kind
    grp.attrs['n_bins'] = nbin
    if seed is not None:
        grp.attrs['seed'] = seed
    grp.attrs['date'] = time.asctime()
    for key in ratio:
        flav,ts,pr = key
        r = ratio[key]
        grp1 = grp.create_group("ratio/%s/ts%02d/%s" % (flav,ts,pr))
        grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
        grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
        grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
        grp2 = grp.create_group("ratio/%s/ts%02d/%s/num" % (flav,ts,pr))
        n = r["num"]
        grp2.create_dataset("ave", n['ave'].shape, dtype='f', data=n['ave'])
        grp2.create_dataset("err", n['err'].shape, dtype='f', data=n['err'])
        grp2.create_dataset("bin", n['bin'].shape, dtype='f', data=n['bin'])   
        grp2 = grp.create_group("ratio/%s/ts%02d/%s/den" % (flav,ts,pr))
        d = r["den"]
        grp2.create_dataset("ave", d['ave'].shape, dtype='f', data=d['ave'])
        grp2.create_dataset("err", d['err'].shape, dtype='f', data=d['err'])
        grp2.create_dataset("bin", d['bin'].shape, dtype='f', data=d['bin'])   
    for key in ratiofit:
        flav,ts,pr,tfit = key
        r = ratiofit[key]
        grp1 = grp.create_group("ratiofit/%s/ts%02d/%s/%d-%d" % (flav,ts,pr,tfit[0],tfit[1]))
        grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
        grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
        grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
        grp1.create_dataset("chi", r['chi'].shape, dtype='f', data=r['chi'])   
    for key in excifit:
        flav,pr,twfit,dts,tins = key
        for ftype in excifit[key]:
            grp1 = grp.create_group("excifit/%s/%s/%d-%d/ts%s/%d-%d/%s" %
                                    (flav,pr,twfit[0],twfit[1],",".join(["%d" % x for x in dts]),tins[0],tins[1],ftype))
            r = excifit[key][ftype]
            grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
            grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
            grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
            if ftype == '1-step':
                f = excifit[key][ftype]['fitparams']
                grp2 = grp1.create_group("fitparams")
                grp2.create_dataset("ave", f['ave'].shape, dtype='f', data=f['ave'])
                grp2.create_dataset("err", f['err'].shape, dtype='f', data=f['err'])
                grp2.create_dataset("bin", f['bin'].shape, dtype='f', data=f['bin'])   
                grp2.create_dataset("chi", f['chi'].shape, dtype='f', data=f['chi'])
                grp2.create_dataset("twop_fr", f['twop_fr'].shape, dtype='f', data=f['twop_fr'])   
            if ftype == '2-step':
                f = excifit[key][ftype]['fitparams']['twop']
                grp2 = grp1.create_group("fitparams/twop")
                grp2.create_dataset("ave", f['ave'].shape, dtype='f', data=f['ave'])
                grp2.create_dataset("err", f['err'].shape, dtype='f', data=f['err'])
                grp2.create_dataset("bin", f['bin'].shape, dtype='f', data=f['bin'])   
                grp2.create_dataset("chi", f['chi'].shape, dtype='f', data=f['chi'])
                grp2.create_dataset("twop_fr", f['twop_fr'].shape, dtype='f', data=f['twop_fr'])   
                f = excifit[key][ftype]['fitparams']['thrp']
                grp2 = grp1.create_group("fitparams/thrp")
                grp2.create_dataset("ave", f['ave'].shape, dtype='f', data=f['ave'])
                grp2.create_dataset("err", f['err'].shape, dtype='f', data=f['err'])
                grp2.create_dataset("bin", f['bin'].shape, dtype='f', data=f['bin'])   
                grp2.create_dataset("chi", f['chi'].shape, dtype='f', data=f['chi'])
    for key in sumrati:
        flav,pr = key
        r = sumrati[key]
        grp1 = grp.create_group("sumratio/%s/%s" % (flav,pr))
        grp1.create_dataset("ts", r['ts'].shape, dtype='f', data=r['ts'])
        grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
        grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
        grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
    for key in sumratifit:
        flav,pr,dts = key
        r = sumratifit[key]
        grp1 = grp.create_group("sumratiofit/%s/%s/ts%s" % (flav,pr,",".join(["%d" % x for x in dts])))
        grp1.create_dataset("ts", r['ts'].shape, dtype='f', data=r['ts'])
        grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
        grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
        grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
        grp1.create_dataset("chi", r['chi'].shape, dtype='f', data=r['chi'])   
