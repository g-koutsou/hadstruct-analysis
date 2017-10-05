import numpy as np
import h5py
import time
import common as co

def errors_bs(bins):
    return bins.std(axis=0)

def errors_jk(bins):
    return bins.std(axis=0)*np.sqrt(bins.shape[0]-1)

#### INPUTS ####
#
trajlists = {
    "dt12": {"nsrc": 16, "fname": "trajs.list"},
    "dt14": {"nsrc": 16, "fname": "trajs.list"},
    "dt16": {"nsrc": 16, "fname": "trajs.list"},
}
nbin = 16
seed = 17
binning_kind = "jackknife"
currents = ["=der:g0D0:sym=", "=der:gxDx:sym=", "=der:gyDy:sym=", "=der:gzDz:sym=",]
proj_opts = "P0",
mefffitts = (8,26)
dts = 12,14,16
sumrfitdts = [(12,14,16),]
excifittypes = ["1-step","2-step"]
excifitdts = [(12,14,16),(14,16),(12,14)]
excifittin = (2,3)
excifittwopts = (2,15)
projs = "P0",
flavs = "up","dn"
Z_vD = 1.1251
outfile = "xq.h5"
ow_outfile = True
#
################

if binning_kind == "bootstrap":
    def errors(x): return errors_bs(x)
elif binning_kind == "jackknife":
    def errors(x): return errors_jk(x)    

sdts = tuple(["dt%2d" % dt for dt in dts])

def get_trajs(fn):
    with open(fn, "r") as fp:
        n = list(map(str.strip, fp.readlines()))
    return n

#
# Get trajectory numbers
#
for sdt in sdts:
    trajlists[sdt]["trajs"] = get_trajs(trajlists[sdt]["fname"])

#
# Read two-point and three-point functions
#
twop = {}
for dt in dts:
    sdt = "dt%2d" % dt
    twop[dt] = co.binning(co.get_2pt(trajlists[sdt]['trajs'], trajlists[sdt]['nsrc']).real, nbin, kind=binning_kind)

thrp = {}
for proj in projs:
    for dt in dts:
        for flav in flavs:
            trajs = trajlists["dt%2d" % dt]["trajs"]
            nsrc = trajlists["dt%2d" % dt]["nsrc"]
            chns,arr = co.get_3pt(currents, trajs, nsrc, dt, proj, flav)
            thrp[proj, dt, flav] = co.binning(Z_vD*arr.real, nbin, kind=binning_kind, seed=seed)
            thrp[proj, dt, flav]['arr'] = Z_vD*arr.real

#
# Nucleon effective mass
#
meff = {}
mfit = {}
for dt in dts:
    twp = {'ave': twop[dt]['ave'][:,0],
           'err': twop[dt]['err'][:,0],
           'bin': twop[dt]['bin'][:,:,0]}
    meff[dt] = co.baryon_meff(twp, errfunc=errors)
    mfit[dt] = co.fit_meff(meff[dt], mefffitts, errfunc=errors)

#
#
# Create the isov=up-dn and isos=up+dn combinations
#
for proj in projs:
    for dt in dts:
        au = thrp[proj, dt, "up"]["arr"]
        ad = thrp[proj, dt, "dn"]["arr"]
        isv = au-ad
        iss = au+ad
        thrp[proj, dt, "isov"] = co.binning(isv, nbin, kind=binning_kind, seed=seed)
        thrp[proj, dt, "isov"]["arr"] = isv
        thrp[proj, dt, "isos"] = co.binning(iss, nbin, kind=binning_kind, seed=seed)
        thrp[proj, dt, "isos"]["arr"] = iss

### Append isov and isos to flavor list
flavs = "isos","isov","up","dn"

#
# Create the ratios
#
ratio = {}
for flav in flavs:
    for dt in dts:
        ### twop[:,0] is the forward-propagating proton
        twopave = twop[dt]['ave'][dt,0]
        twopbin = twop[dt]['bin'][:,dt,0]        
        for po in proj_opts:
            if po == "P0":
                ### Take temporal component                
                def subtrace(thrp):
                    trace = thrp[:,:].mean(axis=1)
                    thrp = np.array([thrp[:,i] - trace for i in range(4)])
                    return -4.0/3.0*thrp[0, :]
                thrpave = subtrace(thrp["P0",dt,flav]["ave"])/mfit[dt]['ave']
                thrpbin = (np.array(
                    list(map(subtrace, thrp["P0",dt,flav]["bin"]))
                ).T/mfit[dt]['bin']).T
                # thrpave = thrp["P0",dt,flav]["ave"][:,0]/mfit[dt]['ave']
                # thrpbin = (thrp["P0",dt,flav]["bin"][:,:,0].T/mfit[dt]['bin']).T
                ratiave = thrpave/twopave
                ratibin = (thrpbin.T/twopbin).T
                ratio[flav,dt,"P0"] = {"num": {"ave": thrpave,
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
excifit = {}
for flav in flavs:
    for po in proj_opts:
        for edts in excifitdts:
            stats = [(dt,trajlists["dt%2d" % dt]) for dt in edts]
            stats = [(dt, v["nsrc"]*len(v["trajs"])) for dt,v in stats]
            dt = max(stats)[0]
            twp = {'ave': twop[dt]['ave'][:,0],
                   'err': twop[dt]['err'][:,0],
                   'bin': twop[dt]['bin'][:,:,0]}            
            for ifit in excifittin:
                thp = [ratio[flav, dt, po]["num"] for dt in edts]
                x = co.fit_exci(edts, twp, thp, excifittwopts, (ifit, ifit))
                if x is not None:
                    excifit[flav, po, edts, (ifit,-ifit)] = x

#
# Get the summed ratio
#
sumrati = {}
for flav in flavs:
    for po in proj_opts:
        ### Skip 1st and last tins
        sl = slice(2,-2)
        sra = np.array([ratio[flav, dt, po]['ave'][sl].sum() for dt in dts])
        srb = np.array([ratio[flav, dt, po]['bin'][:,sl].sum(axis=1) for dt in dts]).T
        sre = srb.std(axis=0)
        sumrati[flav, po] = {"ts": np.array(dts),
                             "ave": sra,
                             "err": sre,
                             "bin": srb,}
                             
#
# Fit the summed ratio
#
sumratifit = {}
for flav in flavs:
    for po in proj_opts:
        for sdts in sumrfitdts:
            sumratifit[flav, po, sdts] = co.fit_sumr(sdts, sumrati[flav, po])

mode = "w" if ow_outfile else "a"
with h5py.File(outfile, mode) as fp:
    s = filter(lambda x: 'set' in x, fp.keys())
    n = len(list(s))
    grp = fp.create_group("set%d" % n)        
    trajs_info = []
    nconf_info = []
    for dt in dts:
        trajs = ",".join(trajlists["dt%2d" % dt]['trajs'])
        nsrc = trajlists["dt%2d" % dt]['nsrc']
        trajs_info.append("(dt=%d|nsrc=%d|trajs=%s)" % (dt, nsrc, trajs))
        nconf_info.append("(dt=%d|nsrc=%d|nconf=%d)" % (dt, nsrc, len(trajs.split(","))))
    grp.attrs['trajs'] = ",\n".join(trajs_info)
    grp.attrs['nconf'] = ",\n".join(nconf_info)
    grp.attrs['renorm'] = Z_vD
    grp.attrs['resampling'] = binning_kind
    grp.attrs['n_bins'] = nbin
    grp.attrs['date'] = time.asctime()
    try:
        for dt in dts:
            r = meff[dt]
            grp1 = grp.create_group("nuclmeff/ts%d" % dt)
            grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
            grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
            grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
            grp1 = grp.create_group("nuclmefffit/ts%d/%d-%d" % (dt,mefffitts[0],mefffitts[1]+1))
            r = mfit[dt]
            grp1.create_dataset("ave", r['ave'].shape, dtype='f', data=r['ave'])
            grp1.create_dataset("err", r['err'].shape, dtype='f', data=r['err'])
            grp1.create_dataset("bin", r['bin'].shape, dtype='f', data=r['bin'])   
            grp1.create_dataset("chi", r['chi'].shape, dtype='f', data=r['chi'])   
    except NameError:
        pass
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
        flav,pr,dts,tins = key
        for ftype in excifit[key]:
            grp1 = grp.create_group("excifit/%s/%s/ts%s/%d,%d/%s" %
                                    (flav,pr,",".join(["%d" % x for x in dts]),tins[0],tins[1],ftype))
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
