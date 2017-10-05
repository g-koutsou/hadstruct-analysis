import time
from common import bootstrap as bs
from common import jackknife as jk
import numpy as np
import h5py
import scipy.optimize
lsq = scipy.optimize.leastsq
np.seterr(all='raise')

def get_3pt(currents, trajs, nsrc, dt, proj, flav, factor=1.0, fname="../h5-data/thrp-msq0000.h5"):
    arr = list()
    with h5py.File(fname, "r") as fp:
        for cur in currents:
            grp = "/nsrc%02d/%s/%s/dt%02d/%s" % (nsrc, cur, proj, dt, flav)
            tr = list(map(bytes.decode, np.array(fp[grp]["trajs"])))
            # check if all trajs are in tr
            if set(trajs).issubset(tr):                
                idx = [tr.index(x) for x in trajs]
                arr.append(np.array(fp[grp]["arr"])[idx].squeeze())
            else:
                if proj in ("P4","P5","P6"):
                    grp0 = "/nsrc%02d/%s/%s/dt%02d/%s" % (nsrc, cur, proj, dt, flav)
                    tr0 = list(map(bytes.decode, np.array(fp[grp0]["trajs"])))
                    grp1 = "/nsrc%02d/%s/%s/dt%02d/%s" % (nsrc, cur, "P3", dt, flav)
                    tr1 = list(map(bytes.decode, np.array(fp[grp1]["trajs"])))
                    if set(trajs).issubset(tr0+tr1):
                        tr0 = [x for x in trajs if x in tr0]
                        tr1 = [x for x in trajs if x in tr1 and x not in tr0]
                        tr = list(map(bytes.decode, np.array(fp[grp0]["trajs"])))
                        idx = [tr.index(x) for x in tr0]
                        arr0 = np.array(fp[grp0]["arr"][idx])
                        tr = list(map(bytes.decode, np.array(fp[grp1]["trajs"])))
                        idx = [tr.index(x) for x in tr1]
                        arr1 = np.array(fp[grp1]["arr"][idx])
                        ps = ["P4","P5","P6"]
                        ps.remove(proj)
                        for p in ps:
                            grp1 = "/nsrc%02d/%s/%s/dt%02d/%s" % (nsrc, cur, p, dt, flav)
                            tr = list(map(bytes.decode, np.array(fp[grp1]["trajs"])))
                            idx = [tr.index(x) for x in tr1]
                            arr1 -= np.array(fp[grp1]["arr"][idx])
                        arr2 = np.concatenate((arr0, arr1), axis=0)
                        idx = [(tr0+tr1).index(x) for x in trajs]
                        arr.append(arr2[idx].squeeze())
    return currents,np.array(arr).transpose([1,2,0])*factor

def get_2pt(trajs, nsrc):
    arr = list()
    with h5py.File("../h5-data/twop.h5", "r") as fp:
        for intrp in ["ppm","pmm"]:
            for fb in ["fwd", "bwd"]:
                grp = "/nsrc%02d/msq0000/%s/%s" % (nsrc, intrp, fb)
                tr = list(map(bytes.decode, np.array(fp[grp]["trajs"])))
                idx = [tr.index(x) for x in trajs]
                arr.append(np.array(fp[grp]["arr"])[idx].squeeze())
    return np.array(arr).transpose(1,2,0)

def binning(arr, n, axis=0, seed=1, kind="bootstrap"):
    if kind == "bootstrap":
        arr = np.rollaxis(arr, axis=axis)
        ave = arr.mean(axis=0)
        abi = bs.bootstrap(arr, n, seed=seed)
        err = abi.std(axis=0)
    elif kind == "jackknife":
        arr = np.rollaxis(arr, axis=axis)
        ave = arr.mean(axis=0)
        Ntr = arr.shape[0]
        abi = jk.jackknife(arr, n)
        err = abi.std(axis=0)*np.sqrt((Ntr-n)/n)
    else:
        raise ValueError('parameter kind should be "bootstrap" or "jackknife"')
    
    return {"ave": ave, "bin": abi, "err": err}

def fit_plat(ratio, dt, fitrange, errfunc=None):
    """Fits the ratio to a constant. Takes insertions in
    [fitrange[0],fitrange[1]]. dt is the sink-source separation. If
    the fit range is such that less than two points are included, this
    function returns None.

    """
    if errfunc is None:
        def errfunc(x): return x.std(axis=0)
    i,f = fitrange
    s = slice(i,f+1)
    if len(list(range(dt+1))[s]) < 2:
        return None
    def fit(c,e):
        k = 1 # number of parameters
        num = (c/e**2).sum()
        den = (1/e**2).sum()
        chi = ((num/den)-c)/e
        dof = len(c)-k
        AIC = - k - (chi**2).sum()/2
        AICc = - k*(k+1)/(len(c)-k-1) + AIC
        return num/den,(chi**2).sum()/dof,AIC,AICc
    ave = ratio["ave"]
    rbi = ratio["bin"]
    err = ratio["err"]
    fav,chisqdof,AIC,AICc = fit(ave[s], err[s])
    fbi = np.array(list(map(lambda x: fit(x[s], err[s]), rbi)))[:,0]
    fer = errfunc(fbi)
    return {"ave": fbi.mean(axis=0), "err": fer, "bin": fbi, "chi": chisqdof, "AIC": AIC, "AICc": AICc}

def twopfit(ts, y, e, guess=[0.5,0.5,1,1]):
    """Fits two-point function up to the 1st excited state"""
    def func(params, x):
        m,dm,a0,a1 = params
        y = a0*np.exp(-m*x)*(1 + a1*np.exp(-dm*x))
        return y
    def chi(params, x, y, e):
        return (func(params, x) - y)/e
    k = len(guess)
    p,info = lsq(chi, guess, args=(ts, y, e), maxfev=100000)
    chisq = (chi(p, ts, y, e)**2).sum()
    chisqdof = chisq/(len(ts)-len(p))
    AIC = - k - chisq/2
    AICc = - k*(k+1)/(len(y)-k-1) + AIC
    return p,chisqdof,AIC,AICc

def thrpfit(ti, ts, y, e, m, dm, guess=[1,0.1,0.1]):
    """Fits three-point function keeping 1st excited state, given ground
    state mass m and mass-gap dm
    
    """
    def func(params, ti, ts, m, dm):
        a0,a1,a2 = params
        y = a0*np.exp(-m*ts)*(1 +
                              a1*np.exp(-dm*(ts-ti)) +
                              a1*np.exp(-dm*ti) +
                              a2*np.exp(-dm*ts))
        return y
    def chi(params, ti, ts, y, e, m, dm):
        return (func(params, ti, ts, m, dm) - y)/e
    k = len(guess)
    p,info = lsq(chi, guess, args=(ti, ts, y, e, m, dm), maxfev=100000)
    chisq = (chi(p, ti, ts, y, e, m, dm)**2).sum()
    chisqdof = chisq/(len(ts)-len(p))
    AIC = - k - chisq/2
    AICc = - k*(k+1)/(len(y)-k-1) + AIC
    return p,chisqdof,AIC,AICc

def ratiofit(ti, ts, y, e, guess=[1,0.5,1,1,1]):
    """Fits three- to two-point function ratio. Ground state mass and
    two-point function overlap are eliminated. Find matrix element a0,
    mass-gap dm, overlaps a1, a2, b1. ts, ti, y and e should be of
    same length.

    """
    def func(params, ti, ts):
        a0,dm,a1,a2,b1 = params
        y = a0
        y = y + a1*(np.exp(-dm*(ts-ti)) + np.exp(-dm*ti))
        y = y + a2*np.exp(-dm*ts)
        y = y/(1 + b1*np.exp(-dm*ts))
        return y
    def chi(params, ti, ts, y, e):
        return (func(params, ti, ts) - y)/e
    k = len(guess)
    p,info = lsq(chi, guess, args=(ti, ts, y, e), maxfev=100000)
    chisq = (chi(p, ti, ts, y, e)**2).sum()
    chisqdof = chisq/(len(ts)-len(p))
    AIC = - k - chisq/2
    AICc = - k*(k+1)/(len(y)-k-1) + AIC
    return p,chisqdof,AIC,AICc,info

def thrptwopfit(ti, dt, ts, y, e, guess=None):
    """Simultaneous fit of three- and two-point functions keeping 1st
    excited state. dt is the three-point function sink-source
    separations, same length as ti. ts is the two-point function
    independent coordinate.

    """
    def twp(params, x):
        m,dm,a0,a1 = params
        y = a0*np.exp(-m*x)*(1 + a1*np.exp(-dm*x))
        return y
    def thp(params, ti, ts):
        m,dm,b0,b1,b2 = params
        y = b0*np.exp(-m*ts)*(1 +
                              b1*np.exp(-dm*(ts-ti)) +
                              b1*np.exp(-dm*ti) +
                              b2*np.exp(-dm*ts))
        return y
    def chi(params, ti, dt, ts, y, e):
        m,dm,a0,a1,b0,b1,b2 = params
        f0 = thp([m,dm,b0,b1,b2], ti, dt)
        f1 = twp([m,dm,a0,a1], ts)
        c0 = (f0-y[0])/e[0]
        c1 = (f1-y[1])/e[1]
        return np.concatenate([c0,c1])
    if guess is None:
        guess = [0.5, 0.5, y[0][1], 1.0, y[1][1], 1.0, 1.0]
    k = len(guess)
    p,info = lsq(chi, guess, args=(ti, dt, ts, y, e), maxfev=100000)
    chisq = (chi(p, ti, dt, ts, y, e)**2).sum()
    chisqdof = chisq/(len(ts)+len(ti)-len(p))
    AIC = - k - chisq/2
    AICc = - k*(k+1)/(len(np.concatenate(y))-k-1) + AIC
    return p,chisqdof,AIC,AICc

def fit_exci(dts, twop, thrp, fit_range_twop, fit_range_thrp,
             one_step_fit=True, two_step_fit=True, ratio_fit=True, errfunc=None):
    if len(dts) != len(thrp):
        raise AssertionError("len(dts) != len(thrp)")
    if errfunc is None:
        def errfunc(x): return x.std(axis=0)
    ### If the choice of fit_range_thrp and dts are such that for at
    ### least one dt there are no points to fit, then return None
    fi,ff = fit_range_thrp
    ti = [np.arange(fi,dt+1-ff) for dt in dts]
    if [] in [x.tolist() for x in ti]:
        return None    
    #
    # Set-up the two-point function
    #
    ts = np.arange(fit_range_twop[0],fit_range_twop[1]+1)
    twave = twop["ave"][ts]
    twbin = twop["bin"][:,ts]
    twerr = twop["err"][ts]
    #
    # Fit the two-point function if we're to do a 2-step fit
    #
    if two_step_fit:
        try:
            pave,chitwop,AICtw,AICctw = twopfit(ts, twave, twerr)
            pbin = np.array(list(map(lambda x: twopfit(ts, x, twerr, guess=pave)[0], twbin)))
        except FloatingPointError:
            return None
    #
    # Set-up three-point fit
    #
    thave,thbin,therr = zip(*[(x["ave"],x["bin"],x["err"]) for x in thrp])
    thave = np.concatenate([thave[i][x] for i,x in enumerate(ti)])
    thbin = np.concatenate([thbin[i][:,x] for i,x in enumerate(ti)], axis=1)
    therr = np.concatenate([therr[i][x] for i,x in enumerate(ti)])
    dt = np.concatenate([[dts[i]]*len(x) for i,x in enumerate(ti)])
    ti = np.concatenate(ti)
    ret = {}
    if one_step_fit:
        #
        # *** One - step fit ***
        # Simultaneous fit to two- and three-point function
        #
        try:
            qave,chithrp,AIC,AICc = thrptwopfit(ti, dt, ts, [thave,twave], [therr,twerr])
            qbin = np.array(list(map(lambda x,y: thrptwopfit(ti, dt, ts, [x,y], [therr,twerr], guess=qave)[0],
                                     thbin,twbin)))
        except FloatingPointError:
            return None
        ret['1-step'] = {"fitparams": {"twop_fr": np.array(fit_range_twop),
                                       "ave": qave,
                                       "bin": qbin,
                                       "err": errfunc(qbin),
                                       "chi": chithrp,
                                       "AIC": AIC,
                                       "AICc": AICc},
                         "ave": qave[4]/qave[2],
                         "bin": (qbin[:,4]/qbin[:,2]),
                         "err": errfunc(qbin[:,4]/qbin[:,2]),}
    if two_step_fit:        
        #
        # *** Two - step fit ***
        # Use masses from two-point function fit as input to three-point function fit
        #
        mave,dmave = pave[:2]
        mbin,dmbin = pbin.T[:2,:]
        try:
            qave,chithrp,AIC,AICc = thrpfit(ti, dt, thave, therr, mave, dmave)
            qbin = np.array(list(map(lambda x,m,dm: thrpfit(ti, dt, x, therr, m, dm, guess=qave)[0],
                                     thbin, mbin, dmbin)))
        except FloatingPointError:
            return None
        ret['2-step'] = {"fitparams": {"twop": {"twop_fr": np.array(fit_range_twop),
                                                "ave": pave,
                                                "bin": pbin,
                                                "err": errfunc(pbin),
                                                "chi": chitwop,
                                                "AIC": AICtw,
                                                "AICc": AICctw},
                                       "thrp": {"ave": qave,
                                                "bin": qbin,
                                                "err": errfunc(qbin),
                                                "chi": chithrp,
                                                "AIC": AIC,
                                                "AICc": AIC}},
                         "ave": qave[0]/pave[2],
                         "bin": (qbin[:,0]/pbin[:,2]),
                         "err": errfunc(qbin[:,0]/pbin[:,2]),}
    if ratio_fit:        
        #
        # *** Ratio fit ***
        # Fit to ratio three- to two-point function
        #
        ti = [np.arange(fi,dt+1-ff) for dt in dts]
        twave = np.array([twop["ave"][x] for x in dts]) 
        twbin = np.array([twop["bin"][:,x] for x in dts])
        twerr = errfunc(twbin)
        raave,rabin,raerr = zip(*[(x["ave"],x["bin"],x["err"]) for x in thrp])
        raave = np.concatenate([raave[i][x]/twave[i] for i,x in enumerate(ti)])
        rabin = np.concatenate([(rabin[i][:,x].T/twbin[i,:]).T for i,x in enumerate(ti)], axis=1)
        raerr = errfunc(rabin)
        dt = np.concatenate([[dts[i]]*len(x) for i,x in enumerate(ti)])
        ti = np.concatenate(ti)
        qave,chi,AIC,AICc,info = ratiofit(ti, dt, raave, raerr)
        if info not in [1,2,3,4]:
            ret['ratio'] = None
        else:
            qbin = np.array(list(map(lambda x: ratiofit(ti, dt, x, raerr, guess=qave)[0], rabin)))
            ret['ratio'] = {"fitparams": {"ave": qave,
                                          "bin": qbin,
                                          "err": errfunc(qbin),
                                          "chi": chi,
                                          "AIC": AIC,
                                          "AICc": AICc},
                            "ave": qave[0],
                            "bin": qbin[:,0],
                            "err": errfunc(qbin[:,0])}
    return ret

def fit_sumr(tsfit, sumr, errfunc=None):
    if errfunc is None:
        def errfunc(x): return x.std(axis=0)
    ts = sumr['ts']
    it = [i for i,x in enumerate(ts) if x in tsfit] 
    ave = sumr['ave'][it]
    err = sumr['err'][it]
    sbi = sumr['bin'][:,it]
    ts = ts[it]
    def func(params, x):
        a,b = params
        return a*x + b
    def chi(params, x, y, e):
        return (func(params, x) - y)/e
    def fit(x, y, e, guess=[1,1]):
        k = len(guess)
        p,info = lsq(chi, guess, args=(x,y,e))
        chisq = (chi(p, x, y, e)**2).sum()
        chisqdof = chisq/(len(x)-2)
        AIC = - k - chisq/2
        if len(y)-k-1 == 0:
            AICc = np.array(np.inf)
        else:
            AICc = AIC - k*(k+1)/(len(y)-k-1)            
        return p,chisqdof,AIC,AICc
    pave,chisqdof,AIC,AICc = fit(ts, ave, err)
    pbin = np.array(list(map(lambda x: fit(ts, x, err)[0], sbi)))
    return {"ts": ts,
            "ave": pave,
            "bin": pbin,
            "err": errfunc(pbin),
            "chi": chisqdof,
            "AIC": AIC,
            "AICc": AICc}

def baryon_meff(twp, errfunc=None):
    if errfunc is None:
        def errfunc(x): return x.std(axis=0)
    Lt = twp["ave"].shape[0]
    ave = twp["ave"]
    mbi = twp["bin"]
    x0 = np.arange(Lt)
    xp = (x0 + 1) % Lt
    # xm = (Lt + x0 - 1) % Lt
    ma = np.emath.log(ave[x0]/ave[xp])
    mb = np.emath.log(mbi[:,x0]/mbi[:,xp])
    ma[ma.imag != 0] = np.nan
    idx = mb.imag != 0
    idx = idx.any(axis=0)
    mb[:,idx] = np.nan
    ma = ma.real
    mb = mb.real
    me = errfunc(mb)
    return {"ave": mb.mean(axis=0), "err": me, "bin": mb}

def fit_meff_1st(meff, fitrange, errfunc=None):
    if errfunc is None:
        def errfunc(x): return x.std(axis=0)
    sl = slice(fitrange[0],fitrange[1]+1)
    mave = meff['ave'][sl]
    mbin = meff['bin'][:,sl]
    merr = meff['err'][sl]
    k = 1
    def fit(y,e):
        num = (y/e**2).sum()
        den = (1/e**2).sum()
        chisq = ((((num/den)-y)/e)**2).sum()
        chisqdof = chisq/(len(y)-1)
        AIC = - k - chisq/2
        AICc = AIC - k*(k+1)/(len(y)-k-1)
        return num/den,chisqdof,AIC,AICc
    ma,chi,AIC,AICc = fit(mave, merr)
    mb = np.array(list(map(lambda x: fit(x, merr)[0], mbin)))
    me = errfunc(mb)
    return {"ave": mb.mean(axis=0), "err": me, "bin": mb, "chi": chi, "AIC": AIC, "AICc": AICc}

def fit_meff_2st(meff, fitrange, errfunc=None, T=None):
    if errfunc is None:
        def errfunc(x): return x.std(axis=0)
    sl = slice(fitrange[0],fitrange[1]+1)
    ts = np.arange(T)[sl]
    mave = meff['ave'][sl]
    mbin = meff['bin'][:,sl]
    merr = meff['err'][sl]
    k = 2
    def fit(x, y, e):
        def f(m, dm, c, x):
            return m + np.log(1+c*np.exp(-dm*x)) - np.log(1+c*np.exp(-dm*(x+1)))
        def chi(p, x, y, e):
            m,dm,c = p
            return (y - f(m, dm, c, x))/e
        guess = [0.5, 0.2, 10.]
        p,_ = lsq(chi, guess, args=(x, y, e))
        chisq = (chi(p, x, y, e)**2).sum()
        chisqdof = chisq/(len(y)-k)
        AIC = - k - chisq/2
        AICc = AIC - k*(k+1)/(len(y)-k-1)
        return p,chisqdof,AIC,AICc
    try:
        pa,chi,AIC,AICc = fit(ts, mave, merr)
        pb = np.array(list(map(lambda x: fit(ts, x, merr)[0], mbin)))
    except FloatingPointError:
        return None
    pe = errfunc(pb)
    return {"ave": pb.mean(axis=0), "err": pe, "bin": pb, "chi": chi, "AIC": AIC, "AICc": AICc}

def fit_meff(meff, fitrange, errfunc=None, two_state=False, T=None):
    if not two_state:
        return fit_meff_1st(meff, fitrange, errfunc)
    else:
        return fit_meff_2st(meff, fitrange, errfunc, T=T)
