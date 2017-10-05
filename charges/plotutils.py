import numpy as np
import h5py

def get_ratio(fn, setname, dt, chan='isov', num=False):
    grp = '/%s/ratio/%s/ts%02d/Pk' % (setname, chan, dt)
    with h5py.File(fn, "r") as fp:
        if not num:
            ave = np.array(fp[grp]['ave'])
            err = np.array(fp[grp]['err'])
            rbi = np.array(fp[grp]['bin'])
        else:
            ave = np.array(fp[grp]['num/ave'])
            err = np.array(fp[grp]['num/err'])
            rbi = np.array(fp[grp]['num/bin'])
    return {"ave": ave, "err": err, "bin": rbi}

def get_ratifit(fn, setname, dt, fitr, chan='isov'):
    grp = '/%s/ratiofit/%s/ts%02d/Pk/%d-%d/' % (setname, chan, dt, *fitr)
    with h5py.File(fn, "r") as fp:
        ave = np.array(fp[grp]['ave'])
        err = np.array(fp[grp]['err'])
        rbi = np.array(fp[grp]['bin'])
    return {"ave": ave, "err": err, "bin": rbi}

def get_excifit(fn, setname, twop_fr, dts, fitr, chan='isov', params=False):
    tss = "ts" + ",".join(["%02d" % x for x in sorted(dts)])
    grp = '/%s/excifit/%s/Pk/%d-%d/%s/%d-%d/1-step' % (setname, chan, *twop_fr, tss, *fitr)
    with h5py.File(fn, "r") as fp:
        if grp not in fp:
            return None
        if not params:
            ave = np.array(fp[grp]['ave'])
            err = np.array(fp[grp]['err'])
            rbi = np.array(fp[grp]['bin'])
            return {"ave": ave, "err": err, "bin": rbi}
        else:
            ave = np.array(fp[grp]['fitparams/ave'])
            err = np.array(fp[grp]['fitparams/err'])
            rbi = np.array(fp[grp]['fitparams/bin'])
            chi = np.array(fp[grp]['fitparams/chi'])
            return {"ave": ave, "err": err, "bin": rbi, "chi": chi}
