'''
Run this to make a model line and do some BAO fitting
'''

import BAOfit as bf
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200.0


Nmock = 1000

import argparse

import pycorr

parser = argparse.ArgumentParser()
parser.add_argument("--tracer", help="tracer type",default='Roman')
parser.add_argument("--zmin", help="minimum redshift",default=1.35,type=float)
parser.add_argument("--zmax", help="maximum redshift",default=1.65,type=float)
parser.add_argument("--rmin", help="minimum separation",default=50,type=float)
parser.add_argument("--rmax", help="maximum separation",default=150,type=float)

parser.add_argument("--bs", help="bin size in Mpc/h, some integer multiple of 1",default=4,type=int)
parser.add_argument("--cfac", help="any factor to apply to the cov matrix",default=1,type=float)
parser.add_argument("--dataver", help="data version",default='test')
parser.add_argument("--njack", help="number of jack knife used",default='50')
parser.add_argument("--weight", help="weight type used for xi",default='default_FKP')
parser.add_argument("--reg", help="regions used for xi",default='NScomb_')
parser.add_argument("--dperp", help="transverse damping; default is about right for z~1",default=1.5,type=float) # originally 2.5
parser.add_argument("--drad", help="radial damping; default is about right for z~1",default=3.0,type=float) # originally 5.0
parser.add_argument("--sfog", help="streaming velocity term; default standardish value",default=1.0,type=float) # originally 3.0
parser.add_argument("--beta", help="fiducial beta in template; shouldn't matter for pre-rec",default=0.4,type=float)
parser.add_argument("--gentemp", help="whether or not to generate BAO templates",default=True,type=bool)
parser.add_argument("--cov_type",help="how cov matrix was generated; choices are 'theory','EZ',or 'LN'",default='JK')
parser.add_argument("--input_dir",help="where to find paircounts if not from the DA directory",default=None)
parser.add_argument("--nran",help="string for file name denoting the number of random files used",default='_nran10')
parser.add_argument("--split",help="string for file name denoting where randoms were split for RR counts",default='_split20')

#parser.add_argument("--gencov", help="whether or not to generate cov matrix",default=True,type=bool)
#parser.add_argument("--acov", help="whether or not to to analytic cov",default=False,type=bool)
parser.add_argument("--rectype", help="type of reconstruction",default=None)
args = parser.parse_args()

rmin = args.rmin
rmax = args.rmax
maxb = 80.
binc = 0

zmin = args.zmin
zmax = args.zmax
bs = args.bs

if args.cov_type=='theory':
    #args.gencov= False
    fn = '/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/da02/smu/cov/ximonopole_'+args.tracer+'_NScomb_'+str(args.zmin)+'_'+str(args.zmax)+'_'+args.weight+'_lin'+str(args.bs)+'_cov_RascalC.txt'
    try:
        
        covm = np.loadtxt(fn)
    except:
        sys.exit('failed to load '+fn)

if args.gentemp:
    #make BAO template given parameters above, using DESI fiducial cosmology and cosmoprimo P(k) tools
    #mun is 0 for pre rec
    #sigs is only relevant if mun != 0 and should then be the smoothing scale for reconstructions
    #beta is b/f, so should be changed depending on tracer
    #sp is the spacing in Mpc/h of the templates that get written out, most of the rest of the code assumes 1
    #BAO and nowiggle templates get written out for xi0,xi2,xi4 (2D code reconstructions xi(s,mu) from xi0,xi2,xi4)
    if args.rectype == None:
        bf.mkxifile_3dewig(sp=1.,v='n',mun=0,beta=args.beta,sfog=args.sfog,sigt=args.dperp,sigr=args.drad,sigs=10.)
        munw = '0'
        
    elif 'iso' in args.rectype:
        bf.mkxifile_3dewig(sp=1.,v='n',mun=1,beta=args.beta,sfog=args.sfog,sigt=args.dperp,sigr=args.drad,sigs=10.)
        munw= '1'
wm = str(args.beta)+str(args.sfog)+str(args.dperp)+str(args.drad)
mod = np.loadtxt('BAOtemplates/xi0DESI'+wm+'10.0'+munw+'.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smDESI'+wm+'10.0'+munw+'.dat').transpose()[1]



def sigreg_c12(al,chill,fac=1.,md='f'):
    #report the confidence region +/-1 for chi2
    #copied from ancient code
    chim = 1000
    
    
    chil = []
    for i in range(0,len(chill)):
        chil.append((chill[i],al[i]))
        if chill[i] < chim:
            chim = chill[i]
            am = al[i]
            im = i
    #chim = min(chil)   
    a1u = 2.
    a1d = 0
    a2u = 2.
    a2d = 0
    oa = 0
    ocd = 0
    s0 = 0
    s1 = 0
    for i in range(im+1,len(chil)):
        chid = chil[i][0] - chim
        if chid > 1. and s0 == 0:
            a1u = (chil[i][1]/abs(chid-1.)+oa/abs(ocd-1.))/(1./abs(chid-1.)+1./abs(ocd-1.))
            s0 = 1
        if chid > 4. and s1 == 0:
            a2u = (chil[i][1]/abs(chid-4.)+oa/abs(ocd-4.))/(1./abs(chid-4.)+1./abs(ocd-4.))
            s1 = 1
        ocd = chid  
        oa = chil[i][1]
    oa = 0
    ocd = 0
    s0 = 0
    s1 = 0
    for i in range(1,im):
        chid = chil[im-i][0] - chim
        if chid > 1. and s0 == 0:
            a1d = (chil[im-i][1]/abs(chid-1.)+oa/abs(ocd-1.))/(1./abs(chid-1.)+1./abs(ocd-1.))
            s0 = 1
        if chid > 4. and s1 == 0:
            a2d = (chil[im-i][1]/abs(chid-4.)+oa/abs(ocd-4.))/(1./abs(chid-4.)+1./abs(ocd-4.))
            s1 = 1
        ocd = chid  
        oa = chil[im-i][1]
    if a1u < a1d:
        a1u = 2.
        a1d = 0
    if a2u < a2d:
        a2u = 2.
        a2d = 0
            
    return am,a1d,a1u,a2d,a2u,chim  

ells = 0

def get_xi0cov(md='EZ'):
        
    #dirm = '/global/project/projectdirs/desi/users/dvalcin/Mocks/2PCF/'
    if md == 'EZ':
        dirm = '/users/PHS0336/sbet/'+args.tracer+'/Xi/'
        #fnm = 'xi_lognormal_lrg_sub_'
        fnm = 'xi_ez_'+args.tracer+'_cutsky_seed' #550_z0.6_0.8.npy
        if args.rectype is not None:
            sys.exit('no recon for EZ mocks yet')
    if md == 'LN':
        dirm = '/global/cfs/cdirs/desi/users/dvalcin/Mocks'
    xinpy = dirm+fnm+'1'+'_z'+str(args.zmin)+'_'+str(args.zmax)+'.npy'
    result = pycorr.TwoPointCorrelationFunction.load(xinpy)
    rebinned = result[:(result.shape[0]//bs)*bs:bs]
    xin0 = rebinned(ells=ells)

    nbin = len(xin0)
    print(nbin)
    xiave = np.zeros((nbin))
    cov = np.zeros((nbin,nbin))

    Ntot = 0
    fac = 1.
    for i in range(1,Nmock):
        nr = str(i)
        #xii = np.loadtxt(dirm+fnm+nr+'.txt').transpose()
        xinpy = dirm+fnm+nr+'_z'+str(args.zmin)+'_'+str(args.zmax)+'.npy'
        result = pycorr.TwoPointCorrelationFunction.load(xinpy)
        rebinned = result[:(result.shape[0]//bs)*bs:bs]
        xic = rebinned(ells=ells)

        xiave += xic
        Ntot += 1.
    print( Ntot)        
    xiave = xiave/float(Ntot)
    for i in range(1,Nmock):
        nr = str(i)
        xinpy = dirm+fnm+nr+'_z'+str(args.zmin)+'_'+str(args.zmax)+'.npy'
        result = pycorr.TwoPointCorrelationFunction.load(xinpy)
        rebinned = result[:(result.shape[0]//bs)*bs:bs]
        xic = rebinned(ells=ells)

        #xii = np.loadtxt(dirm+fnm+nr+'.txt').transpose()
        #xic = xii[1]
        for j in range(0,nbin):
            xij = xic[j]#-angfac*xiit[j]
            for k in range(0,nbin):
                xik = xic[k]#-angfac*xiit[k]
                cov[j][k] += (xij-xiave[j])*(xik-xiave[k])

    cov = cov/float(Ntot)                   
        
    return cov


if args.input_dir == None:
    datadir =  '/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/da02/'
else:
    datadir = args.input_dir
#data = datadir+'xi024LRGDA02_'+str(zmin)+str(zmax)+'2_default_FKPlin'+str(bs)+'.dat'
zw = ''
if zmin == 0.8 and zmax == 2.1:
    zw = 'lowz'
if args.rectype == None:
    #data = datadir +'/smu/xipoles_LRG_'+args.reg+str(zmin)+'_'+str(zmax)+'_'+args.weight+'_lin'+str(bs)+'_njack'+args.njack+'.txt'
    #data = datadir +'/smu/allcounts_'+args.tracer+'_'+args.reg+str(zmin)+'_'+str(zmax)+zw+'_'+args.weight+'_lin_njack'+args.njack+args.nran+args.split+'.npy'
    data = './recon_cor_func_nojac_sub20_smooth10.npy'
    datajk = './recon_cor_func_good_sub20_smooth10.npy'
else:
    sys.exit('recon not supported yet')
    #data = datadir +'/smu/xipoles_LRG_'+args.rectype+args.reg+str(zmin)+'_'+str(zmax)+'_'+args.weight+'_lin'+str(bs)+'_njack'+args.njack+'.txt'

result = pycorr.TwoPointCorrelationFunction.load(datajk)

s, xiell, cov = result.get_corr(ells=ells, return_sep=True, return_cov=True)
covdia = np.diag(cov)
# making non-diagonal elements 0 future cosmologist reading this you might want to change that:
cov = np.diag(covdia)
result = pycorr.TwoPointCorrelationFunction.load(data)
rebinned = result[:(result.shape[0]//bs)*bs:bs]

s, xiell = result.get_corr(ells=ells, return_sep=True, return_cov=False)

std = np.diag(cov)**0.5
s_start = 0
#below needed because cov matrix starts at s = 20
if args.cov_type == 'theory':
    s_start = 20
    bin_start = s_start//bs
    print('removing '+str(bin_start)+' bins from the beginning of the data vector')
    s = s[bin_start:]
    xiell = xiell[bin_start:]
    std = std[bin_start:]


print(len(s),len(xiell),len(cov))    
#d = np.loadtxt(data).transpose()
xid = xiell#d[2]
rl = []
nbin = 0
#for i in range(0,len(d[0])):
for i in range(0,len(xid)):
    r = i*bs+bs/2.+binc+s_start
    rbc = .75*((r+bs/2.)**4.-(r-bs/2.)**4.)/((r+bs/2.)**3.-(r-bs/2.)**3.) #correct for pairs should have slightly larger average pair distance than the bin center
    rl.append(rbc) 
    if rbc > rmin and rbc < rmax:
        nbin += 1
rl = np.array(rl)

#if args.cov_type != 'theory':
#    covm = get_xi0cov() #will become covariance matrix to be used with data vector
covm = cov
cfac = args.cfac#5/4
covm *= cfac**2.
diag = []
for i in range(0,len(covm)):
    diag.append(np.sqrt(covm[i][i]))
diag = np.array(diag)
plt.plot(rl,rl*diag,label='used for fit')
#plt.plot(rl,rl*d[5],label='jack-knife')
plt.plot(rl,rl*std,label='jack-knife from paircounts')
plt.xlabel('s (Mpc/h)')
plt.ylabel(r's$\sigma$')
plt.legend()
plt.title('apply a factor '+str(round(cfac,2))+' to the mock/theory error')
plt.savefig('baofirstplot.jpg')
plt.clf()


spa=.001
outdir = os.environ['HOME']+'/DA02baofits/'
print('doing BAO fit')
print(covm.shape)
lik = bf.doxi_isolike(xid,covm,mod,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo=args.tracer+str(zmin)+str(zmax)+wm+str(bs),diro=outdir)
print(args.tracer+str(zmin)+str(zmax)+wm+str(bs))
print(args.tracer+'sm'+str(zmin)+str(zmax)+wm+str(bs))
print('minimum chi2 is '+str(min(lik))+' for '+str(nbin-5)+' dof')
print('doing no BAO fit')
liksm = bf.doxi_isolike(xid,covm,modsm,modsm,rl,bs=bs,rmin=rmin,rmax=rmax,npar=3,sp=1.,Bp=.4,rminb=50.,rmaxb=maxb,spa=spa,mina=.8,maxa=1.2,Nmock=Nmock,v='',wo=args.tracer+'sm'+str(zmin)+str(zmax)+wm+str(bs),diro=outdir)
#print(lik)
#print(liksm)
al = [] #list to be filled with alpha values
for i in range(0,len(lik)):
    a = .8+spa/2.+spa*i
    al.append(a)
#below assumes you have matplotlib to plot things, if not, save the above info to a file or something

sigs = sigreg_c12(al,lik)
print('result is alpha = '+str((sigs[2]+sigs[1])/2.)+'+/-'+str((sigs[2]-sigs[1])/2.))

plt.plot(al,lik-min(lik),'k-',label='BAO template')
plt.plot(al,liksm-min(lik),'k:',label='no BAO')
plt.xlabel(r'$\alpha$ (relative isotropic BAO scale)')
plt.ylabel(r'$\Delta\chi^{2}$')
plt.legend()
plt.savefig('baosecondplot.jpg')
plt.close()

