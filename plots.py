'''
Run this to make so so many plots, assumes you've already run the scripts to generate the
correlation functions/errors as well as the model line
'''

import BAOfit as bf
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200.0

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

from pycorr import TwoPointCorrelationFunction, project_to_multipoles

ells = (0,2,4)

#getting model
wm = str(args.beta)+str(args.sfog)+str(args.dperp)+str(args.drad)
mod = np.loadtxt('BAOtemplates/xi0DESI'+wm+'10.0'+munw+'.dat').transpose()[1]
modsm = np.loadtxt('BAOtemplates/xi0smDESI'+wm+'10.0'+munw+'.dat').transpose()[1]

# sub10
data = os.environ['HOME']+'/cor_func_nojac_sub10.npy'
datajk = os.environ['HOME']+'/cor_func_good_sub10.npy'
# sub20
#datajk = os.environ['HOME']+'/cor_func_good.npy'
#data = os.environ['HOME']+'/cor_func_nojac.npy'

result = pycorr.TwoPointCorrelationFunction.load(datajk)

s, xiell, cov = project_to_multipoles(result, ells=ells, return_cov=True)
result = pycorr.TwoPointCorrelationFunction.load(data)
rebinned = result[:(result.shape[0]//bs)*bs:bs]

s, xiell = project_to_multipoles(result, ells=ells, return_cov=False)

result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_good_sub10_smooth10.npy')
s_m_rec, xiell_m_rec, jk_rec = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=True)
std_rec = np.diag(jk_rec)**(0.5)
result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_nojac_sub10_smooth10.npy')
s_m_rec, xiell_m_rec = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=False)

fmod = 'ximod'+args.tracer+str(zmin)+str(zmax)+wm+str(bs)+'.dat'
mod = np.loadtxt(fmod).transpose()
#print(mod[0],mod[1],mod[2],mod[3])
plt.plot(mod[0],mod[1],'k-',label='Model', linewidth=4)
plt.plot(s, xiell[1-1],label='Pre Reconstruction',c='xkcd:vibrant blue',alpha=0.7, ls='--', linewidth=4)
plt.errorbar(s_m_rec, xiell_m_rec[1-1], std_rec[0:50], capsize=2,label=r'Post Reconstruction, $10 \; h^{-1}{\rm Mpc}$ Smoothed',c='red',alpha=0.9, ls='-', linewidth=4)
plt.xlim(2,145)
plt.xticks(fontsize=16)
#plt.ylim(-15,25)
plt.yticks(fontsize=16)
plt.xlabel(r'$s\; (h^{-1}{\rm Mpc})$',fontsize=16)
plt.ylabel(r'$\xi(s)$',fontsize=16)
plt.title('Correlation Function Monopole, 90% of Galaxies',fontsize=15)
plt.legend(fontsize=11,loc='best')
plt.tight_layout()
plt.savefig(os.environ['HOME']+'/sub10smooth10nos2.jpg')
plt.close()
exit()

#plt.errorbar(rl,rl**2.*xid,rl**2*diag,fmt='ro',label='data')
fmod = 'ximod'+args.tracer+str(zmin)+str(zmax)+wm+str(bs)+'.dat'
mod = np.loadtxt(fmod).transpose()
print(mod[0],mod[1],mod[2],mod[3])
plt.plot(mod[0],mod[0]**2.*mod[1],'k-',label='Model', linewidth=4)
plt.plot(s, xiell[1-1]*s**2,label='Pre Reconstruction',c='xkcd:vibrant blue',alpha=0.7, ls='--', linewidth=4)
plt.errorbar(s_m_rec, xiell_m_rec[1-1]*s_m_rec**2, std_rec[0:50]*s_m_rec**2, capsize=2,label=r'Post Reconstruction, $10 \; h^{-1}{\rm Mpc}$ Smoothed',c='red',alpha=0.9, ls='-', linewidth=4)
plt.xlim(45,145)
plt.xticks(fontsize=16)
plt.ylim(-15,25)
plt.yticks(fontsize=16)
plt.xlabel(r'$s\; (h^{-1}{\rm Mpc})$',fontsize=16)
plt.ylabel(r's^2 $\xi(s)\;(h^{-2}{\rm Mpc^2})$',fontsize=16)
plt.title('Correlation Function Monopole, 90% of Galaxies',fontsize=15)
plt.legend(fontsize=11,loc=3)
plt.tight_layout()
plt.savefig(os.environ['HOME']+'/sub10smooth10.jpg')
plt.close()

# overplotting different smoothing scales:

result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_nojac_sub10_smooth10.npy')
s_m_rec10, xiell_m_rec10 = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=False)
#error bars for 15:
result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_good_sub10_smooth15.npy')
s_m_rec, xiell_m_rec, jk_rec = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=True)
std_rec15 = np.diag(jk_rec)**(0.5)
result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_nojac_sub10_smooth15.npy')
s_m_rec15, xiell_m_rec15 = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=False)

plt.plot(mod[0],mod[0]**2.*mod[1],'k-',label='Model', linewidth=3)
plt.plot(s_m_rec, xiell_m_rec[1-1]*s_m_rec**2,label=r'$7 \; h^{-1}{\rm Mpc}$ Smoothed',c='#648fff',alpha=0.9, ls='-', linewidth=3)
plt.plot(s_m_rec10, xiell_m_rec10[1-1]*s_m_rec10**2,label=r'$10 \; h^{-1}{\rm Mpc}$ Smoothed',c='#dc267f',alpha=0.9, ls='--', linewidth=3)
plt.plot(s_m_rec15, xiell_m_rec15[1-1]*s_m_rec15**2,label=r'$15 \; h^{-1}{\rm Mpc}$ Smoothed',c='#fe6100',alpha=0.9, ls='-.', linewidth=3)
#plt.errorbar(s_m_rec15, xiell_m_rec15[1-1]*s_m_rec15**2, std_rec15[0:50]*s_m_rec15**2, capsize=2,label=r'$15 \; h^{-1}{\rm Mpc}$ Smoothed',c='#ffb000',alpha=0.9, ls='-', linewidth=3)
plt.xlim(50,140)
plt.xticks(fontsize=16)
plt.ylim(-10,20)
plt.yticks(fontsize=16)
plt.xlabel(r'$s\; (h^{-1}{\rm Mpc})$',fontsize=16)
plt.ylabel(r'$s^2 \xi(s)\;(h^{-2}{\rm Mpc^2})$',fontsize=16)
plt.title('Correlation Function Monopole, 90% of Galaxies',fontsize=15)
plt.legend(fontsize=12,loc=3)
plt.tight_layout()
plt.savefig(os.environ['HOME']+'/sub10allsmooth.jpg')
plt.close()

#getting sub 20 unrecon too:

# sub10
data = os.environ['HOME']+'/cor_func_nojac_sub10.npy'
datajk = os.environ['HOME']+'/cor_func_good_sub10.npy'
# sub20
datajk20 = os.environ['HOME']+'/cor_func_good.npy'
data20 = os.environ['HOME']+'/cor_func_nojac.npy'

result = pycorr.TwoPointCorrelationFunction.load(datajk20)

s20, xiell20, cov20 = project_to_multipoles(result, ells=ells, return_cov=True)
result = pycorr.TwoPointCorrelationFunction.load(data20)

s20, xiell20 = project_to_multipoles(result, ells=ells, return_cov=False)


# overplotting two best:

result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_good_sub20_smooth10.npy')
s_m_rec20_7, xiell_m_rec20_7, jk_rec = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=True)
std_rec20_7 = np.diag(jk_rec)**(0.5)
result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_nojac_sub20_smooth10.npy')
s_m_rec20_7, xiell_m_rec20_7 = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=False)

result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_good_sub10_smooth10.npy')
s_m_rec10_15, xiell_m_rec10_15, jk_rec = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=True)
std_rec10_15 = np.diag(jk_rec)**(0.5)
result_mocks_rec = TwoPointCorrelationFunction.load(os.environ['HOME']+'/recon_cor_func_nojac_sub10_smooth10.npy')
s_m_rec10_15, xiell_m_rec10_15 = project_to_multipoles(result_mocks_rec, ells=ells, return_cov=False)

plt.plot(mod[0],mod[0]**2.*mod[1],'k-',label='Model', linewidth=3)
plt.plot(s, xiell[1-1]*s**2,label='Pre Reconstruction, 90%',c='xkcd:vibrant blue',alpha=0.7, ls='--', linewidth=3)
plt.plot(s20, xiell20[1-1]*s20**2,label='Pre Reconstruction, 80%',c='#fe6100',alpha=0.7, ls='--', linewidth=3)
plt.errorbar(s_m_rec10_15, xiell_m_rec10_15[1-1]*s_m_rec10_15**2,std_rec10_15[0:50]*s_m_rec10_15**2, capsize=2,label=r'$10 \; h^{-1}{\rm Mpc}$ Smoothed, 90%',c='#648fff',alpha=0.9, ls='-', linewidth=3)
plt.errorbar(s_m_rec20_7, xiell_m_rec20_7[1-1]*s_m_rec20_7**2, std_rec20_7[0:50]*s_m_rec20_7**2, capsize=2,label=r'$10 \; h^{-1}{\rm Mpc}$ Smoothed, 80%',c='#dc267f',alpha=0.9, ls='-', linewidth=3)
plt.xlim(50,140)
plt.xticks(fontsize=16)
plt.ylim(-10,20)
plt.yticks(fontsize=16)
plt.xlabel(r'$s\; (h^{-1}{\rm Mpc})$',fontsize=16)
plt.ylabel(r'$s^2 \xi(s)\;(h^{-2}{\rm Mpc^2})$',fontsize=16)
plt.title('Correlation Function Monopole',fontsize=15)
plt.legend(fontsize=12,loc=3)
plt.tight_layout()
plt.savefig(os.environ['HOME']+'/besties_w_unrecon.jpg')
plt.close()

