import os
import logging
import datetime
import time
import argparse
import re
from glob import glob 
from turtle import color
import numpy as np
from nbodykit.lab import *
import yaml
import fitsio
from astropy.io import fits
from astropy.table import Table,vstack
from astropy.cosmology import FlatLambdaCDM
import pyrecon
from pyrecon import  utils,IterativeFFTParticleReconstruction,MultiGridReconstruction,setup_logging
from mockfactory import LagrangianLinearMock, utils, setup_logging
from cosmoprimo import *
from cosmoprimo.fiducial import DESI
from cosmoprimo.fiducial import Planck2018FullFlatLCDM
from LSS.tabulated_cosmo import TabulatedDESI
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator,\
project_to_multipoles, project_to_wp, setup_logging, TwoPointCounter
from pypower import CatalogFFTPower
from scipy.integrate import quad
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)
import tempfile
import healpy as hp
import random
import h5py


''' 
prints out some information about your galaxy catalog,
you might have to change the column names if you're not working with roman mocks
'''

# number of catalogs and cpus
Ncat = 4
ncpu = 40

ts = time.time()

# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--basedir", help="base directory for output, default is CSCRATCH",default='./data')
parser.add_argument("--version", help="catalog version; use 'test' unless you know what you are doing!",default='test')
parser.add_argument("--verspec",help="version for redshifts",default='e_da0.2')
parser.add_argument("--nthreads",help = "Number of CPU cores in use", default=ncpu)
parser.add_argument("--rectype",help="IFT or MG supported so far",default='MG')
parser.add_argument("--convention",help="recsym or reciso supported so far",default='reciso')
parser.add_argument("--nran",help="how many of the random files to concatenate",default=Ncat)
parser.add_argument("-z","--zrange", help ='list of min and max redshift cuts', default = [1.,2.])
parser.add_argument("-v", "--verbose", help=' For debugging, makes the code talkative', default = False, action='store_true')
parser.add_argument("-o","--overwrite", help= 'Re-generates concatanated files', default=False, action='store_true')


args = parser.parse_args()
    
setup_logging()

basedir = args.basedir
version = args.version
specrel = args.verspec
verbose = args.verbose
zrange = args.zrange
over = args.overwrite

print(zrange)
nran = int(args.nran)

if verbose == False:
    print(args)

maindir = basedir+'/'
ldirspec = maindir+specrel
dirout = ldirspec+'/'

if args.rectype == 'MG':
    recfunc = MultiGridReconstruction
    
if args.rectype == 'IFT':
    recfunc = IterativeFFTParticleReconstruction

if verbose == True:
    print('###') 
    print('running xi rom recon') 
    print(' ') 
    print('Listing Input Parameters')
    print('---')
    print(' ')
    print('Catalog Version: ' + version)
    if version != 'test':
        print('NOTICE: version is not listed as test') 
    print(' ')
    print('Redshift file location: ' + specrel)
    print(' ')
    print('Redshift range: ')
    print('      min: ' + str(min(zrange)) )
    print('      max: ' + str(max(zrange)) )
    print(' ')
    print("Number of cpu cores: " + str(ncpu))
    print(' ')
    print("Number of random files: " + str(Ncat))
    print(' ')
    print('Rectype: ' + args.rectype)
    print(' ')
    print('Convention: ' + args.convention)
    print(' ')
    print('Newly generated files will be saved to: ' + basedir)
    print(' ') 
    print('###')
    print('Checking for simulated data...')
    
    filescheckh5 = glob('Roman/*.fits')
    filescheckdat = glob('Roman/*.dat')
    
    if len(filescheckh5) == 0:
        print('0 Atlas files found. Please check input directories')
        print('Terminating Execution.')
        exit()
        
    else: 
        print(str(len(filescheckh5)) + ' Atlas files found.')
        print(' ')
  
    
    if len(filescheckdat) == 0:
        print('0 Random position files found. Please check input directories')
        print('Terminating Execution.')
        exit()
        
    else: 
        print(str(len(filescheckdat)) +' Random position files found.')
    print('###') 
    
if verbose == True:
    print('Runtime so far: '+ str(ts - time.time())+'s')
    print("Defining Functions: 0%")
    
def getrdz_fromxyz(cat):
    distance, ra, dec = utils.cartesian_to_sky(cat)
    # DESI tabulated values go to z = 10. Do we want to go further? im not sure 
    distance_to_redshift = utils.DistanceToRedshift(comoving_distance, zmax=10)
    z = distance_to_redshift(distance)
    return ra,dec,z 
    
# for reg in regl:
fb = dirout+'LRGzdone'

if verbose == True:
    print("Defining Functions: 50%")

def radec2hpix(nside, ra, dec):
    """ 
    Function transforms RA,DEC to HEALPix index in ring ordering
    
    parameters
    ----------
    nside : int
    
    ra : array_like
        right ascention in deg
    
    dec : array_like
        declination in deg
    
    
    returns
    -------
    hpix : array_like
        HEALPix indices
    
    """
    hpix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), nest=True)
    return hpix    
    
if verbose == True:
    print("Defining Functions: 100%")    
    
# defining boundaries and other things:

if verbose == True: 
    print('Defining boundaries...')
    print(' ')
edges = (np.linspace(0.01, 200, 51), np.linspace(-1., 1., 201))

if verbose == True:
    print('###')

# Loading cosmology and defining stuff:
ells = (0,2,4)
nm = 2
zmin = min(zrange)
zmax = max(zrange)
zmed = (zmin+zmax)/2.
bias = 0.9*zmed+0.5
kedges = np.arange(0.01, 0.4, 0.005)
njac=100

if verbose == True: 
    print('Loading Cosmology from Cosmoprimo package...')
    print(' ')
cosmo = DESI(h=0.6774,Omega_cdm=0.2589,Omega_b=0.05,sigma8=0.8159) # from package cosmoprimo
comoving_distance = cosmo.comoving_radial_distance
dist = cosmo.comoving_radial_distance(zmed)
dmin = cosmo.comoving_radial_distance(zmin)
dmax = cosmo.comoving_radial_distance(zmax)

boxcenter = [dist, 0, 0]
nmesh, boxsize, boxcenter, los = 512, 2000., boxcenter, 'x'
regl = ['_DN','_N']
position_columns = ['RA','DEC','Z']
col = ['b','r','g']

fo_camb = cosmo.get_fourier(engine='camb')
pk = fo_camb.pk_interpolator()
xi = pk.to_xi()

if verbose == True:
    print('Runtime so far: '+ str(ts - time.time())+'s')
    
if verbose == True: 
    print('Formulating Growth Rate...')
    print(' ')
ff = cosmo.sigma8_z(z=zmed,of='theta_cb')/cosmo.sigma8_z(z=zmed,of='delta_cb') # growth rate
#--------------------------

# Getting data columns:
    
RA = []
DEC = []
Z_cosmo = []
Z_obs = []
flux = []

if verbose == True:
    print('Runtime so far: '+ str(np.abs(ts - time.time()))+'s') 
    
if verbose == True: 
    print('Loading ATLAS variant fits files...') 
    print(' ') 


dname = 'Roman/ATLAS_small_concat_Ha16_wsub.fits'
fd=fits.getdata(dname)
# filtering out sub20=False:
#fd=fd[np.where(fd['sub20per']==True)]
RA.extend(fd['RA'])
DEC.extend(fd['DEC'])
Z_cosmo.extend(fd['Z_COS'])
Z_obs.extend(fd['Z_OBS'])

flux.extend(fd['flux_Halpha']*0.28685) # corrects for extinction of h-alpha

RA = np.array(RA)
DEC = np.array(DEC)
Z_cosmo = np.array(Z_cosmo)
Z_obs = np.array(Z_obs)
flux = np.array(flux)

if verbose == True: 
    print('Applying flux cuts...') 
    print(' ') 
flux_cut = np.where(flux > 1e-16)[0]

seld_obs = Z_obs[flux_cut] > zmin
seld_obs &= Z_obs[flux_cut] < zmax
Z_obs_filter = Z_obs[flux_cut][seld_obs]
RA_obs_filter = RA[flux_cut][seld_obs]
DEC_obs_filter = DEC[flux_cut][seld_obs]

print(np.median(Z_obs_filter))
print(np.mean(Z_obs_filter))
print(len(Z_obs_filter))

rname = 'Roman/Random_RADEC.dat'
with open(rname, 'r') as file:
    lines=file.readlines()
file.close()

rname = 'Roman/Random_RADEC_2.dat'
with open(rname, 'r') as file:
    lines2=file.readlines()
file.close()

linestotal = np.concatenate((lines,lines2),axis=0)
del(lines2)

# full random sample:
RA_rdm = np.empty(len(linestotal))
DEC_rdm = np.empty(len(linestotal))

for ct,x in enumerate(linestotal):
    RA_rdm[ct] = (x.split(' ')[0])
    DEC_rdm[ct] = (x.split(' ')[1])

del(linestotal) 

del(lines)

Z_rdm_obs = np.random.choice(Z_obs_filter, len(RA_rdm))

print(len(Z_rdm_obs))

print(len(RA_rdm))

# everything below here is just for making redshift/sky coverage plots

zbinned = np.linspace(1,2,100)
digitize = np.digitize(Z_obs_filter, zbinned)
bin_means = [Z_obs_filter[digitize == i].mean() for i in range(0, len(zbinned))] # check index
bin_means = np.array(bin_means)
dndz = bin_means/(0.01*2000) # dn/dz per square degree

# just need to change bins, weights, and zbins zmin and zmax
bins = 100
z = Z_obs_filter
weights = np.full_like(z, 2500)
zbins = np.linspace(1,2,bins+1)
counts, pbins = np.histogram(z,bins=zbins,weights=None)
bc = (pbins[:-1] + pbins[1:]) / 2
counts = counts/(0.01*2000)

plt.scatter(bc,counts,color='xkcd:Barbie Pink')
plt.xlabel('Redshift (z)',fontsize=16)
plt.ylabel(r'$\frac{dn}{dz}deg^{-2}$',fontsize=16)
plt.title('Galaxy Redshift Distribution',fontsize=16)
plt.savefig('dndz.jpg')
plt.close()

exit()

### Alberto's thing:
# for mocks:
def hpixsum(nside, ra, dec, weights=None):
    # this function agregates ra and dec onto HEALPix with nside and ring ordering.
    hpix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    npix = hp.nside2npix(nside)
    weight_hp = np.bincount(hpix, weights=weights, minlength=npix)
    return weight_hp

# read catalog
#data = ft.read(data_filename)
ra, dec = RA_obs_filter, DEC_obs_filter
data_pixmap = hpixsum(256, ra, dec) # use weights if needed

# to make the skyplot
plt.clf() 
fig = plt.gcf() #plt.figure(figsize=(10,5))
title = 'Mock Galaxy Sky Coverage'
hp.mollview(data_pixmap, rot=-130, xlabel='RA',ylabel='Dec',title=title, hold=True)
filename = 'radecoverage.jpg'
fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=200, facecolor='white')

#for randoms:
def hpixsum(nside, ra, dec, weights=None):
    # this function agregates ra and dec onto HEALPix with nside and ring ordering.
    hpix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    npix = hp.nside2npix(nside)
    weight_hp = np.bincount(hpix, weights=weights, minlength=npix)
    return weight_hp

# read catalog
#data = ft.read(data_filename)
ra, dec = RA_rdm, DEC_rdm
data_pixmap = hpixsum(256, ra, dec) # use weights if needed

# to make the skyplot
plt.clf() 
fig = plt.gcf() #plt.figure(figsize=(10,5))
title = 'Randoms Sky Coverage'
hp.mollview(data_pixmap, rot=-130, xlabel='RA',ylabel='Dec',title=title, hold=True)
filename = 'radecoverage_sorandom.jpg'
fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=200, facecolor='white')