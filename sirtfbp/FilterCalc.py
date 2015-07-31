#-----------------------------------------------------------------------
#Copyright 2015 Daniel M. Pelt
#
#Contact: D.M.Pelt@cwi.nl
#Website: http://www.dmpelt.com
#
#
#This file is part of the PySIRT-FBP, a Python implementation of the
#SIRT-FBP tomographic reconstruction method.
#
#PySIRT-FBP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#PySIRT-FBP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PySIRT-FBP. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import numpy as np
import astra
from . import optomo
import os.path
import hashlib
import math
import base64
import scipy.io
import six
try:
    from six.moves import range
except ImportError:
    # six 1.3.0
    from six.moves import xrange
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    pyfftw=None

def xsToFilter(xs,p,alph,npix,nAng,filterSize):
    """Converts a filter in the image domain in real space to a filter
    in the sinogram domain in Fourier space.

    :param xs: Filter in image domain.
    :type xs: :class:`numpy.ndarray`
    :param p: Projector object to perform forward projections.
    :type p: :class:`OpTomo`
    :param alph: Relaxation parameter of the SIRT method.
    :type alph: :class:`float`
    :param npix: Number of detector pixels.
    :type npix: :class:`int`
    :param nAng: Number of acquired angles.
    :type nAng: :class:`int`
    :param filterSize: Number of filter pixels.
    :type filterSize: :class:`int`
    """
    ff = alph*(p*xs).reshape((nAng,npix))
    if pyfftw==None:
        f = np.zeros((nAng,2*(filterSize-1)+1),dtype=np.float32)
    else:
        f = pyfftw.n_byte_align_empty((nAng,2*(filterSize-1)+1),16,'float32')
        f.fill(0)
    f[:,0:int(npix/2)+1] = ff[:,int(npix/2):npix]
    f[:,f.shape[1]-int(npix/2):f.shape[1]] = f[:,1:int(npix/2)+1][:,::-1]
    if pyfftw==None:
        f_fft = np.fft.fft(f,axis=1)
    else:
        f_fft = pyfftw.interfaces.numpy_fft.fft(f,axis=1)
    filt = np.zeros((nAng,filterSize),dtype=np.float32)
    filt[:,0:filterSize] = f_fft[:,0:filterSize]
    return filt*2*nAng/np.pi

def getFilterFile(saveDir, proj_geom, ss, nIters, returnAll):
    """Creates a filename in ``saveDir`` that is unique
    to the combination of input parameters.

    :param saveDir: Folder to save file in.
    :type saveDir: :class:`string`
    :param proj_geom: ASTRA projection geometry.
    :type proj_geom: :class:`dict`
    :param ss: Supersampling that is used.
    :type ss: :class:`int`
    :param nIters: Number of iterations that is used.
    :type nIters: :class:`int`
    :param returnAll: True if all filters are returned, False if only the last one is returned.
    :type returnAll: :class:`bool`
    """
    hs = ""
    for i,j in sorted(proj_geom.items()):
        try:
            for k,l in sorted(j.items()):
                try:
                    hs = hs + k + ": " + l.tostring().encode('ascii') + "; "
                except (AttributeError,UnicodeDecodeError):
                    hs = hs + k + ": " + str(l) + "; "
        except AttributeError:
            try:
                hs = hs + i + ": " + j.tostring().encode('ascii') + "; "
            except (AttributeError,UnicodeDecodeError):
                hs = hs + i + ": " + str(j) + "; "
    hs = hs + 'ss: ' + str(ss) + "; "
    #hs = hs + 'nIters: ' + str(nIters) + "; "
    #hs = hs + 'returnAll: ' + str(returnAll) + "; "
    fn = base64.b64encode(six.b(hashlib.md5(six.b(hs)).hexdigest())).decode('utf-8') + ".mat"
    ffn = os.path.join(saveDir,fn)
    return ffn

def getFilterGeometry(proj_geom):
    """Returns a projection geometry for a FBP filter
    that has the correct size given the input ASTRA
    projection geometry.

    :param proj_geom: ASTRA projection geometry.
    :type proj_geom: :class:`dict`
    """
    npix = proj_geom['DetectorCount']
    if npix%2==0:
        npix = npix+1
    nexpow = int(pow(2, math.ceil(math.log(2 * proj_geom['DetectorCount'], 2))))
    filterSize = int(nexpow / 2) + 1
    return astra.create_proj_geom('parallel',1.0,filterSize,proj_geom['ProjectionAngles'])

cache={}

def getFilters(proj_geom, ss, nIters, saveDir=None, projector='cuda', returnAll=False):
    """Caluclate the SIRT-like filter for a given projection geometry.

    :param proj_geom: ASTRA projection geometry.
    :type proj_geom: :class:`dict`
    :param ss: Supersampling to use.
    :type ss: :class:`int`
    :param nIters: Number of SIRT iterations to approximate.
    :type nIters: :class:`int`
    :param saveDir: Folder to save cache file in. Set to ``None`` to disable caching.
    :type saveDir: :class:`string`
    :param projector: ASTRA projector to use (e.g. ``'line'``,``'linear'``,``'cuda'``, ... ).
    :type projector: :class:`string`
    :param returnAll: If True, return all filters up to ``nIters``, if False, only return last.
    :type returnAll: :class:`bool`
    """
    nCalc = nIters
    hasSave = False
    if not saveDir==None:
        # Check whether we have already calculated this filter before
        # If so, return the pre-computed filter
        ffn = getFilterFile(saveDir, proj_geom, ss, nIters, returnAll)
        if os.path.exists(ffn):
            try:
                dat = cache[str(nIters) + ffn]
            except KeyError:
                dat = scipy.io.loadmat(ffn)
                cache.clear()
                cache[str(nIters) + ffn]=dat
            curF = dat['outF']
            x = dat['x']
            xs = dat['xs']
            if curF.shape[0]>=nIters:
                return curF[nIters-1]
            hasSave=True
            nCalc = nIters+50

    nDet = proj_geom['DetectorCount']
    npix = nDet
    nAng = proj_geom['ProjectionAngles'].shape[0]

    # Always use an odd number of detectors (to be able to set the center pixel to one)
    if npix%2==0:
        npix = npix+1

    proj_geom = astra.create_proj_geom('parallel',1.0,npix,proj_geom['ProjectionAngles'])
    vol_geom = astra.create_vol_geom(npix)
    pi = astra.create_projector(projector,proj_geom,vol_geom)
    p = optomo.OpTomo(pi,extraOptions={'DetectorSuperSampling':ss,'PixelSuperSampling':ss})

    # Start by setting center pixel to one, rest to zero
    if hasSave==False:
        x = np.zeros((npix,npix),dtype=np.float32)
        x[npix//2,npix//2]=1
        xs = np.zeros_like(x)
    nexpow = int(pow(2, math.ceil(math.log(2 * nDet, 2))))
    filterSize = int(nexpow / 2) + 1
    alph = 1./(nAng*nDet)

    if not saveDir==None:
        outF = np.zeros((nCalc,nAng,filterSize),dtype=np.float32)

    startI=0
    if hasSave:
        outF[0:curF.shape[0]] = curF
        startI = curF.shape[0]

    # Actual calculation of the filter
    for i in range(startI,nCalc):
        # Sum each filter image
        xs += x
        # Apply A matrix to image
        x -= alph*(p.T*(p*x)).reshape(x.shape)
        astra.extrautils.clipCircle(x)
        if not saveDir==None:
            outF[i] = xsToFilter(xs,p,alph,npix,nAng,filterSize)

    if not saveDir==None:
        scipy.io.savemat(ffn,{'outF':outF, 'x':x, 'xs': xs}, do_compression=True)
    else:
        return xsToFilter(xs,p,alph,npix,nAng,filterSize)
    return outF[nIters-1]