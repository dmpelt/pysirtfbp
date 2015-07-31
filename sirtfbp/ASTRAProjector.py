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

from . import optomo
from . import FilterCalc
import numpy as np
import astra

class ASTRAProjector(optomo.OpTomo):
    """Object that imitates a projection matrix with a given projector.

    This object can do forward projection by using the ``*`` operator::

        W = ASTRAProjector(proj_id)
        fp = W*image
        bp = W.T*sinogram

    It can also be used in minimization methods of the :mod:`scipy.sparse.linalg` module::

        W = ASTRAProjector(proj_id)
        output = scipy.sparse.linalg.lsqr(W,sinogram)

    Extra options for the forward projection and backprojection can be specified:

        W = ASTRAProjector(proj_id, extraOptions={'PixelSuperSampling':3,'DetectorSuperSampling':3})

    Compared to ``OpTomo``, this adds the ``fbp`` method, which performs FBP with a specified filter:

        W = ASTRAProjector(proj_id)
        rec = W.fbp(sinogram,filter)

    :param proj_id: ID to a projector.
    :type proj_id: :class:`int`
    :param extraOptions: Extra options to use during FP/BP (i.e. for cfg['option']).
    :type extraOptions: :class:`dict`
    """

    def __init__(self, proj_id, extraOptions = {}):
        optomo.OpTomo.__init__(self,proj_id, extraOptions=extraOptions)
        self.fpg = FilterCalc.getFilterGeometry(self.pg)
        self.fshape = (self.sshape[0], self.fpg['DetectorCount'])

    def __checkArray(self, arr, shp):
        if len(arr.shape)==1:
            arr = arr.reshape(shp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.flags['C_CONTIGUOUS']==False:
            arr = np.ascontiguousarray(arr)
        return arr

    def fbp(self,s,filt, extraOptions = {}):
        """Reconstruct an object using FBP with a custom filter.

        :param s: The projection data.
        :type s: :class:`numpy.ndarray`
        :param filt: The filter to use.
        :type filt: :class:`numpy.ndarray`
        :param extraOptions: Extra options to use during reconstruction (i.e. for cfg['option']).
        :type extraOptions: :class:`dict`
        """
        s = self.__checkArray(s, self.sshape)
        sid = self.data_mod.link('-sino',self.pg,s)
        f = self.__checkArray(filt, self.fshape)
        fid = self.data_mod.link('-sino',self.fpg,f)
        v = np.zeros(self.vshape,dtype=np.float32)
        vid = self.data_mod.link('-vol',self.vg,v)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        cfg['FilterType']='sinogram'
        cfg['FilterSinogramId']=fid
        cfg['ProjectorId'] = self.proj_id
        cfg['option'] = self.extraOptions.copy()
        cfg['option'].update(extraOptions)
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra.algorithm.delete(alg_id)
        self.data_mod.delete([vid,sid,fid])
        return v
