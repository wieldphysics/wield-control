#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Functions to create a SISO state space system from inputs.
"""
import numbers
import numpy as np
import warnings

# from ..algorithms.statespace.dense import xfer_algorithms
from ..algorithms.statespace.dense import zpk_algorithms
from ..statespace.ss import RawStateSpaceUser, RawStateSpace

from .. import MIMO
from . import siso
from . import util
from . import zpk
from . import response


class SISOStateSpace(RawStateSpaceUser, siso.SISOCommonBase):
    fiducial_rtol = 1e-8
    fiducial_atol = 1e-13
    """
    class to represent SISO Transfer functions using dense state space matrix representations.
    """
    def __init__(
        self,
        ss,
        fiducial=None,
        fiducial_rtol=None,
        fiducial_atol=None,
    ):
        """
        flags: this is a set of flags that indicate computed property flags for the state space. Examples of such properties are "schur_real_upper", "schur_complex_upper", "hessenburg_upper", "balanced", "stable"
        """
        super().__init__(ss=ss)

        self.test_fresponse(
            fiducial=fiducial,
            rtol=fiducial_rtol,
            atol=fiducial_atol,
            update=True,
        )
        return

    def _fiducial_w_set(self, rtol):
        # create a list of poiints at each resonance and zero, as well as 1 BW away
        rt_rtol = rtol**0.5
        if self.A.shape[-1] < self.N_MAX_FID:
            z, p = self._zp

            zr = z[abs(z.imag) < 1e-10]
            zc = z[z.imag > 1e-10]
            pr = p[abs(p.imag) < 1e-10]
            pc = p[p.imag > 1e-10]

            # augment the list to include midpoints between all resonances
            domain_w = np.sort(np.concatenate([
                zr, zc.imag, abs(zc.imag) + abs(zc.real),
                pr, pc.imag, abs(pc.imag) + abs(pc.real),
            ])).real + rt_rtol
            # and midpoints
            domain_w = np.concatenate([domain_w, (domain_w[0:-1] + domain_w[1:])/2])
        else:
            warnings.warn(f"StateSpace is large (>{self.N_MAX_FID} states), using reduced response fiducial auditing heuristics. TODO to make this smarter", util.NumericalWarning)
            domain_w = np.asarray([rt_rtol])
        return domain_w

    _zp_tup = None

    @property
    def _zp(self):
        """
        Create a raw z, p tuple from the direct calculation
        """
        if self._zp_tup is None:
            z, p = zpk_algorithms.ss2zp(
                A=self.ss.A,
                B=self.ss.B,
                C=self.ss.C,
                D=self.ss.D,
                E=self.ss.E,
                idx_in=0,
                idx_out=0,
                fmt="scipy",
            )
            self._zp_tup = (z, p)
        return self._zp_tup

    _ZPK = None

    @property
    def asZPK(self):
        if self._ZPK is not None:
            self._ZPK
        z, p = self._zp
        # the gain is not specified here,
        # as it is established from the fiducial data
        self._ZPK = zpk.zpk(
            z, p,
            hermitian=self.ss.hermitian,
            time_symm=self.ss.time_symm,
            convention='scipy',
            fiducial=self.fiducial,
            fiducial_rtol=self.fiducial_rtol,
            fiducial_atol=self.fiducial_atol,
        )
        return self._ZPK

    @property
    def asSS(self):
        return self

    def mimo(self, row, col):
        """
        Convert this statespace system into a MIMO type with a single named input and output

        row: name of the single output
        col: name of the single input
        """
        return MIMO.MIMOStateSpace(
            ss=self.ss,
            inputs={col: 0},
            outputs={row: 0},
        )

    def fresponse(self, f=None, w=None, s=None, z=None):
        tf = self.ss.fresponse_raw(f=f, w=w, s=s, z=z)[..., 0, 0]
        return response.SISOFResponse(
            tf=tf,
            w=w, f=f, s=s, z=z,
            hermitian=self.ss.hermitian,
            time_symm=self.ss.time_symm,
            dt=self.dt,
            snr=None,
        )

    def inv(self):
        return self.__class__(
            self.ss.inv(),
            fiducial=1/self.fiducial,
            fiducial_rtol=self.fiducial_rtol,
            fiducial_atol=self.fiducial_atol,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asSS

            if len(self.fiducial) + len(other.fiducial) < self.N_MAX_FID:
                slc = slice(None, None, 1)
            else:
                slc = slice(None, None, 2)
            fid_other_self = other.fresponse(**self.fiducial.domain_kw(slc))
            fid_self_other = self.fresponse(**other.fiducial.domain_kw(slc))

            return self.__class__(
                ss=self.ss @ other.ss,
                fiducial=(self.fiducial[slc] * fid_other_self).concatenate(fid_self_other * other.fiducial[slc]),
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                ss=self.ss * other,
                fiducial=self.fiducial * other,
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                ss=other * self.ss,
                fiducial=other * self.fiducial,
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                self.ss / other,
                fiducial=self.fiducial / other,
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                ss=other * self.ss.inv(),
                fiducial=other / self.fiducial,
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        else:
            return NotImplemented

    def __pow__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            if other == -1:
                return self.inv()
            elif other == 1:
                return self
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __add__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            # convert to statespace form
            other = statespace(other)
            knownSS = True

        if knownSS or isinstance(other, SISOStateSpace):
            return self.__class__(
                self.ss + other.ss,
            )
        elif isinstance(other, siso.SISO):
            other = other.asSS
            warnings.warn(
                "Implicit conversion to statespace for math. Use filt.asSS or SISO.statespace(filt) to make explicit and suppress this warning"
            )
            # now recurse on this method
            return self + other
        return NotImplemented

    def __radd__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            # convert to statespace form
            other = statespace(other)
            # use commutativity of addition
            return self + other

        return NotImplemented

    def __sub__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            # convert to statespace form
            other = statespace(other)
            knownSS = True

        if knownSS or isinstance(other, SISOStateSpace):
            return self.__class__(
                ss=self.ss - other.ss,
            )
        elif isinstance(other, siso.SISO):
            other = other.asSS
            warnings.warn(
                "Implicit conversion to statespace for math. Use filt.asSS or SISO.statespace(filt) to make explicit and suppress this warning"
            )
            # now recurse on this method
            return self - other
        return NotImplemented

    def __rsub__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            # convert to statespace form
            other = statespace(other)
            knownSS = True

        if knownSS or isinstance(other, SISOStateSpace):
            return self.__class__(
                ss=other.ss - self.ss,
            )
        elif isinstance(other, siso.SISO):
            other = other.asSS
            warnings.warn(
                "Implicit conversion to statespace for math. Use filt.asSS or SISO.statespace(filt) to make explicit and suppress this warning"
            )
            # now recurse on this method
            return self - other
        return NotImplemented


def statespace(
    *args,
    A=None,
    B=None,
    C=None,
    D=None,
    E=None,
    hermitian=True,
    time_symm=False,
    dt=None,
    flags={},
    fiducial=None,
    fiducial_w=None,
    fiducial_f=None,
    fiducial_s=None,
    fiducial_z=None,
    fiducial_rtol=None,
    fiducial_atol=None,
):
    """
    Form a SISO LTI system from statespace matrices.

    """
    def all_none():
        return (
            (A is None)
            & (B is None)
            & (C is None)
            & (D is None)
            & (E is None)
        )
    if len(args) == 0:
        pass
        # must include the A=,B=,... arguments
    elif len(args) == 1:
        arg = args[0]
        if isinstance(arg, siso.SISO):
            arg = arg.asSS
        if isinstance(arg, SISOStateSpace):
            # TODO, check that some of the other arguments don't override it
            if all_none:
                return arg
            A = A if A is not None else arg.A
            B = B if B is not None else arg.B
            C = C if C is not None else arg.C
            D = D if D is not None else arg.D
            E = E if E is not None else arg.E
        elif isinstance(arg, (tuple, list)):
            A, B, C, D, E = arg
        elif isinstance(arg, numbers.Number):
            A = np.asarray([[]]).reshape(0, 0)
            B = np.asarray([[]]).reshape(0, 1)
            C = np.asarray([[]]).reshape(1, 0)
            D = np.asarray([[arg]])
            E = None
            arg = np.asarray(arg)
            if arg.imag == 0:
                hermitian = True
                if arg.real > 0:
                    time_symm = True
        else:
            # TODO convert scipy LTI and python-control objects too
            raise TypeError("Unrecognized conversion type for SISO.ss")
    elif len(args) == 4:
        A, B, C, D = args
        E = None
    elif len(args) == 5:
        A, B, C, D, E = args
    else:
        raise RuntimeError("Unrecognized argument format")

    fiducial = util.build_fiducial(
        fiducial=fiducial,
        fiducial_w=fiducial_w,
        fiducial_f=fiducial_f,
        fiducial_s=fiducial_s,
        fiducial_z=fiducial_z,
        dt=dt,
    )
    return SISOStateSpace(
        ss=RawStateSpace(
            A, B, C, D, E,
            dt=dt,
            flags=flags,
            hermitian=hermitian,
            time_symm=time_symm,
        ),
        fiducial=fiducial,
        fiducial_rtol=fiducial_rtol,
        fiducial_atol=fiducial_atol,
    )

