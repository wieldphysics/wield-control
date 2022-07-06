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

from ..statespace.dense import xfer_algorithms
from ..statespace.dense import zpk_algorithms
from ..statespace.ss import RawStateSpaceUser, RawStateSpace

from .. import MIMO
from . import siso
from . import zpk
from . import response


class NumericalWarning(UserWarning):
    pass


class SISOStateSpace(RawStateSpaceUser, siso.SISO):
    fiducial_rtol = 1e-8
    """
    class to represent SISO Transfer functions using dense state space matrix representations.
    """
    def __init__(
        self,
        ss,
        fiducial_s=None,
        fiducial_f=None,
        fiducial_w=None,
        fiducial=None,
        fiducial_rtol=None,
    ):
        """
        flags: this is a set of flags that indicate computed property flags for the state space. Examples of such properties are "schur_real_upper", "schur_complex_upper", "hessenburg_upper", "balanced", "stable"
        """
        super().__init__(ss=ss)

        domain_w = None
        if fiducial_f is not None:
            domain_w = 2 * np.pi * np.asarray(fiducial_f)
        if fiducial_w is not None:
            assert(domain_w is None)
            domain_w = np.asarray(fiducial_w)
        if fiducial_s is not None:
            assert(domain_w is None)
            domain_w = np.asarray(fiducial_s) / 1j

        self.test_response(
            s=fiducial_s,
            f=fiducial_f,
            w=fiducial_w,
            response=fiducial,
            rtol=fiducial_rtol,
            update=True,
        )
        return

    def test_response(
        self,
        s=None,
        f=None,
        w=None,
        response=None,
        rtol=None,
        update=False,
    ):
        domain_w = None
        if f is not None:
            domain_w = 2 * np.pi * np.asarray(f)
        if w is not None:
            assert(domain_w is None)
            domain_w = np.asarray(w)
        if s is not None:
            assert(domain_w is None)
            domain_w = np.asarray(s) / 1j

        if domain_w is not None and len(domain_w) == 0:
            if update:
                self.fiducial = domain_w
                self.fiducial_w = domain_w
                self.fiducial_rtol = rtol
                return
            return

        if rtol is None:
            rtol = self.fiducial_rtol
            if rtol is None:
                rtol = self.__class__.fiducial_rtol

        if domain_w is None:
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
                warnings.warn(f"StateSpace is large (>{self.N_MAX_FID} states), using reduced response fiducial auditing heuristics. TODO to make this smarter", NumericalWarning)
                domain_w = np.asarray([rt_rtol])

        self_response = self.fresponse(w=domain_w).tf
        if response is not None:
            if callable(response):
                response = response(w=domain_w)
            np.testing.assert_allclose(
                self_response,
                response,
                atol=0,
                rtol=rtol,
                equal_nan=False,
            )
        else:
            # give it one chance to select better points
            select_bad = (~np.isfinite(self_response)) | (self_response == 0)
            if update and np.any(select_bad):
                if np.all(select_bad):
                    domain_w = np.array([rt_rtol])
                    self_response = self.fresponse(w=domain_w).tf
                else:
                    self_response = self_response[~select_bad]
                    domain_w = domain_w[~select_bad]
            response = self_response

        if update:
            self.fiducial = response
            self.fiducial_w = domain_w
            self.fiducial_rtol = rtol
        return

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
            fiducial_w=self.fiducial_w,
            fiducial_rtol=self.fiducial_rtol,
        )
        return self._ZPK

    @property
    def asSTATESPACE(self):
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

    def fresponse(self, f=None, w=None, s=None):
        tf = self.ss.fresponse_raw(f=f, s=s, w=w)[..., 0, 0]
        return response.SISOFResponse(
            tf=tf,
            w=w, f=f, s=s,
            hermitian=self.ss.hermitian,
            time_symm=self.ss.time_symm,
            snr=None,
        )

    def inv(self):
        return self.__class__(
            self.ss.inv(),
            fiducial=1/self.fiducial,
            fiducial_w=self.fiducial_w,
            fiducial_rtol=self.fiducial_rtol,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asSS

            if len(self.fiducial_w) + len(other.fiducial_w) < self.N_MAX_FID:
                slc = slice(None, None, 1)
            else:
                slc = slice(None, None, 2)
            fid_other_self = other.fresponse(w=self.fiducial_w[slc]).tf
            fid_self_other = self.fresponse(w=other.fiducial_w[slc]).tf

            return self.__class__(
                ss=self.ss @ other.ss,
                fiducial=np.concatenate([
                    self.fiducial[slc] * fid_other_self,
                    fid_self_other * other.fiducial[slc]
                ]),
                fiducial_w=np.concatenate([
                    self.fiducial_w[slc],
                    other.fiducial_w[slc]
                ]),
                fiducial_rtol=self.fiducial_rtol,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                ss=self.ss * other,
                fiducial=self.fiducial * other,
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
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
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
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
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
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
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
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

        if knownSS or isinstance(other, siso.SISOStateSpace):
            self.__class__(
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

        if knownSS or isinstance(other, siso.SISOStateSpace):
            self.__class__(
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

        if knownSS or isinstance(other, siso.SISOStateSpace):
            self.__class__(
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
    hermitian=True,
    time_symm=False,
    dt=None,
    flags={},
    fiducial=None,
    fiducial_w=None,
    fiducial_f=None,
    fiducial_s=None,
    fiducial_rtol=None,
):
    """
    Form a SISO LTI system from statespace matrices.

    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, siso.SISO):
            arg = arg.asSS
        if isinstance(arg, SISOStateSpace):
            return arg
        elif isinstance(arg, (tuple, list)):
            A, B, C, D, E = arg
        elif isinstance(arg, numbers.Number):
            A = np.asarray([[]])
            B = np.asarray([[]])
            C = np.asarray([[]])
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
    return SISOStateSpace(
        RawStateSpace(
            A, B, C, D, E,
            dt=dt,
            flags=flags,
            hermitian=hermitian,
            time_symm=time_symm,
        ),
        fiducial=fiducial,
        fiducial_s=fiducial_s,
        fiducial_f=fiducial_f,
        fiducial_w=fiducial_w,
        fiducial_rtol=fiducial_rtol,
    )

