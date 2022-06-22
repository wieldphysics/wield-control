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

from ..statespace.dense import zpk_algorithms

from . import rootset
from .rootset import SDomainRootSet
from . import siso
from . import ss


class ZPK(siso.SISO):
    """
    ZPK class to represent SISO Transfer functions.

    This class internally uses the s-domain in units of radial frequency and gain.
    """
    def __init__(
        self,
        z: SDomainRootSet,
        p: SDomainRootSet,
        k: numbers.Complex,
        hermitian: bool = True,
        time_symm: bool = False,
        dt=None,
    ):
        assert(isinstance(z, SDomainRootSet))
        assert(isinstance(p, SDomainRootSet))

        self.z = z
        self.p = p
        self.k = k
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt
        
        if self.hermitian:
            assert(k.imag == 0)

        if dt is None:
            if self.hermitian:
                assert(self.z.mirror_real)
                assert(self.p.mirror_real)
            if self.time_symm:
                assert(self.z.mirror_imag)
                assert(self.p.mirror_imag)
        else:
            if self.hermitian:
                assert(self.z.mirror_real)
                assert(self.p.mirror_real)
            if self.time_symm:
                assert(self.z.mirror_disc)
                assert(self.p.mirror_disc)
        return

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        yield tuple(self.z.astuple())
        yield tuple(self.p.astuple())
        yield self.k

    @property
    def asZPK(self):
        return self

    _SS = None
    @property
    def asSS(self):
        if self._SS is not None:
            self._SS
        assert(self.hermitian)
        z = self.z.drop_mirror_imag()
        p = self.p.drop_mirror_imag()
        # Currently only supports hermitian inputs
        ABCDE = zpk_algorithms.zpk_rc(
            Zc=z.c_plane,
            Zr=z.r_line,
            Pc=p.c_plane,
            Pr=p.r_line,
            k=self.k,
            convention="scipy",
        )

        self._SS = ss.ss(
            ABCDE,
            hermitian=self.hermitian,
            time_symm=self.time_symm
        )
        return self._SS

    def response(self, *, f=None, w=None, s=None, with_lnG=False):
        domain = None
        if f is not None:
            domain = 2j * np.pi * np.asarray(f)
        if w is not None:
            assert(domain is None)
            domain = 1j * np.asarray(w)
        if s is not None:
            assert(domain is None)
            domain = np.asarray(s)

        h, lnG = self.p.response_lnG(domain, 1/self.k)
        h, lnG = self.z.response_lnG(domain, 1/h, -lnG)

        if with_lnG:
            return h, lnG
        else:
            return h * np.exp(lnG)

    def __mul__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asZPK
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            assert(self.dt == other.dt)
            return self.__class__(
                z=self.z * other.z,
                p=self.p * other.p,
                k=self.k * other.k,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                z=self.z,
                p=self.p,
                k=self.k * other,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                z=self.z,
                p=self.p,
                k=other * self.k,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __div__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                z=self.z,
                p=self.p,
                k=self.k / other,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented


def zpk(
        *args,
        z=None,
        p=None,
        k=None,
        zc=None,
        zr=None,
        zi=None,
        pc=None,
        pr=None,
        pi=None,
        response=None,
        response_kw=None,
        response_rtol=1e-6,
        hermitian=True,
        time_symm=False,
        convention='scipy',
        classifier=None,
        dt=None,
):
    """
    Form a SISO LTI system from root locations as zeros and poles.

    Takes a number of argument forms.

    The symmetry of the ZPK system can be specified, and default to standard
    values for SISO LTI filters of real variables. That is that they have
    Hermitian symmetry, where the roots are mirrored over the real line. This
    the response at negative frequencies to be the conjugate of that at
    positive frequencies. Another form of symmetry is time reversal symmetry,
    which requires that roots be mirrored over the real line. This forces roots
    to be unstable, and so is typically not physical. This form of symmetry is
    useful for representing frequency response of power spectra.

    the z,p,k arguments are checked if they respect the specified symmetry arguments.

    The arguments zc,zr,zi and pc,pr,pi are symmetry-specified forms. That is,
    they will implicitly be mirrored as needed to enforce the symmetry
    specifications. For this reason, they can be more concise, as they will not
    require providing redundant poles or zeros.

    zc and pc are the complex poles and should have positive imaginary term if
    hermitian is True. If time_symm is True, then zc and pc should also have
    negative real part. Warnings will be issued if roots are outside the
    expected areas, given the symmetry, as this can indicate possible redundant
    root specification.

    """
    if dt is None:
        tRootSet = SDomainRootSet
    else:
        raise NotImplementedError("Z Domain not yet implemented")
        # tRootSet = ZDomainRootSet

    if classifier is None:
        classifier = rootset.default_root_classifier

    ZPKprev = None
    if len(args) == 0:
        pass
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, siso.SISO):
            arg = arg.asZPK
        if isinstance(arg, ZPK):
            ZPKprev = arg
        elif isinstance(arg, (tuple, list)):
            assert(z is None)
            assert(p is None)
            assert(k is None)
            z, p, k = arg

    if len(args) == 2:
        # assume then that they are a z,p,k triple
        assert(z is None)
        assert(p is None)
        assert(k is None)
        z, p = args
        k = None

    if len(args) == 3:
        # assume then that they are a z,p,k triple
        assert(z is None)
        assert(p is None)
        assert(k is None)
        z, p, k = args

    if k is None:
        assert(response is not None)
        k = 1

    cut_rootset = classifier.classify_function(
        tRootSet=tRootSet,
        hermitian=hermitian,
        time_symm=time_symm,
    )
    if z is not None:
        zRS = cut_rootset(z, 'zeros')
    if p is not None:
        pRS = cut_rootset(p, 'poles')

    if zc is not None or zr is not None or zi is not None:
        zRS = tRootSet(
            c_plane=zc,
            r_line=zr,
            i_line=zi,
            hermitian=hermitian,
            time_symm=time_symm,
        ) * zRS

    if pc is not None or pr is not None or pi is not None:
        pRS = tRootSet(
            c_plane=pc,
            r_line=pr,
            i_line=pi,
            hermitian=hermitian,
            time_symm=time_symm,
        ) * pRS

    if convention == 'scipy':
        root_normalization = 1
        gain_normalization = 1
    else:
        raise RuntimeError("Convention {} not recognized".format(convention))

    ZPKnew = ZPK(
        z=root_normalization*zRS,
        p=root_normalization*pRS,
        k=gain_normalization*k,
        dt=dt,
        hermitian=hermitian,
        time_symm=time_symm,
    )
    if ZPKprev:
        ZPKnew = ZPKprev * ZPKnew

    if response is not None:
        if response_kw is None:
            # create a list of poiints at each resonance and zero, as well as 1 BW away
            w_ptlist = [
                ZPKnew.z.r_line,
                ZPKnew.z.c_plane.imag,
                ZPKnew.z.c_plane.imag + ZPKnew.z.c_plane.real,
                ZPKnew.p.r_line,
                ZPKnew.p.c_plane.imag,
                ZPKnew.p.c_plane.imag + ZPKnew.p.c_plane.real,
            ]
            # augment the list to include midpoints between all resonances
            w_ptlist = np.sort(np.concatenate(w_ptlist))
            w_ptlist = np.concatenate([w_ptlist, (w_ptlist[0:-1] + w_ptlist[1:])/2])
            response_kw = dict(w=w_ptlist)

        norm_pts1 = response(**response_kw)
        norm_pts2 = ZPKnew.response(**response_kw)
        norm_rel = norm_pts1 / norm_pts2
        norm_med = np.median(abs(norm_rel))
        # print("NORM MEDIAN", norm_med)
        if response_rtol is not None:
            # TODO, make better error reporting that a conversion has failed
            assert(np.all(abs(norm_rel / norm_med - 1) < response_rtol))
        ZPKnew = ZPKnew * norm_med

    return ZPKnew


