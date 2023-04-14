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

from ..algorithms.statespace.dense import zpk_algorithms

from ..algorithms.zpk import srootset, zrootset
from . import siso
from . import ss
from . import util
from . import response


class ZPK(siso.SISOCommonBase):
    """
    ZPK class to represent SISO Transfer functions.

    This class internally uses the s-domain in units of radial frequency and gain.
    """
    fiducial_rtol = 1e-8
    fiducial_atol = 1e-13
    _SS = None

    def __init__(
        self, *,
        z: srootset.SDomainRootSet,
        p: srootset.SDomainRootSet,
        k: numbers.Complex,
        hermitian: bool = True,
        time_symm: bool = False,
        dt=None,
        fiducial=None,
        fiducial_rtol=None,
        fiducial_atol=None,
    ):
        """
        response_f: give a set of response points to use to verify that various algorithms preserve the transfer function. If None (or not given) then a set of response points are established using pole and zero locations.
        response_w: alternate radial frequency input for response_f
        response_s: alternate s-domain imaginary radial input for response_f

        response: transfer function responses mapped to the response_w values
        """
        if dt is None:
            assert(isinstance(z, srootset.SDomainRootSet))
            assert(isinstance(p, srootset.SDomainRootSet))
        else:
            assert(isinstance(z, zrootset.ZDomainRootSet))
            assert(isinstance(p, zrootset.ZDomainRootSet))

        self.zeros = z
        self.poles = p
        self.k = k
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt

        if self.hermitian:
            assert(k.imag == 0)

        if dt is None:
            if self.hermitian:
                assert(self.zeros.mirror_real)
                assert(self.poles.mirror_real)
            if self.time_symm:
                assert(self.zeros.mirror_imag)
                assert(self.poles.mirror_imag)
        else:
            if self.hermitian:
                assert(self.zeros.mirror_real)
                assert(self.poles.mirror_real)
            if self.time_symm:
                assert(self.zeros.mirror_disc)
                assert(self.poles.mirror_disc)
        self.test_fresponse(
            fiducial=fiducial,
            rtol=fiducial_rtol,
            atol=fiducial_atol,
            update=True,
        )
        return

    @property
    def z(self):
        """
        return an array of all of the zeros
        """
        return self.zeros.all()

    @property
    def p(self):
        """
        return an array of all of the zeros
        """
        return self.poles.all()

    def __str__(self):
        zstr = str(self.zeros)
        pstr = str(self.poles)
        if '\n' in zstr or '\n' in pstr:
            return "zpk(k={k},\n    z = {z},\n    p = {p})".format(k=self.k, z=zstr, p=pstr)
        else:
            return "zpk(z={z}, p={p}, k={k})".format(k=self.k, z=zstr, p=pstr)
        return

    def _fiducial_w_set(self, rtol):
        if self.dt is None:
            # create a list of points at each resonance and zero, as well as 1 BW away
            rt_rtol = rtol**0.5
            domain_w = [
                self.zeros.r_line,
                self.zeros.c_plane.imag,
                abs(self.zeros.c_plane),
                self.poles.r_line,
                self.poles.c_plane.imag,
                abs(self.poles.c_plane),
            ]

            # augment the list to include midpoints between all resonances
            domain_w = np.sort(np.concatenate(domain_w)).real + rt_rtol
            # and midpoints
            domain_w = np.concatenate([domain_w, (domain_w[0:-1] + domain_w[1:])/2])

            if len(domain_w) > self.N_MAX_FID:
                warnings.warn(f"StateSpace is large (>{self.N_MAX_FID} states), using reduced response fiducial auditing heuristics. TODO to make this smarter", util.NumericalWarning)
                domain_w = np.asarray([rt_rtol])
            return domain_w
        else:
            # create a list of points at each resonance and zero, as well as 1 BW away
            rt_rtol = rtol**0.5
            domain_w = [
                np.angle(self.zeros.disc_line, deg=False) / self.dt,
                np.angle(self.zeros.cplx_plane, deg=False) / self.dt,
                np.pi * (1 - self.zeros.real_line[self.zeros.real_line > 0]) / self.dt,
                np.angle(self.poles.disc_line, deg=False) / self.dt,
                np.angle(self.poles.cplx_plane, deg=False) / self.dt,
                np.pi * (1 - self.poles.real_line[self.poles.real_line > 0]) / self.dt,
            ]

            # augment the list to include midpoints between all resonances
            domain_w = np.sort(np.concatenate(domain_w)).real + rt_rtol
            # and midpoints
            domain_w = np.concatenate([domain_w, (domain_w[0:-1] + domain_w[1:])/2])

            if len(domain_w) > self.N_MAX_FID:
                warnings.warn(f"StateSpace is large (>{self.N_MAX_FID} states), using reduced response fiducial auditing heuristics. TODO to make this smarter", util.NumericalWarning)
                domain_w = np.asarray([rt_rtol])
            return domain_w

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        yield tuple(self.zeros.astuple())
        yield tuple(self.poles.astuple())
        yield self.k

    @property
    def asZPK(self):
        return self

    def flip_to_stable(self):
        return self.__class__(
            z=self.zeros.flip_to_stable(),
            p=self.poles.flip_to_stable(),
            k=self.k,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
            fiducial=self.fiducial.like_empty(),  # will have to rebuild the fiducial
            fiducial_rtol=self.fiducial_rtol,
            fiducial_atol=self.fiducial_atol,
        )

    @property
    def asSS(self):
        if self._SS is not None:
            self._SS
        assert(self.hermitian)
        z = self.zeros.drop_mirror_imag()
        p = self.poles.drop_mirror_imag()
        # Currently only supports hermitian inputs
        ABCDE = zpk_algorithms.zpk_rc(
            Zc=z.c_plane,
            Zr=z.r_line,
            Pc=p.c_plane,
            Pr=p.r_line,
            k=self.k,
            convention="scipy",
            orientation="upper"
        )

        self._SS = ss.statespace(
            ABCDE,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            fiducial=self.fiducial,
            fiducial_rtol=self.fiducial_rtol,
            fiducial_atol=self.fiducial_atol,
            flags={"schur_real_upper", "hessenburg_upper"},
        )
        return self._SS

    def fresponse(
            self,
            *,
            f=None,
            w=None,
            s=None,
            z=None,
            with_lnG=False
    ):
        domain = util.build_sorz(
            f=f,
            w=w,
            s=s,
            z=z,
            dt=self.dt,
        )

        # return an empty response
        # attempting to compute it on
        # empty input can throw errors
        if len(domain) == 0:
            return response.SISOFResponse(
                tf=domain,
                w=w,
                f=f,
                s=s,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                snr=None,
            )

        h, lnG = self.poles.fresponse_lnG(domain, 1)
        h, lnG = self.zeros.fresponse_lnG(domain, self.k/h, -lnG)

        if with_lnG:
            return h, lnG
        else:
            tf = h * np.exp(lnG)

        return response.SISOFResponse(
            tf=tf,
            w=w,
            f=f,
            s=s,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            snr=None,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asZPK
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            if len(self.fiducial) + len(other.fiducial) < self.N_MAX_FID:
                slc = slice(None, None, 1)
            else:
                slc = slice(None, None, 2)
            fid_other_self = other.fresponse(**self.fiducial.domain_kw(slc))
            fid_self_other = self.fresponse(**other.fiducial.domain_kw(slc))
            assert(self.dt == other.dt)
            return self.__class__(
                z=self.zeros * other.zeros,
                p=self.poles * other.poles,
                k=self.k * other.k,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
                fiducial=(self.fiducial[slc] * fid_other_self).concatenate(fid_self_other * other.fiducial[slc]),
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                z=self.zeros,
                p=self.poles,
                k=self.k * other,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
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
                z=self.zeros,
                p=self.poles,
                k=other * self.k,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
                fiducial=other * self.fiducial,
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asZPK
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            if len(self.fiducial) + len(other.fiducial) < self.N_MAX_FID:
                slc = slice(None, None, 1)
            else:
                slc = slice(None, None, 2)
            fid_other_self = other.fresponse(**self.fiducial.domain_kw(slc))
            fid_self_other = self.fresponse(**other.fiducial.domain_kw(slc))
            assert(self.dt == other.dt)
            return self.__class__(
                z=self.zeros * other.poles,
                p=self.poles * other.zeros,
                k=self.k / other.k,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
                fiducial=(self.fiducial[slc] / fid_other_self).concatenate(fid_self_other / other.fiducial[slc]),
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                z=self.zeros,
                p=self.poles,
                k=self.k / other,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
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
                z=self.poles,
                p=self.zeros,
                k=other / self.k,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
                fiducial=other / self.fiducial,
                fiducial_rtol=self.fiducial_rtol,
                fiducial_atol=self.fiducial_atol,
            )
        else:
            return NotImplemented

    def inv(self):
        return self.__class__(
            z=self.poles,
            p=self.zeros,
            k=1 / self.k,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
            fiducial=1/self.fiducial,
            fiducial_rtol=self.fiducial_rtol,
            fiducial_atol=self.fiducial_atol,
        )

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

    def square_root(self, keep_i=True):
        """
        If the current filter is time_symm, then return the stable, minimum delay square root filter
        """
        assert(self.time_symm)
        # just make a copy, but don't interpret the poles has having mirrors
        if keep_i:
            z = self.zeros.__class__(
                c_plane=self.zeros.c_plane,
                r_line=self.zeros.r_line,
                i_line=self.zeros.i_line,
                z_point=self.zeros.z_point,
                hermitian=self.zeros.hermitian,
                time_symm=False,
            )
            p = self.poles.__class__(
                c_plane=self.poles.c_plane,
                r_line=self.poles.r_line,
                i_line=self.poles.i_line,
                z_point=self.poles.z_point,
                hermitian=self.poles.hermitian,
                time_symm=False,
            )
        else:
            z = self.zeros.__class__(
                c_plane=self.zeros.c_plane,
                r_line=self.zeros.r_line,
                z_point=self.zeros.z_point,
                hermitian=self.zeros.hermitian,
                time_symm=False,
            )
            p = self.poles.__class__(
                c_plane=self.poles.c_plane,
                r_line=self.poles.r_line,
                z_point=self.poles.z_point,
                hermitian=self.poles.hermitian,
                time_symm=False,
            )

        # can't re-use the fiducial, so create a new one from
        # the previous points
        return self.__class__(
            z=z,
            p=p,
            k=self.k**2,
            hermitian=self.hermitian,
            time_symm=False,
            dt=self.dt,
            fiducial=None,
            fiducial_rtol=self.fiducial_rtol,
            fiducial_atol=self.fiducial_atol,
        )


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
        angular: bool = None,
        fiducial=None,
        fiducial_w=None,
        fiducial_f=None,
        fiducial_s=None,
        fiducial_z=None,
        fiducial_rtol=None,
        fiducial_atol=None,
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

    angular: indicates whether the roots z, p, zr, zc, .. are in angular or frequency units.
    Note that this does not affect the interpretation of the gain. By default, it is determined
    by the convention.

    """
    if dt is None:
        tRootSet = srootset.SDomainRootSet
        if classifier is None:
            classifier = srootset.default_root_classifier
    else:
        tRootSet = zrootset.ZDomainRootSet
        if classifier is None:
            classifier = zrootset.default_root_classifier

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
        assert(fiducial is not None)
        k = 1
        k_was_None = True
    else:
        k_was_None = False

    classify_rootset = classifier.classify_function(
        tRootSet=tRootSet,
        hermitian=hermitian,
        time_symm=time_symm,
    )
    if z is not None:
        z = np.asarray(z)
        zRS = classify_rootset(z, 'zeros')
    else:
        zRS = None

    if p is not None:
        p = np.asarray(p)
        pRS = classify_rootset(p, 'poles')
    else:
        pRS = None

    if zc is not None or zr is not None or zi is not None:
        zRS2 = tRootSet(
            c_plane=zc,
            r_line=zr,
            i_line=zi,
            hermitian=hermitian,
            time_symm=time_symm,
        )
        if zRS is not None:
            zRS = zRS * zRS2
        else:
            zRS = zRS2

    if pc is not None or pr is not None or pi is not None:
        pRS2 = tRootSet(
            c_plane=pc,
            r_line=pr,
            i_line=pi,
            hermitian=hermitian,
            time_symm=time_symm,
        )
        if pRS is not None:
            pRS = pRS * pRS2
        else:
            pRS = pRS2

    if dt is None:
        if convention == 'scipy':
            root_normalization = 1
            gain_normalization = 1
            if angular is None:
                angular = True
        elif convention.lower() == 'iirrational':
            root_normalization = 1
            gain_normalization = (2*np.pi)**(len(pRS) - len(zRS))
            if angular is None:
                angular = False
        else:
            raise RuntimeError("Convention {} not recognized".format(convention))

        if not angular:
            root_normalization *= np.pi * 2

        ZPKnew = ZPK(
            z=root_normalization*zRS,
            p=root_normalization*pRS,
            k=gain_normalization*k,
            dt=dt,
            hermitian=hermitian,
            time_symm=time_symm,
            fiducial=None,
            fiducial_rtol=fiducial_rtol,
            fiducial_atol=fiducial_atol,
        )
    else:
        assert(angular is None)
        if convention == 'scipy':
            gain_normalization = 1
        else:
            raise RuntimeError("Convention {} not recognized".format(convention))

        ZPKnew = ZPK(
            z=zRS,
            p=pRS,
            k=gain_normalization*k,
            dt=dt,
            hermitian=hermitian,
            time_symm=time_symm,
            fiducial=None,
            fiducial_rtol=fiducial_rtol,
            fiducial_atol=fiducial_atol,
        )

    if ZPKprev:
        #TODO check that ZPKprev has the same dt sample rate
        ZPKnew = ZPKprev * ZPKnew

    fiducial = util.build_fiducial(
        fiducial=fiducial,
        fiducial_w=fiducial_w,
        fiducial_f=fiducial_f,
        fiducial_s=fiducial_s,
        fiducial_z=fiducial_z,
        dt=dt,
    )
    if k_was_None:
        ZPKnew.test_fresponse(
            fiducial=fiducial.like_empty(),
            rtol=fiducial_rtol,
            update=True,
        )
        norm_rel = fiducial.tf / ZPKnew.fiducial.tf
        norm_med = np.nanmedian(abs(norm_rel))
        if np.isfinite(norm_med):
            ZPKnew = ZPKnew * norm_med
            # print("NORM MEDIAN", norm_med)
            # TODO, make better error reporting that a conversion has failed
            np.testing.assert_allclose(
                norm_rel / norm_med, 1, rtol=ZPKnew.fiducial_rtol, atol=ZPKnew.fiducial_atol
            )
        else:
            assert(np.all(np.isfinite(ZPKnew.fiducial)))
    else:
        ZPKnew.test_fresponse(
            fiducial=fiducial,
            update=True,
        )

    return ZPKnew


