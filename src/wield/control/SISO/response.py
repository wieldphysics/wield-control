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
# import warnings
import numpy as np

from .. import MIMO

from . import siso


class SISOFResponse(siso.SISO):
    """
    Class to hold transfer function response of SISO systems
    """

    def __init__(
        self, *,
        f=None,
        w=None,
        s=None,
        z=None,
        tf=None,
        snr=None,
        dt=None,
        hermitian: bool = True,
        time_symm: bool = False,
    ):
        """
        snr of None means that the tf was computed numerically. A snr of False (or 0) means that it is from data but is unknown
        """
        domain = None
        if f is not None:
            domain = f
            # must use the dict assignment since there are properties which alias
            self.__dict__['f'] = f
        if w is not None:
            assert(domain is None)
            domain = w
            self.__dict__['w'] = w
        if s is not None:
            assert(domain is None)
            assert(dt is None)
            domain = s
            self.__dict__['s'] = s
        if z is not None:
            assert(domain is None)
            assert(dt is not None)
            assert(dt > 0)
            domain = z
            self.__dict__['z'] = z
        assert(domain is not None)
        shape = domain.shape

        if tf is not None:
            self.tf_sm = tf
            self.tf = np.broadcast_to(self.tf_sm, shape)
        else:
            self.tf_sm = None
            self.tf = None

        self.dt = dt

        if snr is not None:
            self.snr_sm = np.asarray(snr)
            self.snr = np.broadcast_to(self.snr_sm, shape)
        else:
            self.snr_sm = None
            self.snr = None

        self.hermitian = hermitian
        self.time_symm = time_symm
        return

    @property
    def f(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here

        f = self.__dict__.get('_f', None)
        if f is not None:
            return f

        f = self.__dict__.get('f', None)
        if f is None:
            w = self.__dict__.get('w', None)
            if w is not None:
                f = w / (2 * np.pi)
            else:
                if self.dt is None:
                    s = self.__dict__['s']
                    f = s / (2j * np.pi)
                else:
                    z = self.__dict__['z']
                    f = (np.angle(z) + np.log(abs(z))*1j) / (self.dt * 2 * np.pi)
        self.__dict__['_f'] = f
        return f

    @property
    def w(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here
        w = self.__dict__.get('_w', None)
        if w is not None:
            return w

        w = self.__dict__.get('w', None)
        if w is None:
            f = self.__dict__.get('f', None)
            if f is not None:
                w = f * (2 * np.pi)
            else:
                if self.dt is None:
                    s = self.__dict__['s']
                    w = s / 1j
                else:
                    z = self.__dict__['z']
                    w = (np.angle(z) + np.log(abs(z))*1j) / (self.dt)
        self.__dict__['_w'] = w
        return w

    @property
    def s(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here

        assert(self.dt is None)

        s = self.__dict__.get('_s', None)
        if s is not None:
            return s

        s = self.__dict__.get('s', None)
        if s is None:
            f = self.__dict__.get('f', None)
            if f is not None:
                s = f * (2j * np.pi)
            else:
                w = self['w']
                s = w * 1j
        self.__dict__['_s'] = s
        return s

    @property
    def z(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here

        assert(self.dt is None)

        z = self.__dict__.get('_z', None)
        if z is not None:
            return z

        z = self.__dict__.get('z', None)
        if z is None:
            f = self.__dict__.get('f', None)
            if f is not None:
                z = np.exp(f * (2j * np.pi) * self.dt)
            else:
                w = self['w']
                z = np.exp(w * 1j * self.dt)
        self.__dict__['_z'] = z
        return z

    def __getitem__(self, key):
        kw = dict(
            dt=self.dt,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
        )
        f = self.__dict__.get('f', None)
        if f is not None:
            kw['f'] = f[key]
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                kw['w'] = w[key]
            else:
                if self.dt is None:
                    kw['s'] = self.__dict__['s'][key]
                else:
                    kw['z'] = self.__dict__['z'][key]

        if self.tf_sm is not None:
            if len(self.tf_sm.shape) == 0:
                kw['tf'] = self.tf_sm
            else:
                kw['tf'] = self.tf[key]
        else:
            kw['tf'] = None

        if self.snr_sm is not None:
            if len(self.snr_sm.shape) == 0:
                kw['snr'] = self.snr_sm
            else:
                kw['snr'] = self.snr[key]
        else:
            kw['snr'] = None

        return self.__class__(**kw)

    def mimo(self, row, col):
        """
        Convert this statespace system into a MIMO type with a single named input and output

        row: name of the single output
        col: name of the single input
        """

        kw = dict(
            dt=self.dt,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
        )
        f = self.__dict__.get('f', None)
        if f is not None:
            kw['f'] = f
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                kw['w'] = w
            else:
                if self.dt is None:
                    kw['s'] = self.__dict__['s']
                else:
                    kw['z'] = self.__dict__['z']

        if self.snr_sm is not None:
            kw['snr'] = self.snr_sm.reshape(self.snr_sm.shape + (1, 1))
        else:
            kw['snr'] = None

        return MIMO.MIMOFResponse(
            tf=self.tf_sm.reshape(self.tf_sm.shape + (1, 1)),
            inputs={col: 0},
            outputs={row: 0},
            **kw
        )

    def domain_kw(self, key=None):
        """
        Return a dict with one of the domain keys.

        The argument "key" can be used to index or slice the domain before returning it.
        Its default of "None" does not perform any indexing
        """
        kw = dict()
        if key is not None:
            f = self.__dict__.get('f', None)
            if f is not None:
                kw['f'] = f[key]
            else:
                w = self.__dict__.get('w', None)
                if w is not None:
                    kw['w'] = w[key]
                else:
                    if self.dt is None:
                        kw['s'] = self.__dict__['s'][key]
                    else:
                        kw['z'] = self.__dict__['z'][key]
        else:
            f = self.__dict__.get('f', None)
            if f is not None:
                kw['f'] = f
            else:
                w = self.__dict__.get('w', None)
                if w is not None:
                    kw['w'] = w
                else:
                    if self.dt is None:
                        kw['s'] = self.__dict__['s']
                    else:
                        kw['z'] = self.__dict__['z']
        return kw

    def like_empty(self):
        kw = dict(
            tf=None,
            snr=None,
            dt=self.dt,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
        )
        f = self.__dict__.get('f', None)
        if f is not None:
            kw['f'] = f
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                kw['w'] = w
            else:
                if self.dt is None:
                    kw['s'] = self.__dict__['s']
                else:
                    kw['z'] = self.__dict__['z']

        return self.__class__(**kw)

    def __len__(self):
        f = self.__dict__.get('f', None)
        if f is not None:
            return len(f)

        w = self.__dict__.get('w', None)
        if w is not None:
            return len(w)
        else:
            if self.dt is None:
                s = self.__dict__['s']
                return len(s)
            else:
                z = self.__dict__['z']
                return len(z)

    @property
    def mag(self):
        return abs(self.tf)

    @property
    def deg(self):
        return np.angle(self.tf, deg=False)

    @property
    def rad(self):
        return np.angle(self.tf, deg=False)

    @property
    def tf_mag(self):
        return abs(self.tf)

    @property
    def tf_deg(self):
        return np.angle(self.tf, deg=False)

    @property
    def tf_rad(self):
        return np.angle(self.tf, deg=False)

    @property
    def fplot_mag(self):
        return self.f, self.tf_mag

    @property
    def fplot_deg(self):
        return self.f, self.tf_deg

    @property
    def fplot_deg45(self):
        """pair of self.f and angle with max a 45 and with NaN cuts. Good for loglog(*self.fplot_deg45, **kw)"""
        return self.domain_angle_cut(max=45, arg='f', deg=True)

    @property
    def fplot_deg90(self):
        """pair of self.f and angle with max a 90 and with NaN cuts. Good for loglog(*self.fplot_deg90, **kw)"""
        return self.domain_angle_cut(max=90, arg='f', deg=True)

    @property
    def fplot_deg135(self):
        """pair of self.f and angle with max a 135 and with NaN cuts. Good for loglog(*self.fplot_deg135, **kw)"""
        return self.domain_angle_cut(max=135, arg='f', deg=True)

    @property
    def fplot_deg180(self):
        """pair of self.f and angle with max a 180 and with NaN cuts. Good for loglog(*self.fplot_deg180, **kw)"""
        return self.domain_angle_cut(max=180, arg='f', deg=True)

    @property
    def fplot_deg225(self):
        """pair of self.f and angle with max a 225 and with NaN cuts. Good for loglog(*self.fplot_deg225, **kw)"""
        return self.domain_angle_cut(max=225, arg='f', deg=True)

    @property
    def fplot_deg270(self):
        """pair of self.f and angle with max a 270 and with NaN cuts. Good for loglog(*self.fplot_deg270, **kw)"""
        return self.domain_angle_cut(max=270, arg='f', deg=True)

    @property
    def fplot_deg315(self):
        """pair of self.f and angle with max a 315 and with NaN cuts. Good for loglog(*self.fplot_deg315, **kw)"""
        return self.domain_angle_cut(max=315, arg='f', deg=True)

    @property
    def fplot_rad(self):
        return self.f, self.tf_rad

    @property
    def wplot_mag(self):
        return self.w, self.tf_mag

    @property
    def wplot_deg(self):
        return self.w, self.tf_deg

    @property
    def wplot_deg45(self):
        return self.domain_angle_cut(max=45, arg='w', deg=True)

    @property
    def wplot_deg90(self):
        return self.domain_angle_cut(max=90, arg='w', deg=True)

    @property
    def wplot_deg135(self):
        return self.domain_angle_cut(max=135, arg='w', deg=True)

    @property
    def wplot_deg180(self):
        return self.domain_angle_cut(max=180, arg='w', deg=True)

    @property
    def wplot_deg225(self):
        return self.domain_angle_cut(max=225, arg='w', deg=True)

    @property
    def wplot_deg270(self):
        return self.domain_angle_cut(max=270, arg='w', deg=True)

    @property
    def wplot_deg315(self):
        return self.domain_angle_cut(max=315, arg='w', deg=True)

    @property
    def wplot_rad(self):
        return self.w, self.tf_rad

    def angle(self, max=90, deg=True):
        """
        Give the angle of the tf with a maximum angle. 

        max: the maximum angle to show. If None, then unrap
        deg: use degrees or not
        """
        if deg:
            min = max - 360
            return (np.angle(self.tf, deg=deg) - min) % 360 + min
        else:
            min = max - (2 * np.pi)
            return (np.angle(self.tf, deg=deg) + min) % (2 * np.pi) + min

    def domain_angle_cut(self, max=90, arg='f', deg=True):
        """
        return a (domain, angle) pair augmented with NaN points to cut the plot at discontinuities

        This is good for plotting
        """
        if arg == 'f':
            domain = self.f
        elif arg == 'w':
            domain = self.w
        elif arg == 's':
            domain = self.s
        elif arg == 'z':
            domain = self.z

        if deg:
            disc_dist = .85 * 360
        else:
            disc_dist = .85 * (2 * np.pi)
        ang = self.angle(max=max, deg=deg)

        snip = abs(ang[1:] - ang[:-1]) > disc_dist
        argsnip = np.argwhere(snip)[:, 0]

        domain_cuts = []
        angle_cuts = []

        argp = 0
        for arg in argsnip:
            domain_cuts.append(
                domain[argp: arg+1]
            )
            angle_cuts.append(
                ang[argp: arg+1]
            )
            # add the cut between the other two
            domain_cuts.append([(domain[arg] + domain[arg+1])/2])
            angle_cuts.append([float('NaN')])
            argp = arg+1
        domain_cuts.append(
            domain[argp:]
        )
        angle_cuts.append(
            ang[argp:]
        )

        return np.concatenate(domain_cuts), np.concatenate(angle_cuts)

    def __init_kw(self, other=None, snr=None):
        """
        Build a kw dictionary for building a new version of this class
        """
        if other is None:
            if snr is None:
                snr = self.snr_sm
            kw = dict(
                snr=snr,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
            )
        else:
            if snr is None:
                if self.snr_sm is None:
                    snr = other.snr_sm
                elif other.snr_sm is None:
                    snr = self.snr_sm
                else:
                    snr = (self.snr_sm**-2 + other.snr_sm**-2)**-0.5
            kw = dict(
                snr=snr,
                dt=self.dt,
                hermitian=self.hermitian and other.hermitian,
                time_symm=self.time_symm and other.time_symm,
            )
        f = self.__dict__.get('f', None)
        if f is not None:
            kw['f'] = f
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                kw['w'] = w
            else:
                if self.dt is None:
                    kw['s'] = self.__dict__['s']
                else:
                    kw['z'] = self.__dict__['z']
        return kw

    def check_domain(self, other):
        f = self.__dict__.get('f', None)
        if f is not None:
            if f is other.f:
                return
            np.testing.assert_allclose(f, other.f)
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                if w is other.w:
                    return
                np.testing.assert_allclose(w, other.w)
            else:
                assert(self.dt == other.dt)
                if self.dt is None:
                    if self.s is other.s:
                        return
                    np.testing.assert_allclose(self.s, other.s)
                else:
                    if self.z is other.z:
                        return
                    np.testing.assert_allclose(self.z, other.z)

    def __mul__(self, other):
        """
        """
        if isinstance(other, SISOFResponse):
            self.check_domain(other)
            return self.__class__(
                tf=self.tf_sm * other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                tf=self.tf_sm * other,
                **self.__init_kw()
            )
        return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                tf=other * self.tf_sm,
                **self.__init_kw()
            )
        else:
            return NotImplemented
        return NotImplemented

    def __truediv__(self, other):
        """
        """
        if isinstance(other, SISOFResponse):
            self.check_domain(other)
            return self.__class__(
                tf=self.tf_sm / other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                tf=self.tf_sm / other,
                **self.__init_kw()
            )
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                tf=other / self.tf_sm,
                **self.__init_kw()
            )
        else:
            return NotImplemented
        return NotImplemented

    def inv(self):
        return self.__class__(
            tf=1 / self.tf_sm,
            **self.__init_kw()
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

    def __add__(self, other):
        """
        """
        if isinstance(other, SISOFResponse):
            self.check_domain(other)
            return self.__class__(
                tf=self.tf_sm + other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            tf = self.tf_sm + other
            if self.snr_sm:
                snr = (self.snr_sm * abs(self.tf_sm)) / abs(tf)
            else:
                snr = self.snr_sm
            return self.__class__(
                tf=tf,
                **self.__init_kw(snr=snr)
            )
        return NotImplemented

    def __radd__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            tf = other + self.tf_sm
            if self.snr_sm:
                snr = (self.snr_sm * abs(self.tf_sm)) / abs(tf)
            else:
                snr = self.snr_sm
            self.__class__(
                tf=tf,
                **self.__init_kw(snr=snr)
            )
        else:
            return NotImplemented
        return NotImplemented

    def __sub__(self, other):
        """
        """
        if isinstance(other, SISOFResponse):
            self.check_domain(other)
            return self.__class__(
                tf=self.tf_sm + other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            tf = self.tf_sm - other
            if self.snr_sm:
                snr = (self.snr_sm * abs(self.tf_sm)) / abs(tf)
            else:
                snr = self.snr_sm
            return self.__class__(
                tf=tf,
                **self.__init_kw(snr=snr)
            )
        return NotImplemented

    def __rsub__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            tf = other - self.tf_sm
            if self.snr_sm:
                snr = (self.snr_sm * abs(self.tf_sm)) / abs(tf)
            else:
                snr = self.snr_sm
            return self.__class__(
                tf=tf,
                **self.__init_kw(snr=snr)
            )
        else:
            return NotImplemented
        return NotImplemented

    def concatenate(self, other):
        """
        Concatenate two TFs.
        """
        assert(self.dt == other.dt)

        if self.snr_sm is None:
            assert(other.snr_sm is None)
            snr = None
        elif other.snr_sm is None:
            assert(self.snr_sm is None)
        else:
            snr = np.concatenate([self.snr, other.snr])

        if self.tf_sm is None:
            assert(other.tf_sm is None)
            tf = None
        elif other.tf_sm is None:
            assert(self.tf_sm is None)
        else:
            tf = np.concatenate([self.tf, other.tf])

        kw = dict(
            snr=snr,
            tf=tf,
            dt=self.dt,
            hermitian=self.hermitian and other.hermitian,
            time_symm=self.time_symm and other.time_symm,
        )
        f = self.__dict__.get('f', None)
        if f is not None:
            kw['f'] = np.concatenate([f, other.f])
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                kw['w'] = w
                kw['w'] = np.concatenate([w, other.w])
            else:
                if self.dt is None:
                    kw['s'] = np.concatenate([self.s, other.s])
                else:
                    kw['z'] = np.concatenate([self.z, other.z])
        return self.__class__(**kw)
