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
import warnings
import numpy as np

from . import siso


class SISOResponse(siso.SISO):
    """
    Class to hold transfer function response of SISO systems
    """

    def __init__(
        self,
        f=None,
        w=None,
        s=None,
        tf=1,
        snr=None,
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
            domain = s
            self.__dict__['s'] = s
        assert(domain is not None)
        shape = domain.shape

        self.tf_sm = tf
        self.tf = np.broadcast_to(self.tf_sm, shape)

        self.snr_sm = np.asarray(snr)
        self.snr = np.broadcast_to(self.snr_sm, shape)

        self.hermitian = hermitian
        self.time_symm = time_symm
        return

    @property
    def f(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here
        f = self.__dict__.get('f', None)
        if f is not None:
            return f

        w = self.__dict__.get('w', None)
        if w is not None:
            f = w / (2 * np.pi)
        else:
            s = self.__dict__.get('s', None)
            if s is not None:
                f = s / (2j * np.pi)
        self.__dict__['f'] = f
        return f

    @property
    def w(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here
        w = self.__dict__.get('w', None)
        if w is not None:
            return w

        f = self.__dict__.get('f', None)
        if f is not None:
            w = f * (2 * np.pi)
        else:
            s = self.__dict__.get('s', None)
            if s is not None:
                w = s / 1j
        self.__dict__['w'] = w
        return w

    @property
    def s(self):
        # currently @property is a "Data" descriptor
        # so the "f" in __dict__ doesn't take precedence
        # and we have to shunt it here
        s = self.__dict__.get('s', None)
        if s is not None:
            return s

        f = self.__dict__.get('f', None)
        if f is not None:
            s = f * (2j * np.pi)
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                s = w * 1j
        self.__dict__['s'] = s
        return s

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
    def fwith_mag(self):
        return self.f, self.tf_mag

    @property
    def fwith_deg(self):
        return self.f, self.tf_deg

    @property
    def fwith_deg45(self):
        """pair of self.f and angle with max a 45 and with NaN cuts. Good for loglog(*self.fwith_deg45, **kw)"""
        return self.domain_angle_cut(max=45, arg='f', deg=True)

    @property
    def fwith_deg90(self):
        """pair of self.f and angle with max a 90 and with NaN cuts. Good for loglog(*self.fwith_deg90, **kw)"""
        return self.domain_angle_cut(max=90, arg='f', deg=True)

    @property
    def fwith_deg135(self):
        """pair of self.f and angle with max a 135 and with NaN cuts. Good for loglog(*self.fwith_deg135, **kw)"""
        return self.domain_angle_cut(max=135, arg='f', deg=True)

    @property
    def fwith_deg180(self):
        """pair of self.f and angle with max a 180 and with NaN cuts. Good for loglog(*self.fwith_deg180, **kw)"""
        return self.domain_angle_cut(max=180, arg='f', deg=True)

    @property
    def fwith_deg225(self):
        """pair of self.f and angle with max a 225 and with NaN cuts. Good for loglog(*self.fwith_deg225, **kw)"""
        return self.domain_angle_cut(max=225, arg='f', deg=True)

    @property
    def fwith_deg270(self):
        """pair of self.f and angle with max a 270 and with NaN cuts. Good for loglog(*self.fwith_deg270, **kw)"""
        return self.domain_angle_cut(max=270, arg='f', deg=True)

    @property
    def fwith_deg315(self):
        """pair of self.f and angle with max a 315 and with NaN cuts. Good for loglog(*self.fwith_deg315, **kw)"""
        return self.domain_angle_cut(max=315, arg='f', deg=True)

    @property
    def fwith_rad(self):
        return self.f, self.tf_rad

    @property
    def wwith_mag(self):
        return self.w, self.tf_mag

    @property
    def wwith_deg(self):
        return self.w, self.tf_deg

    @property
    def wwith_deg45(self):
        return self.domain_angle_cut(max=45, arg='w', deg=True)

    @property
    def wwith_deg90(self):
        return self.domain_angle_cut(max=90, arg='w', deg=True)

    @property
    def wwith_deg135(self):
        return self.domain_angle_cut(max=135, arg='w', deg=True)

    @property
    def wwith_deg180(self):
        return self.domain_angle_cut(max=180, arg='w', deg=True)

    @property
    def wwith_deg225(self):
        return self.domain_angle_cut(max=225, arg='w', deg=True)

    @property
    def wwith_deg270(self):
        return self.domain_angle_cut(max=270, arg='w', deg=True)

    @property
    def wwith_deg315(self):
        return self.domain_angle_cut(max=315, arg='w', deg=True)

    @property
    def wwith_rad(self):
        return self.w, self.tf_rad

    def angle(self, max=90, deg=True):
        """
        Give the angle of the tf with a maximum angle. 

        max: the maximum angle to show. If None, then unrap
        deg: use degrees or not
        """
        if deg:
            min = max - 360
            return (np.angle(self.tf, deg=deg) + min) % 360 - min
        else:
            min = max - (2 * np.pi)
            return (np.angle(self.tf, deg=deg) + min) % (2 * np.pi) - min

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

        if deg:
            disc_dist = .85 * 360
        else:
            disc_dist = .85 * (2 * np.pi)
        ang = self.angle(max=max, deg=deg)

        snip = abs(ang[1:] - ang[:-1]) > disc_dist
        argsnip = np.argwhere(snip)

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
                kw['s'] = self.__dict__['s']
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
                if self.s is other.s:
                    return
                np.testing.assert_allclose(self.s, other.s)

    def __mul__(self, other):
        """
        """
        if isinstance(other, SISOResponse):
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
        if isinstance(other, SISOResponse):
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
        if isinstance(other, SISOResponse):
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
        if isinstance(other, SISOResponse):
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

