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
        f = self.__dict__.get('f', None)
        if f is not None:
            s = f * (2j * np.pi)
        else:
            w = self.__dict__.get('w', None)
            if w is not None:
                s = w * 1j
        self.__dict__['s'] = s
        return s

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
            self.__class__(
                tf=self.tf_sm * other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            self.__class__(
                tf=self.tf_sm * other,
                **self.__init_kw()
            )
        return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            self.__class__(
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
            self.__class__(
                tf=self.tf_sm / other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            self.__class__(
                tf=self.tf_sm / other,
                **self.__init_kw()
            )
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            self.__class__(
                tf=other / self.tf_sm,
                **self.__init_kw()
            )
        else:
            return NotImplemented
        return NotImplemented

    def inv(self):
        self.__class__(
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
            self.__class__(
                tf=self.tf_sm + other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            tf = self.tf_sm + other
            if self.snr_sm:
                snr = (self.snr_sm * abs(self.tf_sm)) / abs(tf)
            else:
                snr = self.snr_sm
            self.__class__(
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
            self.__class__(
                tf=self.tf_sm + other.tf_sm,
                **self.__init_kw(other)
            )
        elif isinstance(other, numbers.Number):
            tf = self.tf_sm - other
            if self.snr_sm:
                snr = (self.snr_sm * abs(self.tf_sm)) / abs(tf)
            else:
                snr = self.snr_sm
            self.__class__(
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
            self.__class__(
                tf=tf,
                **self.__init_kw(snr=snr)
            )
        else:
            return NotImplemented
        return NotImplemented

