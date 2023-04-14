#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np
import warnings


class SISO:
    # maximum number of fiducial points to maintain during operations
    N_MAX_FID = 50


class SISOCommonBase(SISO):
    def test_fresponse(
        self,
        fiducial=None,
        rtol=None,
        atol=None,
        update=False,
    ):
        if rtol is None:
            rtol = self.fiducial_rtol
            if rtol is None:
                rtol = self.__class__.fiducial_rtol
        if atol is None:
            atol = self.fiducial_atol
            if atol is None:
                atol = self.__class__.fiducial_atol

        if fiducial is not None:
            if callable(fiducial):
                fiducial = fiducial(w=self._fiducial_w_set(rtol))

            self_response = self.fresponse(**fiducial.domain_kw())
            if fiducial.tf is not None:
                np.testing.assert_allclose(
                    self_response.tf,
                    fiducial.tf,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                )
            else:
                fiducial = self_response

        else:
            self_response = self.fresponse(w=self._fiducial_w_set(rtol))
            # give it one chance to select better points
            select_bad = (~np.isfinite(self_response.tf)) | (self_response.tf == 0)
            if update and np.any(select_bad):
                if np.all(select_bad):
                    rt_rtol = rtol**0.5
                    domain_w = np.array([rt_rtol])
                    self_response = self.fresponse(w=domain_w)
                else:
                    self_response = self_response[~select_bad]
            fiducial = self_response

        if update:
            self.fiducial = fiducial
            self.fiducial_rtol = rtol
            self.fiducial_atol = atol
        return

    def normalize(self, gain=1,w=None,s=None,f=None):
        r = self.fresponse(w=w, f=f, s=s)
        return self * (gain / r.mag)
