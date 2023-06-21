#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numbers
import numpy as np
# import warnings

import scipy.linalg

from ..algorithms.statespace.dense import xfer_algorithms
from ..algorithms.statespace.dense import zpk_algorithms
from ..algorithms.statespace.dense import ss_algorithms
from . import ssprint


class BareStateSpace(object):
    """
    State space class to represent MIMO Transfer functions using dense matrix representations

    This class uses raw matrix representations and should not generally be used by users.

    It is used internally by the SISO.SISOStateSpace and MIMO.MIMOStateSpace classes
    """
    def __init__(
        self,
        A, B, C, D, E, *,
        hermitian: bool = True,
        time_symm: bool = False,
        flags={},
        dt=None,
        warn=True,
    ):
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        assert (A.shape[-1] == A.shape[-2])

        if E is not None:
            E = np.asarray(E)
            # if np.all(E == np.eye(E.shape[-1])):
            #     E = None

        if hermitian:
            assert (np.all(A.imag == 0))
            assert (np.all(B.imag == 0))
            assert (np.all(C.imag == 0))
            assert (np.all(D.imag == 0))
            if E is not None:
                assert (np.all(E.imag == 0))

        self.flags = flags

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt
        return

    @classmethod
    def fromD(cls, D):
        return cls(
            A=np.array([[]]).reshape(0, 0),
            B=np.array([[]]).reshape(0, D.shape[-1]),
            C=np.array([[]]).reshape(D.shape[-2], 0),
            D=D,
            E=None,
            hermitian=np.all(D.imag == 0),
            time_symm=True,
            dt=None
        )

    @property
    def ABCDE(self):
        if self.E is None:
            E = np.eye(self.A.shape[-1])
        else:
            E = self.E
        return self.A, self.B, self.C, self.D, E

    @property
    def ABCDe(self):
        return self.A, self.B, self.C, self.D, self.E

    @property
    def Ninputs(self):
        return self.B.shape[-1]

    @property
    def Noutputs(self):
        return self.C.shape[-2]

    @property
    def Nstates(self):
        return self.A.shape[-1]

    @property
    def as_controlLTI(self):
        import control
        A, B, C, D, E = self.ABCDe
        # TODO
        # assert(E is None)
        return control.ss(A, B, C, D)

    @property
    def ABCD(self):
        if self.E is not None:
            raise RuntimeError("Cannot Drop E")
        else:
            assert (np.all(np.eye(self.E.shape[-1]) == self.E))
            self.E = None
        return self.A, self.B, self.C, self.D

    def __iter__(self):
        """
        Represent self like a typical scipy ABCD tuple. This throws away symmetry information
        """
        yield self.A
        yield self.B
        yield self.C
        yield self.D
        if self.E is not None:
            yield self.E

    def time_reversal(self):
        ret = self.__class__(
            A=-self.A,
            B=-self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )
        return ret

    def conjugate(self):
        return self.time_reversal()

    def transpose(self):
        ret = self.__class__(
            A=self.A,
            B=self.C.T,
            C=self.B.T,
            D=self.D.T,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )
        return ret

    def adjoint(self):
        """
        Return the transpose and conjugate (time-reversal) of the system
        """
        ret = self.__class__(
            A=-self.A,
            B=-self.C.T,
            C=self.B.T,
            D=self.D.T,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )
        return ret

    def print_nonzero(self):
        """
        """
        return ssprint.print_dense_nonzero(self)

    _p_vals = None

    @property
    def _p(self):
        """
        Create a raw tuple of poles from direct calculation
        """
        # TODO, not sure this should be included
        if self._p_vals is None:
            p = zpk_algorithms.ss2p(
                A=self.A,
                E=self.E,
                fmt="scipy",
            )
            self._p_vals = p
        return self._p_vals

    def __getitem__(self, key):
        """
        key must be a tuple of a list of row and column elements.

        It can also be tuple of slices
        """
        row, col = key

        ret = self.__class__(
            A=self.A,
            B=self.B[..., :, col],
            C=self.C[..., row, :],
            # double index fixes annoying way that multiple list indices are grouped by numpy
            D=self.D[..., row, :][..., :, col],
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )
        return ret

    def fresponse_raw(
            self,
            *,
            f=None,
            w=None,
            s=None,
            z=None,
            use_laub=True,
            **kwargs
    ):
        # TODO fix this import
        from ..SISO import util
        domain = util.build_sorz(
            f=f,
            w=w,
            s=s,
            z=z,
            dt=self.dt,
        )
        if use_laub:
            self_b = self.balanceA()
            return xfer_algorithms.ss2response_laub(
                A=self_b.A,
                B=self_b.B,
                C=self_b.C,
                D=self_b.D,
                E=self_b.E,
                sorz=domain,
                **kwargs
            )
        else:
            return xfer_algorithms.ss2response_mimo(
                A=self.A,
                B=self.B,
                C=self.C,
                D=self.D,
                E=self.E,
                sorz=domain,
                **kwargs
            )

    def balanceBC_svd(self, which):
        """
        This balances gains using the SVD of either B or C.

        It is not a very good technique as far as it has been tested
        """
        E = self.E
        if E is not None and np.all(E == np.eye(E.shape[-1])):
            E = None
        if E is not None:
            raise NotImplementedError("balancing on descriptor systems not implemented (yet)")

        A = self.A
        B = self.B
        C = self.C

        if which == 'B':
            u, s, v = np.linalg.svd(B)
            uc = u[:, len(s):] 
            u = u[:, :len(s)]
            print(s)

            s[:] = s[:]**0.5
            Z = u @ ((1/s).reshape(-1, 1) * u.transpose()) + uc @ uc.transpose()
            Zi = (u @ ((s).reshape(-1, 1) * u.transpose()) + uc @ uc.transpose())
            # Br = u @ v
            Br = Z @ B
            Ar = Z @ A @ Zi
            Cr = C @ Zi

        elif which == 'C':
            u, s, v = np.linalg.svd(C)

            vc = v[len(s):, :]
            v = v[:len(s), :]

            s[:] = s[:]**0.5
            Z = v.transpose() @ ((s).reshape(-1, 1) * v) + vc.transpose() @ vc
            Zi = (v.transpose() @ ((1/s).reshape(-1, 1) * v) + vc.transpose() @ vc)

            # Cr2 = u @ v
            Cr = C @ Zi

            Ar = Z @ A @ Zi
            Br = Z @ B
        else:
            raise RuntimeError("Unknown job")

        return self.__class__(
            Ar,
            Br,
            Cr,
            self.D,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def balanceABC(self, which='A'):
        """
        Uses the slycot balancer tb01id

        https://github.com/python-control/Slycot/blob/master/slycot/transform.py#L25
        """
        E = self.E
        if E is not None and np.all(E == np.eye(E.shape[-1])):
            E = None
        if E is not None:
            raise NotImplementedError("balancing on descriptor systems not implemented (yet)")

        assert(which in ['A', 'B', 'C', 'ABC', 'AB', 'AC', 'N'])

        if which == 'A':
            return self.balanceA()
        elif which == 'AB':
            job = 'B'
        elif which == 'AC':
            job = 'C'
        elif which == 'ABC':
            job = 'A'

        A = self.A
        B = self.B
        C = self.C

        # Order of the A matrix
        n = self.A.shape[0]
        # Number of inputs
        m = self.B.shape[1]
        # Number of outputs
        p = self.C.shape[0]

        from slycot import tb01id
        s_norm, Ar, Br, Cr, scaled = tb01id(
            n, m, p,
            0,  # maxred
            A, B, C,
            job='A'
        )
        return self.__class__(
            Ar,
            Br,
            Cr,
            self.D,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def balanceA(self, permute=True):
        """
        Return a version of this statespace where A has been balanced for numerical stability.

        TODO, use a pencil method to modify/account for E as well.
        """
        if self.A.shape[-2:] == (0, 0):
            return self

        Ascale = self.A.copy()
        # xGEBAL does not remove the diagonals before scaling.
        # not sure M is needed, was in the ARE generalized diagonalizer
        # M = np.abs(SS) + np.abs(SSE)

        def invert_permutation(p):
            """Return an array s with which np.array_equal(arr[p][s], arr) is True.
            The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.

            from https://stackoverflow.com/a/25535723
            """
            s = np.empty_like(p)
            s[p] = np.arange(p.size)
            return s

        Ascale, (sca, P) = scipy.linalg.matrix_balance(
            Ascale,
            separate=1,
            permute=permute,
            overwrite_a=True,
        )
        Pi = invert_permutation(P)
        # do we need to bother?
        if not np.allclose(sca, np.ones_like(sca)):
            scar = np.reciprocal(sca)

            if self.E is not None:
                Escale = self.E.copy()
                elwisescale = sca * scar[:, None]
                Escale *= elwisescale
            else:
                Escale = self.E
            Bscale = scar.reshape(-1, 1) * self.B[..., Pi, :].copy()
            Cscale = sca.reshape(1, -1) * self.C[..., :, P].copy()
        else:
            Ascale = self.A
            Escale = self.E
            Bscale = self.B
            Cscale = self.C

        return self.__class__(
            A=Ascale,
            B=Bscale,
            C=Cscale,
            D=self.D,
            E=Escale,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def balance_and_truncate_unscaled(self, method='sqrt', equil=True, tol1=0, tol2=0):
        """
            To compute a reduced order model (Ar,Br,Cr,Dr) for an original state-space representation (A,B,C,D) by using either the square-root or the balancing-free square-root Singular Perturbation Approximation (SPA) model reduction method for the alpha-stable part of the system.
            - From SLYCOT Documentation for ab09nd

        Args:
            sys (Bunch): Bunch system with mod as attribute
            method (str, optional): Method to use for balancing. 'sqrt': use the square-root SPA method. 'bfsqrt': use the balancing-free square-root SPA method. Defaults to 'sqrt'.
            equil (bool, optional): If True, preliminarily equilibrates the triplet (A,B,C). Defaults to True.
            iod (dict, optional): input/output dictionary. Defaults to None.

        Returns:
            a similar StateSpace
        """    
        # Define if the model is discrete or continuous
        if self.dt == 0 or self.dt == None: # if the model is discrete it will have a non 0 or None dt
            dico = 'C'
        else:
            dico = 'D'

        E = self.E
        if E is not None and np.all(E == np.eye(E.shape[-1])):
            E = None
        if E is not None:
            raise NotImplementedError("balancing on descriptor systems not implemented (yet)")

        # Define which method to use
        if method == 'sqrt':
            job = 'B'
        elif method == 'bfsqrt':
            job = 'N'

        # Define if the model is balanced
        if equil == True:
            equil = 'S'
        else:
            equil = 'N'

        A = self.A
        B = self.B
        C = self.C
        D = self.D

        # Order of the A matrix
        n = self.A.shape[0]
        # Number of inputs
        m = self.B.shape[1]
        # Number of outputs
        p = self.C.shape[0]

        from slycot import ab09nd
        nr, Ar, Br, Cr, Dr, ns, hsv = ab09nd(
            dico, job, equil,
            n, m, p,
            A, B, C, D,
            tol1=tol1, tol2=tol2,
        ) # ,alpha=None,nr=None,tol1=0,tol2=0,ldwork=None)
        # nr: nr is the order of the resulting reduced order model
        # ns: The dimension of the alpha-unstable subsystem

        # if the reduced model has fever states than the original model tell the user
        #if nr < n:
            #print('The equalized gain model has ' + str(n-nr) + ' fewer state(s) than the original model')
        return self.__class__(
            Ar,
            Br,
            Cr,
            Dr,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def balance_and_truncate(self, rescale_io=True, **kwargs):
        s = self

        if rescale_io:
            Bscale = np.sum(abs(s.B)**2, axis=-2)**0.5
            # print(self.B.shape)
            # print("Bscale", Bscale)
            Cscale = np.sum(abs(s.C)**2, axis=-1)**0.5
            # print(self.C.shape)
            # print("Cscale", Cscale)

            s = self.__class__(
                A=s.A,
                B=s.B / Bscale.reshape(1, -1),
                C=s.C / Cscale.reshape(-1, 1),
                D=s.D,
                E=s.E,
                hermitian=s.hermitian,
                time_symm=s.time_symm,
                dt=s.dt,
            )
        
        s = s.balance_and_truncate_unscaled(**kwargs)

        if rescale_io:
            s = self.__class__(
                A=s.A,
                B=s.B * Bscale.reshape(1, -1),
                C=s.C * Cscale.reshape(-1, 1),
                D=s.D,
                E=s.E,
                hermitian=s.hermitian,
                time_symm=s.time_symm,
                dt=s.dt,
            )
        return s

    def minreal(self, job='minimal', scale=True, tol=None):
        """
        Calculate a minimal realization, removes unobservable and
        uncontrollable states

        Originally from python-control!
        """
        if job == 'both' or job == 'minimal':
            jobchar = 'M'
        elif job == 'observable':
            jobchar = 'O'
        elif job == 'controllable' or job == 'reachable':
            jobchar = 'C'
        else:
            raise RuntimeError("Unrecognized ")
        if scale:
            scalechar = 'S'
        else:
            scalechar = 'N'
        if self.Nstates:
            E = self.E
            if E is not None and np.all(E == np.eye(E.shape[-1])):
                E = None
            if E is not None:
                raise NotImplementedError("Minreal on descriptor systems not implemented (yet)")

            if tol is None:
                tol = 1e-16

            from slycot import tb01pd
            B = np.empty((self.Nstates, max(self.Ninputs, self.Noutputs)))
            B[:, :self.Ninputs] = self.B
            C = np.empty((max(self.Noutputs, self.Ninputs), self.Nstates))
            C[:self.Noutputs, :] = self.C
            A = self.A.copy()
            nr = self.Nstates
            A, B, C, nr = tb01pd(
                self.Nstates,
                self.Ninputs,
                self.Noutputs,
                self.A.copy(),
                B,
                C,
                tol=tol,
                job=jobchar,
                equil=scalechar,
            )

            return self.__class__(
                A[:nr, :nr],
                B[:nr, :self.Ninputs],
                C[:self.Noutputs, :nr],
                self.D,
                None,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return self

    def minreal_rescaled(self, job='minimal', scale=True, tol=None):
        """
        Apply a rescaling to the B and C matrix so that each Col, Row respectively, has norm 1.
        This improves the scaling expected of tol for inputs and outputs with greatly differing scales.
        """
        do_controller = False
        do_observer = False
        if job == 'minimal':
            do_controller = True
            do_observer = True
        elif job == 'observable':
            do_observer = True
        elif job == 'controller':
            do_observer = True
        else:
            raise RuntimeError("Unknown job")

        s = self
        Bscale = np.sum(abs(s.B)**2, axis=-2)**0.5
        # print(self.B.shape)
        # print("Bscale", Bscale)
        Cscale = np.sum(abs(s.C)**2, axis=-1)**0.5
        # print(self.C.shape)
        # print("Cscale", Cscale)

        s = self.__class__(
            A=s.A,
            B=s.B / Bscale.reshape(1, -1),
            C=s.C / Cscale.reshape(-1, 1),
            D=s.D,
            E=s.E,
            hermitian=s.hermitian,
            time_symm=s.time_symm,
            dt=s.dt,
        )
        
        if do_controller:
            # s = s.balanceBC_svd(which='B')
            if scale:
                s = s.balanceABC(which='AB')
            s = s.minreal(job='controllable', scale=False, tol=tol)
            pass

        if do_observer:
            # s = self.balanceBC_svd(which='C')
            if scale:
                s = s.balanceABC(which='AC')
            s = s.minreal(job='observable', scale=False, tol=tol)
            pass

        s = self.__class__(
            A=s.A,
            B=s.B * Bscale.reshape(1, -1),
            C=s.C * Cscale.reshape(-1, 1),
            D=s.D,
            E=s.E,
            hermitian=s.hermitian,
            time_symm=s.time_symm,
            dt=s.dt,
        )
        return s

    def minreal_observable_split(self, nc, tol=0.0):
        """
        Calculate a minimal observable realization on an nc-sized subset of the outputs.
        Then propagate the transformations into the full sized view.

        This allows one to create reduced systems with the full set of inputs and outputs but at reduced order
        """

        if self.Nstates:
            if self.E is not None:
                raise NotImplementedError("Minreal on descriptor systems not implemented (yet)")

            from slycot import tb01pd

            # generate a complete set of inputs
            BZ = np.eye(self.Nstates)

            # use only a subset of C
            C = np.empty((max(nc, self.Ninputs), self.Nstates))
            C[:nc, :] = self.C[:nc, :]

            A, BZ, C, nr = tb01pd(
                self.Nstates,
                self.Nstates,
                nc,
                self.A,
                BZ,
                C,
                tol=tol,
                job='O',
                equil='N',
            )
            BZ = BZ[:nr, :self.Nstates]

            return self.__class__(
                A[:nr, :nr],
                BZ @ self.B,
                self.C @ BZ.T,
                self.D,
                None,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return self

    def minreal_controllable_split(self, nb, scale=False, tol=0.0):
        """
        Calculate a minimal observable realization on an nc-sized subset of the outputs.
        Then propagate the transformations into the full sized view.

        This allows one to create reduced systems with the full set of inputs and outputs but at reduced order

        TODO: Untested
        """
        if self.Nstates:
            if self.E is not None:
                raise NotImplementedError("Minreal on descriptor systems not implemented (yet)")

            from slycot import tb01pd

            # use only a subset of B
            B = np.empty((self.Nstates, max(nb, self.Noutputs)))
            B[:, :nb] = self.B[:, :nb]

            # generate a complete set of outputs
            CZ = np.eye(self.Nstates)

            A, B, CZ, nr = tb01pd(
                self.Nstates,
                nb,
                self.Nstates,
                self.A,
                B,
                CZ,
                tol=tol,
                job='C',
                equil='N',
            )
            CZ = CZ[:self.Nstates, :nr]

            return self.__class__(
                A[:nr, :nr],
                CZ.T @ self.B,
                self.C @ CZ,
                self.D,
                None,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return self

    def L2_norm(self, mode='L2', scale=True, tol=1e-10):
        """
        Using slycot ab13dd
        Return objects:
           The L2 or H2 norm of the system

        Does not work with descriptor systems
        """

        if self.dt == 0 or self.dt == None: # if the model is discrete it will have a non 0 or None dt
            dico = 'C'
        else:
            dico = 'D'

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        # Order of the A matrix
        n = self.A.shape[0]
        # Number of inputs
        m = self.B.shape[1]
        # Number of outputs
        p = self.C.shape[0]

        if E is None:
            E = np.eye(self.A.shape[-1])
            pass
        elif E is not None and np.all(E == np.eye(E.shape[-1])):
            pass
        else:
            raise RuntimeError("Does not work on descriptor systems")

        if mode == 'L2':
            jobn = 'L'
        elif mode == 'H2':
            jobn = 'H'
        else:
            raise RuntimeError("Unrecognized mode")

        from slycot import ab13bd
        L2 = ab13bd(
            dico,
            jobn,
            n, m, p,
            A, B, C, D,
            tol=tol
        )
        return L2

    def Linf_norm(self, scale=True, tol=1e-10):
        """
        Using slycot ab13dd
        Return objects:
                gpeak : float
                        The L-infinity norm of the system, i.e., the peak gain
                        of the frequency response (as measured by the largest
                        singular value in the MIMO case).
                fpeak : float
                        The frequency where the gain of the frequency response
                        achieves its peak value gpeak, i.e.,

                        || G ( j*fpeak ) || = gpeak ,  if dico = 'C', or

                                j*fpeak
                        || G ( e       ) || = gpeak ,  if dico = 'D'.

        Works with descriptor systems
        """

        if self.dt == 0 or self.dt == None: # if the model is discrete it will have a non 0 or None dt
            dico = 'C'
        else:
            dico = 'D'

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        # Order of the A matrix
        n = self.A.shape[0]
        # Number of inputs
        m = self.B.shape[1]
        # Number of outputs
        p = self.C.shape[0]

        if E is None:
            E = np.eye(self.A.shape[-1])
            jobe = 'I'
        elif E is not None and np.all(E == np.eye(E.shape[-1])):
            jobe = 'I'
        else:
            jobe = 'G'

        if scale:
            equil = 'S'
        else:
            equil = 'N'

        if np.any(D):
            jobd = 'D'
        else:
            jobd = 'Z'

        from slycot import ab13dd
        gpeak, omega_peak = ab13dd(
            dico,
            jobe,
            equil,
            jobd,
            n, m, p,
            A, E, B, C, D,
            tol=tol
        )
        return gpeak, omega_peak / (2 * np.pi)

    def feedbackD(self, D):
        """
        Feedback linkage for a single statespace

        connections_rowcol is a list of row, col pairs
        gain is the connection gain to apply
        """

        fbD = D
        clD = np.linalg.solve(np.eye(self.D.shape[-1]) - fbD @ self.D, fbD)

        if self.dt is not None:
            raise NotImplementedError("feedback not yet implemented in discrete time")

        A = self.A + self.B @ clD @ self.C
        B = self.B + self.B @ clD @ self.D
        C = self.C + self.D @ clD @ self.C
        D = self.D + self.D @ clD @ self.D

        return self.__class__(
            A,
            B,
            C,
            D,
            self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def __matmul__(self, other):
        """
        """
        if isinstance(other, BareStateSpace):
            # currently need to do some checking about the inputs
            # and the outputs
            #return NotImplemented

            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            assert(self.dt == other.dt)
            ABCDE = ss_algorithms.chain([self.ABCDE, other.ABCDE])
            return self.__class__(
                A=ABCDE.A,
                B=ABCDE.B,
                C=ABCDE.C,
                D=ABCDE.D,
                E=ABCDE.E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        elif isinstance(other, np.ndarray):
            # assume it is a pure D matrix.
            return self.__class__(
                A=self.A,
                B=self.B @ other,
                C=self.C,
                D=self.D @ other,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        """
        # the case that it is also a BareStateSpace should be covered by the non-reversed __matmul__ above
        if isinstance(other, np.ndarray):
            # note! this probably is never called, as numpy uses __array_ufunc__ instead
            # assume it is a pure D matrix.
            return self.__class__(
                A=self.A,
                B=self.B,
                C=other @ self.C,
                D=other @ self.D,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        print("BareStateSpace NP __array_function__", func)
        # TODO, add implementations
        # see https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # TODO, add implementations
        # see https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
        if ufunc is np.matmul and method == '__call__':
            other, sself = inputs
            assert (sself is self)
            assert (len(kwargs) == 0)
            return self.__class__(
                A=self.A,
                B=self.B,
                C=other @ self.C,
                D=other @ self.D,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            print("BareStateSpace NP __array_ufunc__", ufunc, method)
            return NotImplemented

    def is_square(self):
        return self.D.shape[-1] == self.D.shape[-2]

    def square_size(self):
        assert(self.is_square())
        return self.D.shape[-1]

    def inv(self):
        assert(self.is_square)
        ABCDE = ss_algorithms.inverse_DSS(*self.ABCDE)
        return self.__class__(
            A=ABCDE.A,
            B=ABCDE.B,
            C=ABCDE.C,
            D=ABCDE.D,
            E=ABCDE.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B * other,
                C=self.C,
                D=self.D * other,
                E=self.E,
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
                A=self.A,
                B=self.B,
                C=other * self.C,
                D=other * self.D,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B,
                C=self.C / other,
                D=self.D / other,
                E=self.E
                ,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
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
            other = _number2D_like(self, other)
            # convert to statespace form
            knownSS = True

        if knownSS or isinstance(other, BareStateSpace):
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm

            A, E = joinAE(self, other)
            assert(self.dt == other.dt)

            return self.__class__(
                A=A,
                B=np.block([
                    [self.B],
                    [other.B]
                ]),
                C=np.block([[self.C, other.C]]),
                D=self.D + other.D,
                E=E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        return NotImplemented

    def __radd__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            other = _number2D_like(self, other)
            # convert to statespace form
            return other + self

        return NotImplemented

    def __sub__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            other = _number2D_like(self, other)
            # convert to statespace form
            knownSS = True

        if knownSS or isinstance(other, BareStateSpace):
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm

            A, E = joinAE(self, other)

            return self.__class__(
                A=A,
                B=np.block([
                    [self.B],
                    [-other.B]
                ]),
                C=np.block([[self.C, other.C]]),
                D=self.D - other.D,
                E=E,
                hermitian=hermitian,
                time_symm=time_symm,
            )
        return NotImplemented

    def __rsub__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            # convert to statespace form
            other = _number2D_like(self, other)
            knownSS = True

        if knownSS or isinstance(other, BareStateSpace):
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm

            A, E = joinAE(self, other)

            return self.__class__(
                A=A,
                B=np.block([
                    [-self.B],
                    [other.B]
                ]),
                C=np.block([[self.C, other.C]]),
                D=other.D - self.D,
                E=E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        return NotImplemented

    def __neg__(self):
        """
        """
        return self.__class__(
            A=self.A,
            B=self.B,
            C=-self.C,
            D=-self.D,
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def __pos__(self):
        """
        """
        return self


class BareStateSpaceUser(object):
    def __init__(self, *, ss):
        self.ss = ss

    @property
    def A(self):
        return self.ss.A

    @property
    def B(self):
        return self.ss.B

    @property
    def C(self):
        return self.ss.C

    @property
    def D(self):
        return self.ss.D

    @property
    def E(self):
        return self.ss.E

    @property
    def ABCDE(self):
        return self.ss.ABCDE

    @property
    def ABCDe(self):
        return self.ss.ABCDe

    @property
    def as_controlLTI(self):
        return self.ss.as_controlLTI

    @property
    def dt(self):
        return self.ss.dt

    @property
    def hermitian(self):
        return self.ss.hermitian

    @hermitian.setter
    def hermitian(self, value):
        self.ss.hermitian = value

    @property
    def time_symm(self):
        return self.ss.time_symm

    @time_symm.setter
    def time_symm(self, value):
        self.ss.time_symm = value

    @property
    def structure_flags(self):
        return self.ss.flags

    @property
    def ABCD(self):
        return self.ss.ABCD

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        return iter(self.ss)

    def print_nonzero(self):
        """
        """
        return ssprint.print_dense_nonzero(self.ss)

    def minreal(self, job='minimal', scale_io=True, scale=True, tol=None):
        if scale_io:
            return self.__build_similar__(
                ss=self.ss.minreal_rescaled(
                    job=job,
                    tol=tol,
                ),
            )
        else:
            return self.__build_similar__(
                ss=self.ss.minreal(
                    job=job,
                    scale=scale,
                    tol=tol,
                ),
            )

    def balance_and_truncate(self, **kwargs):
        return self.__build_similar__(
            ss=self.ss.balance_and_truncate(**kwargs),
        )

    def balance(self, **kwargs):
        return self.__build_similar__(
            ss=self.ss.balanceA(**kwargs),
        )


def joinAE(s, o):
    """
    Perform a join on the A and E matrices for two statespaces.
    This is used for the binary add and sub operations
    """
    blU = np.zeros((s.A.shape[-2], o.A.shape[-1]))
    blL = np.zeros((o.A.shape[-2], s.A.shape[-1]))

    if s.E is None and o.E is None:
        E = None
    else:
        if s.E is None:
            sE = np.eye(s.A.shape[-2])
            oE = o.E
        elif o.E is None:
            sE = s.E
            oE = np.eye(o.A.shape[-2])
        else:
            sE = s.E
            oE = o.E
        E = np.block([
            [sE,  blU],
            [blL, oE]
        ])

    A = np.block([
        [s.A,  blU],
        [blL, o.A]
    ])
    return A, E


def _number2D_like(self, other):
    """
    Converts a number or 2D array into a pure-D form of statespace. Essentially like broadcasting
    """
    other = np.asarray(other)
    assert (self.is_square())
    size = self.square_size()
    other_D = np.eye(size) * other
    other = self.__class__(
        A=np.array([[]]).reshape(0, 0),
        B=np.array([[]]).reshape(0, size),
        C=np.array([[]]).reshape(size, 0),
        D=other_D,
        hermitian=(other.imag == 0),
        time_symm=True,
        dt=self.dt
    )
    return other
