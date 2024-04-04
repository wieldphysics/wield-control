#!/usr/bin/python

from scipy import signal
import numpy as np
from collections import namedtuple
import re

cds_filter = namedtuple("cds_filter", ["name", "soscoef", "fs", "design"])
cds_filter_bank = namedtuple("cds_filter_bank", ["name", "filters", "headerAndBody"])

# Copy of scipy normalize() for tf2zp with smaller tolerance
def normalize(b, a):
    num, den = b, a

    den = np.atleast_1d(den)
    num = np.atleast_2d(_align_nums(num))

    if den.ndim != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if num.ndim > 2:
        raise ValueError("Numerator polynomial must be rank-1 or" " rank-2 array.")
    if np.all(den == 0):
        raise ValueError("Denominator must have at least on nonzero element.")

    # Trim leading zeros in denominator, leave at least one.
    den = np.trim_zeros(den, "f")

    # Normalize transfer function
    num, den = num / den[0], den / den[0]

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if np.allclose(col, 0, atol=1e-32):  # Here is the smaller tolerance value
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        print(
            "Badly conditioned filter coefficients (numerator): the "
            "results may be meaningless"
        )
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num[0, :]

    return num, den


# Copy of what scipy does - we need this for tf2zp
def _align_nums(nums):
    try:
        # The statement can throw a ValueError if one
        # of the numerators is a single digit and another
        # is array-like e.g. if nums = [5, [1, 2, 3]]
        nums = np.asarray(nums)

        if not np.issubdtype(nums.dtype, np.number):
            raise ValueError("dtype of numerator is non-numeric")

        return nums

    except ValueError:
        nums = [np.atleast_1d(num) for num in nums]
        max_width = max(num.size for num in nums)

        # pre-allocate
        aligned_nums = np.zeros((len(nums), max_width))

        # Create numerators with padded zeros
        for index, num in enumerate(nums):
            aligned_nums[index, -num.size :] = num

        return aligned_nums


# Copy of what scipy.signal.tf2zpk() does. Needed a lower tolerance value inside normalize()
def tf2zp(b, a):
    b, a = normalize(b, a)
    b = (b + 0.0) / a[0]
    a = (a + 0.0) / a[0]
    k = b[0]
    b /= b[0]
    z = np.roots(b)
    p = np.roots(a)
    return z, p, k


#
def sos2zp(sos):
    """Replicate the MATLAB version of sos2zp because the scipy version is not like the MATLAB version

    Parameters
    ----------
    sos : `float`, array-like
        Second order sections

    Returns
    -------
    z : `float`
        Zeros of the ZPK filter
    p : `float`, array-like
    """

    sos = np.atleast_2d(np.asarray(sos))
    n_sections = sos.shape[0]
    z = np.empty(0, np.complex128)
    p = np.empty(0, np.complex128)
    k = 1
    for section in range(n_sections):
        if sos[section, 5] == 0 and sos[section, 2] == 0:
            b = sos[section, 0:2]
            a = sos[section, 3:5]
        else:
            b = sos[section, 0:3]
            a = sos[section, 3:6]
        if b[-1] == 0 and a[-1] == 0:
            b = b[0]
            a = a[0]
        # [zt,pt,kt] = signal.tf2zpk(b,a)
        [zt, pt, kt] = tf2zp(
            b, a
        )  # Use our own tf2zp because the scipy version has a very tight tolerance on coefficients
        z = np.append(z, zt)
        p = np.append(p, pt)
        k *= kt
    return z, p, k


def readFilterSys(filename, bank, module, **kwargs):
    """Read a filter system from a file and return a ZPK filter and pfilt (a structure containing the filter info from the file)

    Parameters
    ----------
    filename : `str`
        Path and filename of FOTON filter file
    bank : `str`
        Name of the filter bank
    module : `int`, array-like
        Module numers ranging from 1 to 10
    data : array-like, optional
        Array containing [filename, filter .. filter]. This is passed as an optional argument in case
        the file has already been read once and you don't want to spend more I/O time re-reading this
        again

    Returns
    -------
    sysd : :obj:`scipy.signal.ZerosPolesGain`
        ZPK object of the requested FOTON filters
    pfilt : array-like
        Data from the FOTON filter file.
        Passing this back to the same function speeds up searching more filters in the file
    """

    if len(module) == 0:
        sysd = signal.ZerosPolesGain([], [], 1, dt=1.0 / 2 ** 14)
        if len(kwargs) > 0 and kwargs["data"]:
            pfilt = kwargs["data"]
        else:
            pfilt = []
    else:
        if len(kwargs) == 0 or not kwargs["data"]:
            pfilt = readFilterFile(filename)
        elif len(kwargs) > 0 and kwargs["data"]:
            pfilt = kwargs["data"]
            if not re.match(filename, pfilt[0]):
                raise ValueError("Wrong filter filename when using optional data input")
        filterbank_index = 1
        while filterbank_index < len(pfilt) and not re.fullmatch(
            bank, pfilt[filterbank_index].name
        ):
            filterbank_index += 1
        if filterbank_index >= len(pfilt):
            raise ValueError(
                "There was no bank with name {} in the filter file {}".format(
                    bank, pfilt[0]
                )
            )
        sysd = signal.ZerosPolesGain(
            [], [], 1, dt=1.0 / pfilt[filterbank_index].filters[module[0] - 1].fs
        )
        for n in range(0, len(module)):
            index = module[n] - 1
            # [zd,pd,kd] = signal.sos2zpk(pfilt[filterbank_index].filters[index].soscoef)  #removed because this because it is not like Matlab
            [z, p, k] = sos2zp(
                pfilt[filterbank_index].filters[index].soscoef
            )  # This follows the Matlab function sos2zp()
            sysd.zeros = np.append(sysd.zeros, z)
            sysd.poles = np.append(sysd.poles, p)
            sysd.gain *= k
    return sysd, pfilt


def readFilterSys_Scipy(filename, bank, module, **kwargs):
    """Read a filter system from a file and return a ZPK filter and pfilt (a structure containing the filter info from the file)

    Parameters
    ----------
    filename : `str`
        Path and filename of FOTON filter file
    bank : `str`
        Name of the filter bank
    module : `int`, array-like
        Module numers ranging from 1 to 10
    data : array-like, optional
        Array containing [filename, filter .. filter]. This is passed as an optional argument in case
        the file has already been read once and you don't want to spend more I/O time re-reading this
        again

    Returns
    -------
    sysd : :obj:`scipy.signal.ZerosPolesGain`
        ZPK object of the requested FOTON filters
    pfilt : array-like
        Data from the FOTON filter file.
        Passing this back to the same function speeds up searching more filters in the file
    """

    if len(module) == 0:
        sysd = signal.ZerosPolesGain([], [], 1, dt=1.0 / 2 ** 14)
        if len(kwargs) > 0 and kwargs["data"]:
            pfilt = kwargs["data"]
        else:
            pfilt = []
    else:
        if len(kwargs) == 0 or not kwargs["data"]:
            pfilt = readFilterFile(filename)
        elif len(kwargs) > 0 and kwargs["data"]:
            pfilt = kwargs["data"]
            if not re.match(filename, pfilt[0]):
                raise ValueError("Wrong filter filename when using optional data input")
        filterbank_index = 1
        while filterbank_index < len(pfilt) and not re.fullmatch(
            bank, pfilt[filterbank_index].name
        ):
            filterbank_index += 1
        if filterbank_index >= len(pfilt):
            raise ValueError(
                "There was no bank with name {} in the filter file {}".format(
                    bank, pfilt[0]
                )
            )
        sysd = signal.ZerosPolesGain(
            [], [], 1, dt=1.0 / pfilt[filterbank_index].filters[module[0] - 1].fs
        )
        for n in range(0, len(module)):
            index = module[n] - 1
            [zd, pd, kd] = signal.sos2zpk(
                pfilt[filterbank_index].filters[index].soscoef
            )  # removed because this because it is not like Matlab
            # [z,p,k] = sos2zp(pfilt[filterbank_index].filters[index].soscoef)  # This follows the Matlab function sos2zp()
            sysd.zeros = np.append(sysd.zeros, zd)
            sysd.poles = np.append(sysd.poles, pd)
            sysd.gain *= kd
    return sysd, pfilt


def readFilterFile(filename):
    """Read a filter file and return all the SOS coefficients info for the filters

    Parameters
    ----------
    filename : `str`
        Path and filename of FOTON filter file

    Returns
    -------
    out : array-like
        Data from the FOTON filter file.
    """
    out = [filename]
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            if (
                re.match("^#", line)
                and len(line) > 2
                and not re.match("^### \w+", line)
            ):
                if re.search("MODULES", line):
                    line_el = line.split(" ")
                    for n in range(2, len(line_el)):
                        new_filter_bank = cds_filter_bank(
                            line_el[n], [cds_filter] * 10, False
                        )
                        for m in range(0, 10):
                            new_filter_bank.filters[m] = cds_filter(
                                "empty", np.array([1, 0, 0, 1, 0, 0]), 16384, "<none>"
                            )
                        out.append(new_filter_bank)
                elif re.search("SAMPLING", line):
                    line_el = line.split(" ")
                    if line_el[2] == "RATE":
                        for n in range(1, len(out)):
                            for m in range(0, 10):
                                out[n].filters[m] = (
                                    out[n].filters[m]._replace(fs=int(line_el[3]))
                                )
                elif re.search("DESIGN", line):
                    line_el = line.split()
                    fname = line_el[2]
                    index = int(line_el[3])
                    design_str = line_el[4]
                    while line_el[len(line_el) - 1] == "\\":
                        line = next(file)
                        line_el = line.split()
                        if len(line_el) == 1:
                            break
                        else:
                            design_str += line_el[1]
                    filterbank_index = 1
                    while not re.fullmatch(fname, out[filterbank_index].name):
                        filterbank_index += 1
                    out[filterbank_index].filters[index] = (
                        out[filterbank_index].filters[index]._replace(design=design_str)
                    )
            elif re.match("^### \w+", line):
                line_el = line.split()
                fname = line_el[1]
                filterbank_index = 1
                while (
                    filterbank_index < len(out) - 1
                    and fname != out[filterbank_index].name
                ):
                    filterbank_index += 1
                if fname == out[filterbank_index].name:
                    out[filterbank_index] = out[filterbank_index]._replace(
                        headerAndBody=True
                    )
            elif len(line.split()) == 12:
                line_el = line.split()
                fname = line_el[0]
                index = int(line_el[1])
                mname = line_el[6]
                gain = float(line_el[7])
                sos_coeff_lines = int(line_el[3])
                soscoeffs = np.ones((sos_coeff_lines, 6))
                for n in range(0, sos_coeff_lines):
                    if n == 0:
                        soscoeffs[n, 1] = float(line_el[10])
                        soscoeffs[n, 2] = float(line_el[11])
                        soscoeffs[n, 4] = float(line_el[8])
                        soscoeffs[n, 5] = float(line_el[9])
                    else:
                        line = next(file)
                        line_el = line.split()
                        soscoeffs[n, 1] = float(line_el[2])
                        soscoeffs[n, 2] = float(line_el[3])
                        soscoeffs[n, 4] = float(line_el[0])
                        soscoeffs[n, 5] = float(line_el[1])
                soscoeffs[0, :] = np.multiply(
                    soscoeffs[0, :], np.array([gain, gain, gain, 1, 1, 1])
                )
                filterbank_index = 1
                while not re.fullmatch(fname, out[filterbank_index].name):
                    filterbank_index += 1
                out[filterbank_index].filters[index] = (
                    out[filterbank_index].filters[index]._replace(name=mname)
                )
                out[filterbank_index].filters[index] = (
                    out[filterbank_index].filters[index]._replace(soscoef=soscoeffs)
                )

    for n in range(1, len(out) - 1):
        if not out[n].headerAndBody:
            raise ValueError(
                "Header contains module {} but does not exist in body".format(
                    out[n].name
                )
            )

    return out
