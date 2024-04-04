import numpy as np
from scipy import signal
import scipy.linalg
import matplotlib.pyplot as plt

import control

import scipy.io
from os import path

# import SUS.SUS_indices_connections as SUS_indices_connections

from . import readFilter

from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert

from wield.bunch import Bunch
import pickle
import os


def getControlSS(filename):
    """given a filename of a SUS control system, return the state space representation

    Args:
        filename (str): file name of the Foton filter file

    Returns:
        control.statesp.StateSpace: State space representation of the control system
    """
    FotonFileName = filename
    IFO = filename[0:2]  # site name
    gpstime = int(filename.split("_")[1][0:-4])  # get GPS time from file name
    SUS = filename.split("_")[0][5:8]  # get SUS from file name
    topstage = "M1"
    datestr = '{"%s"}' % tconvert(gpstime)  # convert GPS time to date string

    dampDOFs = ["L", "T", "V", "R", "P", "Y"]  # list of damping DOFs
    DampingFiltDict = {}  # dictionary of damping filters
    for (
        dampdof
    ) in dampDOFs:  # for each damping DOF get the filter and convert to state space
        chanPrefix = "%s:SUS-%s_%s_DAMP" % (IFO, SUS, topstage)  # create channel prefix
        print(
            "Getting switch values and gain from: " + "%s_%s" % (chanPrefix, dampdof)
        )  # print channel prefix
        data_swstat = TimeSeries.get(
            "%s_%s_SWSTAT" % (chanPrefix, dampdof), gpstime, gpstime + 1
        )  # get switch status to tell if the filter was on or off
        data_gain = TimeSeries.get(
            "%s_%s_GAIN" % (chanPrefix, dampdof), gpstime, gpstime + 1
        )  # get gain of filter when filter file was in use

        _, filtList, GainOverride = FiltUtils.sfm_decode(
            int(data_swstat[0].value)
        )  # get the filter list and gain override from the switch status.

        zpksys, filtinfo = readFilter.readFilterSys_Scipy(
            FotonFileName, "%s_%s_DAMP_%s" % (SUS, topstage, dampdof), filtList
        )  # get zpk representation of the filter
        z, p, k = FiltUtils.d2c(
            (zpksys.zeros, zpksys.poles, zpksys.gain), 1 / zpksys.dt
        )  # seperate zeros, poles, and gains

        A, B, C, D = signal.zpk2ss(z, p, k)  # convert to state space
        DampingFiltDict[dampdof] = (
            control.StateSpace(A, B, C, D) * data_gain[0].value * GainOverride
        )  # add state space representation to dictionary

    final_sys = control.append(
        DampingFiltDict["L"],
        DampingFiltDict["T"],
        DampingFiltDict["V"],
        DampingFiltDict["R"],
        DampingFiltDict["P"],
        DampingFiltDict["Y"],
    )  # append all the state space systems
    return final_sys  # return combined state space system


def getHSTSModel():
    """fetches the HSTS model from the HSTS_Model.mat file

    Returns:
        state space: state space model of the HSTS
        dict: dictionary of names of the state space model to their indices
    """
    mat = scipy.io.loadmat(
        path.join(path.split(__file__)[0], "hsts_full.mat")
    )  # loads state space model from matlab file
    #A_undamp = mat["A"] - (1e-4 * np.eye(mat["A"].shape[-1]))  # state matrix
    A_undamp = mat["A"] - (1e-7 * np.eye(mat["A"].shape[-1]))  # state matrix
    B_undamp = mat["B"]  # input matrix
    C_undamp = mat["C"]  # output matrix
    D_undamp = mat["D"]  # feedthrough matrix
    UndampedSUS = (A_undamp, B_undamp, C_undamp, D_undamp)

    UdNames = {
        # Inputs
        "P.gnd.disp.L.in": 0,
        "P.gnd.disp.T.in": 1,
        "P.gnd.disp.V.in": 2,
        "P.gnd.disp.R.in": 5,
        "P.gnd.disp.P.in": 4,
        "P.gnd.disp.Y.in": 3,
        "P.m1.drive.L.in": 6,
        "P.m1.drive.T.in": 7,
        "P.m1.drive.V.in": 8,
        "P.m1.drive.R.in": 11,
        "P.m1.drive.P.in": 10,
        "P.m1.drive.Y.in": 9,
        "P.m2.drive.L.in": 12,
        "P.m2.drive.T.in": 13,
        "P.m2.drive.V.in": 14,
        "P.m2.drive.R.in": 17,
        "P.m2.drive.P.in": 16,
        "P.m2.drive.Y.in": 15,
        "P.m3.drive.L.in": 18,
        "P.m3.drive.T.in": 19,
        "P.m3.drive.V.in": 20,
        "P.m3.drive.R.in": 23,
        "P.m3.drive.P.in": 22,
        "P.m3.drive.Y.in": 21,
        # Outputs
        "P.m1.disp.L.out": 0,
        "P.m1.disp.T.out": 1,
        "P.m1.disp.V.out": 2,
        "P.m1.disp.R.out": 5,
        "P.m1.disp.P.out": 4,
        "P.m1.disp.Y.out": 3,
        "P.m2.disp.L.out": 6,
        "P.m2.disp.T.out": 7,
        "P.m2.disp.V.out": 8,
        "P.m2.disp.R.out": 11,
        "P.m2.disp.P.out": 10,
        "P.m2.disp.Y.out": 9,
        "P.m3.disp.L.out": 12,
        "P.m3.disp.T.out": 13,
        "P.m3.disp.V.out": 14,
        "P.m3.disp.R.out": 17,
        "P.m3.disp.P.out": 16,
        "P.m3.disp.Y.out": 15,
    }
    return UndampedSUS, UdNames


def get_or_load_controller(stage: str = "m1"):
    gpstime = 1336840553  # GPS time of the start of the data
    FotonFileName = (
        "H1SUSSRM_" + str(gpstime) + ".txt"
    )  # File name of the Foton filters

    ctrl_fname = "ctrl_ss_file.pkl"
    if os.path.isfile(ctrl_fname):
        with open(ctrl_fname, "rb") as config_dictionary_file:
            controlSS = pickle.load(config_dictionary_file)
        print("Loaded controlSS from file")
    else:
        controlSS = getControlSS(
            FotonFileName
        )  # get the control system state space representation
        with open(ctrl_fname, "wb") as ctrl_file:
            pickle.dump(controlSS, ctrl_file)
        print("Saved controlSS to file")
    return controlSS


def load_sus_ss():
    UndampedSUS, UndampedSUS_Names = getHSTSModel()  # get the undamped SUS plant model
    controlSS = (
        get_or_load_controller()
    )  # get the control system state space representation

    return Bunch(locals())
