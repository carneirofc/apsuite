#!/usr/bin/env python-sirius
"""."""

import numpy as np

from copy import deepcopy as _dcopy
from pymodels import bo, si
from pymodels import middlelayer as _pml
import pyaccel

from siriuspy.namesys import SiriusPVName
from siriuspy.epics import PV


class DeltaPosAng:

    DIST = 3.196

    def __init__(self, acc):
        if acc == 'SI':
            pyacc = si
            sept_name = 'InjSeptF'
        elif acc == 'BO':
            pyacc = bo
            sept_name = 'InjSept'
        else:
            raise Exception('Set accelerator BO or SI')
        self.accmod = pyacc.create_accelerator()
        self.fam = pyacc.get_family_data(self.accmod)
        self.ch_idx = self.fam['CH']['index']
        self.cv_idx = self.fam['CV']['index']
        self.sept_idx = self.fam[sept_name]['index']
        self.csdata = pyacc.get_control_system_data(self.accmod, self.fam)
        self.acc_csdata = dict()
        for k, v in self.csdata.items():
            self.acc_csdata[k] = v
        if acc == 'SI':
            dipname = SiriusPVName('SI-Fam:MA-B1B2')
            dip1 = 'SI-Fam:MA-B1B2-1'
            dip2 = 'SI-Fam:MA-B1B2-2'
            self.acc_csdata[dipname] = self.acc_csdata[dip1]
            self.acc_csdata.pop(dip1)
            self.acc_csdata.pop(dip2)
        self.acc_elements = {
            n: _pml.get_element(
                n, self.accmod, **v) for n, v in self.acc_csdata.items()}

    def get_kicks(self):
        kicks_x = []
        kicks_y = []
        for k, v in self.acc_elements.items():
            if k.dev == 'CH':
                kicks_x.append(v.strength)
            if k.dev == 'CV':
                kicks_y.append(v.strength)
        return kicks_x + kicks_y

    def set_kicks(self, accmod, kicks):
        mod = _dcopy(accmod)
        for ix, ch in enumerate(self.ch_idx):
            mod[ch[0]].hkick_polynom = kicks[ix]
        for iy, cv in enumerate(self.cv_idx):
            mod[cv[0]].vkick_polynom = kicks[len(self.ch_idx)+iy]
        return mod

    def calc_delta_posang(self, accmod, kick_bef, kick_aft):
        mod_bef = self.set_kicks(accmod, kick_bef)
        mod_aft = self.set_kicks(accmod, kick_aft)
        orbit_bef = pyaccel.tracking.find_orbit4(
            accelerator=mod_bef, indices='open')[:, self.sept_idx[0][0]]
        orbit_aft = pyaccel.tracking.find_orbit4(
            accelerator=mod_aft, indices='open')[:, self.sept_idx[0][0]]
        return orbit_aft - orbit_bef
