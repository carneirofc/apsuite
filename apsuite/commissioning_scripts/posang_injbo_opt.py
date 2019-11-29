#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from siriuspy.epics import PV
from siriuspy.namesys import SiriusPVName

from pymodels.middlelayer.devices import DCCT, SOFB, Septum, Kicker
from apsuite.optimization import PSO, SimulAnneal
from apsuite.commissioning_scripts.base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.deltas = {'Corrs': 500, 'InjSept': 0.5, 'InjKckr': 0.5}
        self.niter = 10
        self.wait_change = 5
        self.nrpulses = 20
        self.freq_pulses = 2
        self.timeout_sum = 30
        self.wait_time = 2

    def __str__(self):
        """."""
        st = '{0:30s}= {1:9.3f}\n'.format(
            'delta corrs [urad]', self.deltas['Corrs'])
        st += '{0:30s}= {1:9.3f}\n'.format(
            'delta injsept [mrad]', self.deltas['InjSept'])
        st += '{0:30s}= {1:9.3f}\n'.format(
            'delta injkckr [mrad]', self.deltas['InjKckr'])
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'pulses freq [Hz]', self.freq_pulses)
        st += '{0:30s}= {1:9.3f}\n'.format('SOFB timeout', self.timeout_sum)
        return st


class Corrs():
    """."""

    def __init__(self, name):
        """."""
        self._name = SiriusPVName(name)
        self._curr_sp = PV(self._name.substitute(propty='Current-SP'))
        self._curr_rb = PV(self._name.substitute(propty='Current-RB'))
        maname = self._name
        if not self._name.dis == 'MA':
            maname = self._name.substitute(dis='MA')
        self._kick_sp = PV(maname.substitute(propty='Kick-SP'))
        self._kick_rb = PV(maname.substitute(propty='Kick-RB'))

    @property
    def name(self):
        """."""
        return self._name

    @property
    def kick(self):
        """."""
        return self._kick_rb.value

    @kick.setter
    def kick(self, value):
        self._kick_sp.value = value

    @property
    def current(self):
        """."""
        return self._curr_rb.value

    @current.setter
    def current(self, value):
        self._curr_sp.value = value

    @property
    def connected(self):
        """."""
        conn = self._curr_sp.connected
        conn &= self._curr_rb.connected
        conn &= self._kick_rb.connected
        conn &= self._kick_sp.connected
        return conn


class PSOInjection(BaseClass, PSO):
    """."""

    def __init__(self, save=False):
        """."""
        super().__init__(Params(), PSO(save=save))
        self.devices = {
            'dcct': DCCT(),
            'sofb': SOFB('BO'),
            'injsept': Septum('TB-04:PM-InjSept'),
            'injkckr': Kicker('BO-01D:PM-InjKckr'),
            'ch-1': Corrs('TB-04:MA-CH-1'),
            'cv-1': Corrs('TB-04:MA-CV-1'),
            'cv-2': Corrs('TB-04:MA-CV-2'),
            }
        self.data = {
            'pos_epoch': [],
            'fig_epoch': [],
            'fig_init': [],
            'reference': [],
            'nswarm': self.nswarm,
            }
        self.reference = []
        self.eyes = []
        self.hands = []
        self.f_init = 0
        # PSO.__init__(self, save=save)

    def initialization(self):
        """."""
        self.niter = self.params.niter
        self.set_hands_eyes()
        self.devices['sofb'].nr_points = self.params.nrpulses

        corr_lim = np.ones(3) * self.params.deltas['Corrs']
        sept_lim = np.array([self.params.deltas['InjSept']])
        kckr_lim = np.array([self.params.deltas['InjKckr']])

        up_lim = np.concatenate((corr_lim, sept_lim, kckr_lim))
        down_lim = -1 * up_lim
        self.set_limits(upper=up_lim, lower=down_lim)
        self.reference = np.array([h.kick for h in self.hands])
        self.data['reference'].append(self.reference)
        self.init_obj_func()
        self.data['fig_init'].append(self.f_init)

    def set_hands_eyes(self):
        """."""
        self.eyes = self.devices['sofb'].sum
        self.hands = []
        self.hands.append(self.devices['ch-1'])
        self.hands.append(self.devices['cv-1'])
        self.hands.append(self.devices['cv-2'])
        self.hands.append(self.devices['injsept'])
        self.hands.append(self.devices['injkckr'])

    def get_change(self, part):
        """."""
        return self.reference + self.position[part, :]

    def set_change(self, change):
        """."""
        for k, hand in enumerate(self.hands):
            hand.kick = change[k]

    def wait(self, timeout=10):
        """."""
        self.devices['sofb'].wait(timeout=timeout)

    def reset(self, wait=0):
        """."""
        if self._stopped.wait(wait):
            return False
        self.devices['sofb'].reset()
        if self._stopped.wait(1):
            return False
        return True

    def init_obj_func(self):
        """."""
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_sum)
        self.f_init = -np.mean(self.eyes)

    def calc_obj_fun(self):
        """."""
        f_out = np.zeros(self.nswarm)
        for part in range(self.nswarm):
            self.set_change(self.get_change(part))
            _time.sleep(self.params.wait_change)
            self.reset(self.params.wait_time)
            self.wait(self.params.timeout_sum)
            f_out[part] = -np.mean(self.eyes)
            print(
                'Particle {:02d}/{:d} | Obj. Func. : {:f}'.format(
                    part+1, self.nswarm, f_out[part]))
            posepoch = self.reference + self.position[part, :]
            self.data['pos_epoch'].append(posepoch)
            self.data['fig_epoch'].append(f_out[part])
        return - f_out


class SAInjection(BaseClass, SimulAnneal):
    """."""

    def __init__(self, save=False):
        """."""
        super().__init__(Params())
        self.devices = {
            'dcct': DCCT(),
            'sofb': SOFB('BO'),
            'injsept': Septum('TB-04:PM-InjSept'),
            'injkckr': Kicker('BO-01D:PM-InjKckr'),
            'ch-1': Corrs('TB-04:MA-CH-1'),
            'cv-1': Corrs('TB-04:MA-CV-1'),
            'cv-2': Corrs('TB-04:MA-CV-2'),
            }
        self.data = {
            'pos_epoch': [],
            'fig_epoch': [],
            'fig_init': [],
            'reference': [],
            }
        self.reference = []
        self.eyes = []
        self.hands = []
        self.f_init = 0
        SimulAnneal.__init__(self, save=save)

    def initialization(self):
        """."""
        self.niter = self.params.niter
        self.set_hands_eyes()
        self.devices['sofb'].nr_points = self.params.nrpulses

        corr_lim = np.ones(3) * self.params.deltas['Corrs']
        sept_lim = np.array([self.params.deltas['InjSept']])
        kckr_lim = np.array([self.params.deltas['InjKckr']])

        up_lim = np.concatenate((corr_lim, sept_lim, kckr_lim))
        down_lim = -1 * up_lim
        self.set_limits(upper=up_lim, lower=down_lim)
        self.set_deltas(dmax=up_lim)

        self.reference = np.array([h.kick for h in self.hands])
        self.data['reference'].append(self.reference)
        self.position = self.reference
        self.init_obj_func()
        self.data['fig_init'].append(self.f_init)

    def set_hands_eyes(self):
        """."""
        self.eyes = self.devices['sofb'].sum
        self.hands = []
        self.hands.append(self.devices['ch-1'])
        self.hands.append(self.devices['cv-1'])
        self.hands.append(self.devices['cv-2'])
        self.hands.append(self.devices['injsept'])
        self.hands.append(self.devices['injkckr'])

    def get_change(self):
        """."""
        return self.reference + self.position

    def set_change(self, change):
        """."""
        for k, hand in self.hands:
            hand.kick = change[k]

    def wait(self, timeout=10):
        """."""
        self.devices['sofb'].wait(timeout=timeout)

    def reset(self, wait=0):
        """."""
        if self._stopped.wait(wait):
            return False
        self.devices['sofb'].reset()
        if self._stopped.wait(1):
            return False
        return True

    def init_obj_func(self):
        """."""
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_sum)
        self.f_init = -np.mean(self.eyes)

    def calc_obj_fun(self):
        """."""
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_sum)
        posepoch = self.reference + self.position
        f_out = -np.mean(self.eyes)
        self.data['pos_epoch'].append(posepoch)
        self.data['fig_epoch'].append(f_out)
        return f_out
