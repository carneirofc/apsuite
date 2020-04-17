#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from siriuspy.epics import PV
from siriuspy.devices import DCCT, SOFB

from apsuite.optimization import PSO, SimulAnneal
from apsuite.commissioning_scripts.base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.deltas = {'Corrs': 50, 'InjSept': 0, 'InjKckr': 0}
        self.niter = 20
        self.wait_change = 5
        self.nrpulses = 5
        self.freq_pulses = 2
        self.timeout_sum = 30
        self.wait_time = 2
        self.last_sum_idx = 50

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
        st += '{0:30s}= {1:9.3f}\n'.format(
            'SOFB timeout [s]', self.timeout_sum)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'last bpm to get sum signal', self.last_sum_idx)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'wait time to measure [s]', self.wait_time)
        st += '{0:30s}= {1:9d}\n'.format(
            'number of iterations', self.niter)
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
        super().__init__(Params())
        self.reference = []
        self.eyes = []
        self.hands = []
        self.f_init = 0
        self.params = Params()
        self.sofb = SOFB(SOFB.DEVICES.BO)
        self.dcct = DCCT(DCCT.DEVICES.BO)
        self.quads = Quads()
        self.corrs = Corrs()
        self.kckr = Kicker()
        self.sept = Septum()
        PSO.__init__(self, save=save)
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

    def initialization(self):
        """."""
        self.niter = self.params.niter
        self.nr_turns = self.params.nturns
        self.nr_bpm = self.params.nbpm
        self.bpm_idx = self.nr_bpm + 50 * (self.nr_turns - 1)

        self.get_pvs()

        while True:
            if self.check_connect():
                break

        self.sofb.nr_points = self.params.nbuffer

        quad_lim = np.ones(len(self.quads.sp)) * self.params.deltas['Quads']
        corr_lim = np.ones(len(self.corrs.sp)) * self.params.deltas['Corrs']
        sept_lim = np.array([self.params.deltas['InjSept']])
        kckr_lim = np.array([self.params.deltas['InjKckr']])

        up = np.concatenate((quad_lim, corr_lim, sept_lim, kckr_lim))
        down = -1 * up
        self.set_limits(upper=up, lower=down)

        self.dcct.cmd_turn_off(self.params.dcct_timeout)
        self.dcct.nrsamples = self.params.dcct_nrsamples
        self.dcct.period = self.params.dcct_period
        self.dcct.cmd_turn_on(self.params.dcct_timeout)

        self.reference = np.array([h.value for h in self.hands])
        # self.reset_wait_buffer()
        self.init_obj_func()
        self.data['fig_init'].append(self.f_init)

    def set_hands_eyes(self):
        """."""
        self.eyes = self.devices['sofb'].sum[:self.params.last_sum_idx]
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
            if hand.name.dev not in {'CH', 'CV'}:
                hand.voltage = change[k]
            else:
                hand.kick = change[k]

    def wait(self, timeout=10):
        """."""
        self.sofb.cmd_reset()
        self.sofb.wait_buffer()

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
            'injsept': Septum('TB-04:PU-InjSept'),
            'injkckr': Kicker('BO-01D:PU-InjKckr'),
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
        self.params = Params()
        self.dcct = DCCT(DCCT.DEVICES.BO)
        self.sofb = SOFB(SOFB.DEVICES.BO)
        self.quads = Quads()
        self.corrs = Corrs()
        self.kckr = Kicker()
        self.sept = Septum()
        SimulAnneal.__init__(self, save=save)

    def initialization(self):
        """."""
        self.niter = self.params.niter
        self.set_hands_eyes()
        self.devices['sofb'].nr_points = self.params.nrpulses

        corr_lim = np.ones(3) * self.params.deltas['Corrs']
        sept_lim = [self.params.deltas['InjSept']]
        kckr_lim = [self.params.deltas['InjKckr']]

        up_lim = np.concatenate((corr_lim, sept_lim, kckr_lim))
        down_lim = -1 * up_lim
        ref = []
        for hand in self.hands:
            if hand.name.dev not in {'CH', 'CV'}:
                ref.append(hand.voltage)
            else:
                ref.append(hand.kick)
        self.reference = np.array(ref)
        self.set_deltas(dmax=up_lim-down_lim)
        up_lim += self.reference
        down_lim += self.reference
        self.set_limits(upper=up_lim, lower=down_lim)

        self.data['reference'].append(self.reference)
        self.position = self.reference
        self.init_obj_func()
        self.data['fig_init'].append(self.f_init)

    def set_hands_eyes(self):
        """."""
        self.eyes = self.devices['sofb'].sum[:self.params.last_sum_idx]
        self.hands = []
        self.hands.append(self.devices['ch-1'])
        self.hands.append(self.devices['cv-1'])
        self.hands.append(self.devices['cv-2'])
        self.hands.append(self.devices['injsept'])
        self.hands.append(self.devices['injkckr'])

    def get_change(self):
        """."""
        return self.position

    def set_change(self, change):
        """."""
        for k, hand in enumerate(self.hands):
            if hand.name.dev not in {'CH', 'CV'}:
                hand.voltage = change[k]
            else:
                hand.kick = change[k]

    def wait(self, timeout=10):
        """."""
        self.sofb.cmd_reset()
        self.sofb.wait_buffer()

    def init_obj_func(self):
        """."""
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_sum)
        self.f_init = -np.mean(self.eyes)

    def calc_obj_fun(self):
        """."""
        self.set_change(self.get_change())
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_sum)
        posepoch = self.position
        f_out = -np.mean(self.eyes)
        self.data['pos_epoch'].append(posepoch)
        self.data['fig_epoch'].append(f_out)
        return f_out
