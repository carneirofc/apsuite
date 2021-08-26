"""."""
import time as _time
import numpy as _np
from siriuspy.devices import Tune, TuneCorr, CurrInfoSI, PowerSupplyPU, \
    EVG, EGTriggerPS
from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class InjSIParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.range_delta_tunex = 0.01
        self.range_delta_tuney = 0.01
        self.wait_tunecorr = 1  # [s]
        self.pulse_freq = 2  # [Hz]
        self.initial_kickx = -0.500  # [mrad]
        self.kickx_incrate = -0.005  # [mrad]
        self.delta_curr_threshold = 5  # [%]
        self.minimum_curr_to_inject = 0.5  # [mA]
        self.maximum_curr = 2.0  # [mA]

    def __str__(self):
        """."""
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:15s} = {1:9d}  {2:s}\n'.format
        stg = ftmp('range_delta_tunex', self.range_delta_tunex, '')
        stg += ftmp('range_delta_tuney', self.range_delta_tuney, '')
        stg += ftmp('initial_kickx', self.initial_kickx, '[mrad]')
        stg += ftmp('kickx_incrate', self.kickx_incrate, '[mrad]')
        stg += ftmp('delta_curr_threshold', self.delta_curr_threshold, '[%]')
        stg += ftmp(
            'minimum_curr_to_inject', self.minimum_curr_to_inject, '[mA]')
        stg += ftmp('maximum_curr', self.maximum_curr, '[mA]')
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '[s]')
        stg += ftmp('pulse_freq', self.pulse_freq, '[Hz]')
        return stg


class TuneScanInjSI(_BaseClass):
    """."""

    def __init__(self):
        """."""
        _BaseClass.__init__(self)
        self.devices = dict()
        self.params = InjSIParams()

        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['tunecorr'].cmd_update_reference()
        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['egun'] = EGTriggerPS()

        self.data['measure'] = dict()
        self.data['measure']['tunex'] = []
        self.data['measure']['tuney'] = []
        self.data['measure']['kickx'] = []
        self.data['measure']['delta_curr'] = []

    def turn_on_injsys(self):
        """."""
        self.devices['pingh'].cmd_turn_off_pulse()
        self.devices['egun'].cmd_enable_trigger()
        self.devices['evg'].bucket_list = [1]
        self.devices['evg'].nrpulses = 0
        self.devices['evg'].cmd_update_events()
        _time.sleep(1)
        self.devices['evg'].cmd_turn_on_injection()

    def turn_off_injsys(self):
        """."""
        self.devices['egun'].cmd_disable_trigger()
        self.devices['evg'].cmd_turn_off_injection()
        self.devices['evg'].bucket_list = [1]
        self.devices['evg'].nrpulses = 1
        self.devices['evg'].cmd_update_events()
        _time.sleep(1)

    def _check_current(self, goal_curr, tol=0.01):
        dcct_curr = self.devices['currinfo'].current
        return dcct_curr > goal_curr or abs(dcct_curr - goal_curr) < tol

    def inject_storage_ring(self, goal_curr):
        """."""
        self.turn_on_injsys()
        while not self._check_current(goal_curr):
            _time.sleep(0.2)
        curr = self.devices['currinfo'].current
        stg = f'Stored: {curr:.3f}/{goal_curr:.3f}mA.'
        print(stg)
        self.turn_off_injsys()

    def apply_tune_variation(self, dnux, dnuy):
        """."""
        tunecorr = self.devices['tunecorr']
        tunecorr.delta_tunex = dnux
        tunecorr.delta_tuney = dnuy
        tunecorr.cmd_apply_delta()
        _time.sleep(self.params.wait_tunecorr)

    def find_max_kick(self, initial_kick=None):
        """."""
        pingh = self.devices['pingh']
        inj = self.devices['evg']
        cinfo = self.devices['currinfo']

        kick0 = initial_kick or self.params.initial_kickx
        pingh.strength = kick0

        curr0 = cinfo.current
        self.turn_off_injsys()
        pingh.cmd_turn_on_pulse()
        inj.cmd_turn_on_injection()
        _time.sleep(1)
        curr = cinfo.current
        dcurr = (curr-curr0)/curr0 * 100
        print(f'{dcurr:.2f} % lost with {kick0:.3f} mrad')

        if abs(dcurr) > self.params.delta_curr_threshold:
            print('maximum kick reached!')
            return kick0
        else:
            newkick = kick0 + self.params.kickx_incrate
            print(f'new kick: {newkick:.3f} mrad')
            self.find_max_kick(initial_kick=newkick)

    def scan_tunes(self):
        """."""
        # define grid of tune scanning and for each point find the max. kick
        return None
