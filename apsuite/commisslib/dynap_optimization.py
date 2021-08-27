"""."""
import time as _time
import numpy as _np
from siriuspy.devices import Tune, TuneCorr, CurrInfoSI, PowerSupplyPU, \
    EVG, EGTriggerPS, PowerSupply, SOFB
from siriuspy.namesys import SiriusPVName as _PVName
from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
from apsuite.optimization import SimulAnneal as _SimulAnneal
from pymodels import si as _si


class BaseProcess:
    """."""

    def __init__(self):
        """."""
        self.devices = dict()
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['pingh'] = PowerSupplyPU(
                PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['egun'] = EGTriggerPS()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)

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
        curr = self.devices['currinfo'].current
        return curr > goal_curr or abs(curr - goal_curr) < tol

    def inject_storage_ring(self, goal_curr):
        """."""
        self.turn_on_injsys()
        while not self._check_current(goal_curr):
            _time.sleep(0.2)
        curr = self.devices['currinfo'].current
        stg = f'Stored: {curr:.3f}/{goal_curr:.3f} mA.'
        print(stg)
        self.turn_off_injsys()

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
            print(f'maximum kick reached, {kick0:.3f} mrad')
            return kick0
        else:
            newkick = kick0 + self.params.kickx_incrate
            print(f'new kick: {newkick:.3f} mrad')
            self.find_max_kick(initial_kick=newkick)


class TuneScanParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.pos_dtunex = 0.01
        self.neg_dtunex = 0.01
        self.pos_dtuney = 0.01
        self.neg_dtuney = 0.01
        self.npts_dtunex = 3
        self.npts_dtuney = 3
        self.wait_tunecorr = 1  # [s]
        self.initial_kickx = -0.500  # [mrad]
        self.kickx_incrate = -0.010  # [mrad]
        self.delta_curr_threshold = 5  # [%]
        self.min_curr_to_inject = 0.5  # [mA]
        self.max_curr = 2.0  # [mA]
        self.nr_orbit_corr = 5
        self.filename = ''

    def __str__(self):
        """."""
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:15s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:15s}: {1:}  {2:s}\n'.format
        stg = ftmp('pos_dtunex', self.pos_dtunex, '')
        stg += ftmp('neg_dtunex', self.neg_dtunex, '')
        stg += ftmp('pos_dtuney', self.pos_dtuney, '')
        stg += ftmp('neg_dtuney', self.neg_dtuney, '')
        stg += dtmp('npts_tunex', self.npts_dtunex, '')
        stg += dtmp('npts_tuney', self.npts_dtuney, '')
        stg += ftmp('initial_kickx', self.initial_kickx, '[mrad]')
        stg += ftmp('kickx_incrate', self.kickx_incrate, '[mrad]')
        stg += ftmp('delta_curr_threshold', self.delta_curr_threshold, '[%]')
        stg += ftmp('min_curr_to_inject', self.min_curr_to_inject, '[mA]')
        stg += ftmp('max_curr', self.max_curr, '[mA]')
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '[s]')
        stg += dtmp('nr_orbit_corr', self.nr_orbit_corr, '')
        stg += stmp('filename', self.filename, '')
        return stg


class TuneScanInjSI(_BaseClass, BaseProcess):
    """."""

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(self)
        BaseProcess.__init__(self)
        self.data = dict()
        self.data['measure'] = dict()
        self.params = TuneScanParams()

        if isonline:
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
            self.devices['tunecorr'].cmd_update_reference()

    def apply_tune_variation(self, dnux, dnuy):
        """."""
        tunecorr = self.devices['tunecorr']
        tunecorr.delta_tunex = dnux
        tunecorr.delta_tuney = dnuy
        tunecorr.cmd_apply_delta()
        _time.sleep(self.params.wait_tunecorr)

    def scan_tunes(self, save=True):
        """."""
        nux = _np.linspace(
            -self.params.neg_dtunex,
            +self.params.pos_dtunex,
            self.params.npts_dtunex)
        nuy = _np.linspace(
            -self.params.neg_dtuney,
            +self.params.pos_dtuney,
            self.params.npts_dtuney)

        tunes = list()
        maxkicks = list()
        meas = dict()
        t0_ = _time.time()
        for valx in nux:
            for valy in nuy:
                curr = self.devices['currinfo'].current
                if curr < self.params.min_curr_to_inject:
                    self.inject_storage_ring(
                        goal_curr=self.params.max_curr)
                self.apply_tune_variation(dnux=valx, dnuy=valy)
                self.devices['sofb'].correct_orbit_manually(
                    self.params.nr_orbit_corr)
                mnux = self.devices['tune'].tunex
                mnuy = self.devices['tune'].tuney
                stg = f'nux={mnux:.4f}, nuy={mnuy:.4f}'
                print(stg)
                maxkick = self.find_max_kick()
                print('='*50)
                tunes.append((mnux, mnuy))
                maxkicks.append(maxkick)

                meas['tunes'] = tunes
                meas['maxkicks'] = maxkicks
                self.data['measure'] = meas
                if save:
                    self.save_data(
                        fname=self.params.filename, overwrite=True)
        tf_ = _time.time()
        print(f'Elapsed time: {(tf_-t0_)/60:.2f}min \n')

    def plot_results(self, fname, title):
        """."""
        tunes = self.data['measure']['tunes']
        kicks = self.data['measure']['maxkicks']
        tunestr = [f'({val[0]:.4f},{val[1]:.4f})' for val in tunes]

        fig = _mplt.figure(figsize=(14, 6))
        gs = _mgs.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(tunestr, kicks)
        ax.set_ylabel('Time [ms]')
        ax.set_xlabel('Current [mA]')
        ax.set_title(title)
        if fname:
            fig.savefig(fname, format='png', dpi=300)
        return fig


class SextScanParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.pos_dstrength = 10  # [%]
        self.neg_dstrength = 10  # [%]
        self.niter = 10
        self.initial_kickx = -0.500  # [mrad]
        self.kickx_incrate = -0.010  # [mrad]
        self.delta_curr_threshold = 5  # [%]
        self.min_curr_to_inject = 0.5  # [mA]
        self.max_curr = 2.0  # [mA]
        self.wait_sextupoles = 3  # [s]
        self.nr_orbit_corr = 5
        self.filename = ''

    def __str__(self):
        """."""
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:15s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:15s}: {1:}  {2:s}\n'.format
        stg = ftmp('pos_dstrength', self.pos_dstrength, '[%]')
        stg += ftmp('neg_dstrength', self.neg_dstrength, '[%]')
        stg += dtmp('niter', self.niter, '')
        stg += ftmp('initial_kickx', self.initial_kickx, '[mrad]')
        stg += ftmp('kickx_incrate', self.kickx_incrate, '[mrad]')
        stg += ftmp('delta_curr_threshold', self.delta_curr_threshold, '[%]')
        stg += ftmp('min_curr_to_inject', self.min_curr_to_inject, '[mA]')
        stg += ftmp('max_curr', self.max_curr, '[mA]')
        stg += ftmp('wait_sextupoles', self.wait_sextupoles, '[s]')
        stg += dtmp('nr_orbit_corr', self.nr_orbit_corr, '')
        stg += stmp('filename', self.filename, '')
        return stg


class SextScanInjSI(_SimulAnneal, _BaseClass, BaseProcess):
    """."""

    def __init__(self, isonline):
        """."""
        _SimulAnneal.__init__(self, save=True)
        _BaseClass.__init__(self)
        BaseProcess.__init__(self)
        self.data = dict()
        self.data['measure'] = dict()
        self.params = SextScanParams()
        pvstr = 'SI-Fam:PS-'
        slist = _si.families.families_sextupoles()
        self.sext_names = [_PVName(pvstr+mag) for mag in slist if '0' in mag]
        if isonline:
            for sext in self.sext_names:
                self.devices[sext] = PowerSupply(sext)
            self.initial_strengths = self.get_initial_strengths()
        self.data['measure']['initial_strengths'] = self.initial_strengths
        self.data['measure']['sext_names'] = self.sext_names

    def initialization(self):
        """."""
        self.niter = self.params.nr_iter
        nknobs = self.sext_names.size
        self.position = _np.zeros(nknobs)
        self.limits_upper = _np.ones(nknobs)*self.params.pos_dstrength
        self.limits_lower = -_np.ones(nknobs)*self.params.neg_dstrength
        self.deltas = (self.limits_upper-self.limits_lower)

    def get_initial_strengths(self):
        """."""
        strens = []
        for sext in self.sext_names:
            strens.append(self.devices[sext].strength)
        return strens

    def apply_strengths(self, strengths):
        """."""
        for idx, sext in enumerate(self.sext_names):
            self.devices[sext].strength = strengths[idx]

    def _change_sextupoles(self):
        """."""
        strens = self.position
        new_strens = self.initial_strengths * (1 + strens/100)
        self.apply_strengths(strengths=new_strens)

    def calc_obj_fun(self):
        """."""
        curr = self.devices['currinfo'].current
        if curr < self.params.min_curr_to_inject:
            self.inject_storage_ring(goal_curr=self.params.max_curr)
        self._change_sextupoles()
        _time.sleep(self.params.wait_sextupoles_change)
        self.devices['sofb'].correct_orbit_manually(self.params.nr_orbit_corr)
        return self.find_max_kick()

    def save_data(self, fname, overwrite):
        """."""
        best_strens = self.initial_strengths*(1 + self.hist_best_positions/100)
        self.data['measure']['hist_best_strengths'] = best_strens
        self.data['measure']['hist_best_maxkicks'] = self.hist_best_objfunc
        return super().save_data(fname, overwrite=overwrite)

    def apply_initial_strengths(self):
        """."""
        self.apply_strengths(strengths=self.initial_strengths)

    def apply_optimized_strengths(self):
        """."""
        best_strens = self.initial_strengths
        best_strens *= (1 + self.hist_best_positions[-1, :]/100)
        self.apply_strengths(strengths=best_strens)

    def print_sextupoles_changes(self, strengths):
        """."""
        stren0 = self.initial_strengths
        for idx, sext in enumerate(self.sext_names):
            diff = strengths[idx]-stren0[idx]
            diff /= stren0[idx]
            stg = f'{sext:s}: {stren0[idx]:.4f} 1/m² -> '
            stg += f'{strengths[idx]:.4f} 1/m² '
            stg += f'({diff*100:.4f} %)'
            print(stg)

    def plot_optimization(self, fname, title):
        """."""
        fig = _mplt.figure(figsize=(14, 6))
        gs = _mgs.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.data['measure']['hist_best_maxkicks'])
        ax.set_ylabel('H. Pinger kick [mrad]')
        ax.set_xlabel('iteration')
        ax.set_title(title)
        if fname:
            fig.savefig(fname, format='png', dpi=300)
        return fig
