"""."""
import time as _time
import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

from siriuspy.devices import Tune, TuneCorr, CurrInfoSI, PowerSupplyPU, \
    EVG, EGTriggerPS, PowerSupply, SOFB
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.search import PSSearch as _PSSearch

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from ..optimization import SimulAnneal as _SimulAnneal


class BaseProcess:
    """."""

    DEFAULT_CURR_TOL = 0.01  # [mA]
    INJ_PERIOD = 0.5  # [s]

    def __init__(self):
        """."""
        self.devices = dict()
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['pingh'] = PowerSupplyPU(
                PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['egun'] = EGTriggerPS()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.params = None  # used in self.find_max_kick but set in subclasses

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

    def check_inj_curr(self, curr_min, curr_max, curr_tol=DEFAULT_CURR_TOL):
        """."""
        # current current value
        curr = self.devices['currinfo'].current

        # inject if current value is below minimum
        if curr < curr_min:
            self.inject_storage_ring(curr_max, curr_tol)

    def inject_storage_ring(self, curr_goal, curr_tol=DEFAULT_CURR_TOL):
        """."""
        ftmp = 'Avg. eff.: {:+6.4f} mA / pulse'.format

        print('Injecting...')
        self.turn_on_injsys()

        t0_ = _time.time()
        curr0 = self.devices['currinfo'].current
        while True:
            statusok, curr = self._check_current(curr_goal, curr_tol)
            if statusok:
                break
            # sleep and print avg efficiency
            _time.sleep(0.4)
            nr_pulses = (_time.time() - t0_) / BaseProcess.INJ_PERIOD
            eff = (curr - curr0) / nr_pulses
            print(ftmp(eff))

        self.turn_off_injsys()
        curr = self.devices['currinfo'].current
        print(f'Stored: {curr:.3f}/{curr_goal:.3f} mA.')

    def find_max_kick(self, kick_initial=None):
        """."""
        evg = self.devices['evg']
        pingh = self.devices['pingh']
        cinfo = self.devices['currinfo']

        # set trial kick
        kick0 = kick_initial or self.params.kickx_initial
        pingh.strength = kick0

        # kick PingH and register current loss
        curr0 = cinfo.current
        self.turn_off_injsys()
        pingh.cmd_turn_on_pulse()
        evg.cmd_turn_on_injection()
        _time.sleep(1)
        currf = cinfo.current
        currd = (currf - curr0) / curr0 * 100
        print(f'{currd:.2f} % lost with {kick0:.3f} mrad')

        # if current loss within threshold, keep increasing kick amplitude
        if abs(currd) > self.params.curr_var_threshold:
            print(f'maximum kick reached, {kick0:.3f} mrad')
            return kick0
        else:
            newkick = kick0 + self.params.kickx_incrate
            print(f'new kick: {newkick:.3f} mrad')
            self.find_max_kick(kick_initial=newkick)

    def _check_current(self, curr_goal, curr_tol=DEFAULT_CURR_TOL):
        curr = self.devices['currinfo'].current
        statusok = curr > curr_goal or abs(curr - curr_goal) < curr_tol
        return statusok, curr


class TuneScanParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.dtunex_pos = 0.01
        self.dtunex_neg = 0.01
        self.dtuney_pos = 0.01
        self.dtuney_neg = 0.01
        self.dtunex_npts = 3
        self.dtuney_npts = 3
        self.wait_tunecorr = 1  # [s]
        self.kickx_initial = -0.500  # [mrad]
        self.kickx_incrate = -0.010  # [mrad]
        self.curr_var_threshold = 5  # [%]
        self.curr_min = 0.5  # [mA]
        self.curr_max = 2.0  # [mA]
        self.nr_orbit_corr = 5
        self.filename = ''

    def __str__(self):
        """."""
        dtmp = '{0:15s} = {1:9d}\n'.format
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        stmp = '{0:15s}: {1:}  {2:s}\n'.format
        stg = ftmp('dtunex_pos', self.dtunex_pos, '')
        stg += ftmp('dtunex_neg', self.dtunex_neg, '')
        stg += ftmp('dtuney_pos', self.dtuney_pos, '')
        stg += ftmp('dtuney_neg', self.dtuney_neg, '')
        stg += dtmp('dtunex_npts', self.dtunex_npts)
        stg += dtmp('dtuney_npts', self.dtuney_npts)
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '[s]')
        stg += ftmp('kickx_initial', self.kickx_initial, '[mrad]')
        stg += ftmp('kickx_incrate', self.kickx_incrate, '[mrad]')
        stg += ftmp('curr_var_threshold', self.curr_var_threshold, '[%]')
        stg += ftmp('curr_min', self.curr_min, '[mA]')
        stg += ftmp('curr_max', self.curr_max, '[mA]')
        stg += dtmp('nr_orbit_corr', self.nr_orbit_corr)
        stg += stmp('filename', self.filename, '')
        return stg


class TuneScanInjSI(_BaseClass, BaseProcess):
    """."""

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(self)
        BaseProcess.__init__(self)
        self.data = dict(measure=dict())
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
        parms, lspc = self.params, _np.linspace
        dnuxv = lspc(-parms.dtunex_neg, +parms.dtunex_pos, parms.dtunex_npts)
        dnuyv = lspc(-parms.dtuney_neg, +parms.dtuney_pos, parms.dtuney_npts)

        t0_ = _time.time()
        self.data['time'] = t0_
        self.data['measure'] = dict(tunes=[], maxkicks=[])
        for dnux in dnuxv:
            for dnuy in dnuyv:
                # measure max kick for (dnux, dnuy), add data and save it
                self._measure_max_kick(self.data['measure'], dnux, dnuy)
                if save:
                    self.save_data(
                        fname=parms.filename, overwrite=True)
        tf_ = _time.time()
        print(f'Elapsed time: {(tf_ - t0_) / 60:.2f} min \n')

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

    def _measure_max_kick(self, meas, dnux, dnuy):
        parms, devices = self.params, self.devices

        # check if minimal current is satisfied, inject if necessary
        self.check_inj_curr(parms.curr_min, parms.curr_max)

        # apply tune variation and correct orbit
        self.apply_tune_variation(dnux=dnux, dnuy=dnuy)
        devices['sofb'].correct_orbit_manually(parms.nr_orbit_corr)

        # register actual tunes and print info
        mnux = devices['tune'].tunex
        mnuy = devices['tune'].tuney
        stg = f'nux={mnux:.4f}, nuy={mnuy:.4f}'
        print(stg)

        # measure maximum kick and store data
        maxkick = self.find_max_kick()
        print('='*len(stg))
        meas['tunes'].append((mnux, mnuy))
        meas['maxkicks'].append(maxkick)


class SextSearchParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.niter = 10
        self.dstrength_pos = 10  # [%]
        self.dstrength_neg = 10  # [%]
        self.wait_sextupoles = 2  # [s]
        self.kickx_initial = -0.500  # [mrad]
        self.kickx_incrate = -0.010  # [mrad]
        self.curr_var_threshold = 5  # [%]
        self.curr_min = 0.5  # [mA]
        self.curr_max = 2.0  # [mA]
        self.nr_orbit_corr = 5
        self.filename = ''

    def __str__(self):
        """."""
        dtmp = '{0:15s} = {1:9d}\n'.format
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        stmp = '{0:15s}: {1:}  {2:s}\n'.format
        stg = dtmp('niter', self.niter)
        stg += ftmp('dstrength_pos', self.dstrength_pos, '[%]')
        stg += ftmp('dstrength_neg', self.dstrength_neg, '[%]')
        stg += ftmp('wait_sextupoles', self.wait_sextupoles, '[s]')
        stg += ftmp('kickx_initial', self.kickx_initial, '[mrad]')
        stg += ftmp('kickx_incrate', self.kickx_incrate, '[mrad]')
        stg += ftmp('curr_var_threshold', self.curr_var_threshold, '[%]')
        stg += ftmp('curr_min', self.curr_min, '[mA]')
        stg += ftmp('curr_max', self.curr_max, '[mA]')
        stg += dtmp('nr_orbit_corr', self.nr_orbit_corr)
        stg += stmp('filename', self.filename, '')
        return stg


class SextSearchInjSI(_SimulAnneal, _BaseClass, BaseProcess):
    """."""

    def __init__(self, isonline=True):
        """."""
        _SimulAnneal.__init__(self, save=True)
        _BaseClass.__init__(self)
        BaseProcess.__init__(self)
        self.data = dict(measure=dict())
        self.params = SextSearchParams()

        # get achromatic sextupole power supply family names
        psnames = _PSSearch.get_psnames({'sec':'SI', 'dev':'S.*0'})
        self.psnames = [_PVName(psname) for psname in psnames]

        if isonline:
            for psname in self.psnames:
                self.devices[psname] = PowerSupply(psname)
            self.initial_strengths = self.get_initial_strengths()

        self.data['measure']['initial_strengths'] = self.initial_strengths
        self.data['measure']['psnames'] = self.psnames

    def initialization(self):
        """."""
        self.niter = self.params.niter
        nknobs = len(self.psnames)
        self.position = _np.zeros(nknobs)
        self.limits_upper = _np.ones(nknobs)*self.params.dstrength_pos
        self.limits_lower = -_np.ones(nknobs)*self.params.dstrength_neg
        self.deltas = (self.limits_upper-self.limits_lower)

    def get_initial_strengths(self):
        """."""
        strens = []
        for psname in self.psnames:
            strens.append(self.devices[psname].strength)
        return strens

    def apply_strengths(self, strengths):
        """."""
        for psname, strength in zip(self.psnames, strengths):
            self.devices[psname].strength = strength

    def calc_obj_fun(self):
        """."""
        return self._measure_max_kick()

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
        stren0v = self.initial_strengths
        strenfv = strengths
        for psname, stren0, strenf in zip(self.psnames, stren0v, strenfv):
            diff = 100 * (strenf - stren0) / stren0
            stg = f'{psname:s}: {stren0:.4f} 1/m² -> '
            stg += f'{strenf:.4f} 1/m² ({diff:+.4f} %)'
            print(stg)

    def plot_optimization(self, fname, title):
        """."""
        fig = _mplt.figure(figsize=(14, 6))
        gs = _mgs.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.data['measure']['hist_best_maxkicks'])
        ax.set_ylabel('HPinger kick [mrad]')
        ax.set_xlabel('iteration')
        ax.set_title(title)
        if fname:
            fig.savefig(fname, format='png', dpi=300)
        return fig

    def _change_sextupoles(self, sleep=True):
        """."""
        strens = self.position
        new_strens = self.initial_strengths * (1 + strens/100)
        self.apply_strengths(strengths=new_strens)
        if sleep:
            _time.sleep(self.params.wait_sextupoles_change)

    def _measure_max_kick(self):
        parms, devices = self.params, self.devices

        # check if minimal current is satisfied, inject if necessary
        self.check_inj_curr(parms.curr_min, parms.curr_max)

        # apply sextupole variation and correct orbit
        self._change_sextupoles(sleep=True)
        devices['sofb'].correct_orbit_manually(parms.nr_orbit_corr)

        # measure maximum kick and return
        return self.find_max_kick()
