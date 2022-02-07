"""."""
import time as _time

import numpy as _np

import pyaccel
from siriuspy.devices import SOFB, LILLRF

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class MeasureDispParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.klystron_delta = -2  # [%]
        self.klystron_wait = 40  # [s]
        self.timeout_orb = 10  # [s]
        self.sofb_nrpoints = 10

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stg = ftmp('klystron_delta', self.klystron_delta, '[%]')
        stg += ftmp('klystron_wait', self.klystron_wait, '[s]')
        stg += ftmp('timeout_orb', self.timeout_orb, '[s]')
        stg += dtmp('sofb_nrpoints', self.sofb_nrpoints, '')
        return stg


class MeasureDispTBBO(_BaseClass):
    """."""

    # KLY2_AMP_TO_ENERGY = [1.098, 66.669]  # old
    # KLY2_AMP_TO_ENERGY = [1.01026423, 71.90322743]  # > 2.5nC
    KLY2_AMP_TO_ENERGY = [0.80518365, 87.56545895]  # < 2.5nC

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, params=MeasureDispParams(), isonline=isonline)
        if self.isonline:
            self.devices['bo_sofb'] = SOFB(SOFB.DEVICES.BO)
            self.devices['tb_sofb'] = SOFB(SOFB.DEVICES.TB)
            self.devices['klystron2'] = LILLRF(LILLRF.DEVICES.LI_KLY2)

    @property
    def energy(self):
        """."""
        engy = _np.polyval(
            self.KLY2_AMP_TO_ENERGY, self.devices['klystron2'].amplitude)
        return engy

    @property
    def trajx(self):
        """."""
        tb_trajx = self.devices['tb_sofb'].trajx
        bo_trajx = self.devices['bo_sofb'].trajx
        return _np.hstack([tb_trajx, bo_trajx])

    @property
    def trajy(self):
        """."""
        tb_trajy = self.devices['tb_sofb'].trajy
        bo_trajy = self.devices['bo_sofb'].trajy
        return _np.hstack([tb_trajy, bo_trajy])

    @property
    def nr_points(self):
        """."""
        return min(
            self.devices['tb_sofb'].nr_points,
            self.devices['bo_sofb'].nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.devices['tb_sofb'].nr_points = int(value)
        self.devices['bo_sofb'].nr_points = int(value)

    def wait(self, timeout=10):
        """."""
        self.devices['tb_sofb'].wait_buffer(timeout=timeout)
        self.devices['bo_sofb'].wait_buffer(timeout=timeout)

    def reset(self, wait=0):
        """."""
        _time.sleep(wait)
        self.devices['tb_sofb'].cmd_reset()
        self.devices['bo_sofb'].cmd_reset()

    def measure_dispersion(self):
        """."""
        self.nr_points = self.params.sofb_nrpoints
        delta = self.params.klystron_delta
        kly2 = self.devices['klystron2']

        print('getting trajectory...')
        self.reset(3)
        self.wait(self.params.timeout_orb)
        traj0 = _np.hstack([self.trajx, self.trajy])
        ene0 = self.energy
        print('ok!')

        amp0 = kly2.amplitude
        kly2.amplitude = amp0 + delta
        print('changing klystron2 amplitude...')
        self.reset(self.params.klystron_wait)
        self.wait(self.params.timeout_orb)
        print('ok!')

        print('getting trajectory...')
        self.reset(3)
        self.wait(self.params.timeout_orb)
        traj1 = _np.hstack([self.trajx, self.trajy])
        ene1 = self.energy
        print('ok!')

        print('restoring initial klystron2 amp')
        kly2.amplitude = amp0
        self.reset(self.params.klystron_wait)
        self.wait(self.params.timeout_orb)
        print('ok!')
        print('Finished!')

        denergy = ene1/ene0 - 1
        self.data['timestamp'] = _time.time()
        self.data['energy0'] = ene0
        self.data['energy'] = ene1
        self.data['kly2_amp0'] = amp0
        self.data['kly2_amp1'] = amp0 + delta
        self.data['traj0'] = traj0
        self.data['traj1'] = traj1
        self.data['eta'] = (traj1-traj0)/denergy

    def calc_model_dispmatTBBO(
            self, tb_mod, bo_mod, indices, nturns=3, dKL=None):
        """."""
        dKL = 1e-4 if dKL is None else dKL

        model = tb_mod + nturns*bo_mod
        bpms = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')

        disp_matrix = _np.zeros((len(indices), 2*len(bpms)))
        for idx, mod_indcs in enumerate(indices):
            kl0 = model[mod_indcs].KL
            model[mod_indcs].KL = kl0 + dKL/2
            dispp = self.calc_model_dispersionTBBO(model, bpms)
            model[mod_indcs].KL = kl0 - dKL/2
            dispn = self.calc_model_dispersionTBBO(model, bpms)
            disp_matrix[idx, :] = (dispp-dispn)/dKL
            model[mod_indcs].KL = kl0
        return disp_matrix

    @staticmethod
    def calc_model_dispersionTBBO(model, bpms):
        """."""
        dene = 0.0001
        rin = _np.array([
            [0, 0, 0, 0, +dene/2, 0],
            [0, 0, 0, 0, -dene/2, 0]]).T
        rout, *_ = pyaccel.tracking.line_pass(
            model, rin, bpms)
        dispx = (rout[0, 0, :] - rout[0, 1, :]) / dene
        dispy = (rout[2, 0, :] - rout[2, 1, :]) / dene
        return _np.hstack([dispx, dispy])
