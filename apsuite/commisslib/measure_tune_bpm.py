"Methods for measure tunes by bpm data"

import numpy as _np
import time as _time
import pyaccel as _pa
from numpy.fft import rfft, rfftfreq, rfftn
import matplotlib.pyplot as _plt

from siriuspy.sofb.csdev import SOFBFactory

from siriuspy.devices import CurrInfoBO, \
    Trigger, Event, EVG, RFGen, SOFB, PowerSupplyPU, FamBPMs

from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class BPMeasureParams(_ParamsBaseClass):
    """."""
    def __init__(self):
        """."""
        super().__init__()
        # self.n_bpms = 50  # Numbers of BPMs
        self.nr_points_after = 0
        self.nr_points_before = 10000
        self.bpms_timeout = 30  # [s]
        self.trigger_source = 'DigBO'
        self.trigger_source_mode = 'Injection'
        self.extra_delay = 0
        self.nr_pulses = 1
        


class BPMeasure(_ThreadBaseClass):
    """."""
    def __init__(self, params=None, isonline=True):
        """."""
        params = BPMeasureParams() if params is None else params
        super().__init__(params=params, isonline=isonline)
        self.sofb_data = SOFBFactory.create('BO')
        if self.isonline:
            self._create_devices()

    def _create_devices(self):
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['bobpms'] = FamBPMs(FamBPMs.DEVICES.BO)
        # self.devices['event'] = Event('Study')
        self.devices['event'] = Event('DigBO')
        self.devices['evg'] = EVG()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.BO)
        self.devices['trigbpm'] = Trigger('BO-Fam:TI-BPM')
        self.devices['evg'] = EVG()
        self.devices['rfgen'] = RFGen()
        self.devices['ejekckr'] = PowerSupplyPU(PowerSupplyPU.
                                                DEVICES.BO_EJE_KCKR)

    def configure_bpms(self):
        """."""
        prms = self.params
        bobpms = self.devices['bobpms']
        trigbpm = self.devices['trigbpm']

        # Configure acquisition rate and doing bpms listen external trigger
        bobpms.mturn_config_acquisition(
            nr_points_after=prms.nr_points_after,
            nr_points_before=prms.nr_points_before,
            acq_rate='TbT', repeat=False, external=True)
        bobpms.mturn_reset_flags()

        # Configure BPMs trigger to listen to DigBO event
        trigbpm.source = prms.trigger_source
        trigbpm.nr_pulses = prms.nr_pulses
        delay0 = trigbpm.delay
        trigbpm.delay = delay0 + prms.extra_delay
        self.devices['event'].mode = prms.trigger_source_mode

    def get_orbit_data(self, injection=False, external_trigger=False):
        """Get orbit data from BPMs in TbT acquisition rate
        BPMs must be configured to listen DigBO event and the DigBO
        event must be in Injection mode.
        If injection is True, then injection is turned on before the measure.
        If external_trigger is True, the event will listen a external trigger.
        """
        prms = self.params
        bobpms = self.devices['bobpms']
        trigbpm = self.devices['trigbpm']

        # Inject and start acquisition
        bobpms.cmd_mturn_acq_start()
        if external_trigger:
            self.devices['event'].cmd_external_trigger()
        if injection:
            self.devices['evg'].cmd_turn_on_injection()
        _time.sleep(2)
        ret = bobpms.mturn_wait_update_flags(timeout=prms.bpms_timeout)
        if ret != 0:
            print(f'Problem waiting BPMs update. Error code: {ret:d}')
            return dict()
        orbx, orby = bobpms.get_mturn_orbit()

        # Store data
        bpm0 = bobpms[0]
        csbpm = bpm0.csdata
        data = dict()
        data['timestamp'] = _time.time()
        data['rf_frequency'] = self.devices['rfgen'].frequency
        data['current_150mev'] = self.devices['currinfo'].current150mev
        data['current_1gev'] = self.devices['currinfo'].current1gev
        data['current_2gev'] = self.devices['currinfo'].current2gev
        data['current_3gev'] = self.devices['currinfo'].current3gev
        data['orbx'], data['orby'] = orbx, orby
        data['mt_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = trigbpm.delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
                                        bpm0.switching_mode]
        self.data = data

    def load_orbit(self, data=None, orbx=None, orby=None):
        """Load orbit data into the object. You can pass the
        intire data dictionary or just the orbits. If data argument
        is provided, orbx and orby become optional"""

        if not hasattr(self, 'data'):
            self.data = dict()

        if data is not None:
            self.data = data
        if orbx is not None:
            self.data['orbx'] = orbx
        if orby is not None:
            self.data['orby'] = orby

    def dft(self, bpm_indices=None):
        """Apply a dft at bpms.

        Args:
        - bpm_indices (int, list or np.array): BPM indices whose dft will
        be applied. Default is return the dft of all bpms.

        Returns:
         - spectrumx, spectrumy, freqs: The first two are spectra np.arrays
         of dimension #freqs x #bpm_indices, and freqs is a np.array with
         the frequency domain values.
        """
        if bpm_indices is not None:
            orbx = self.data['orbx'][:, bpm_indices]
            orby = self.data['orby'][:, bpm_indices]
        else:
            orbx = self.data['orbx']
            orby = self.data['orby']

        x_beta = orbx - orbx.mean(axis=0)
        y_beta = orby - orby.mean(axis=0)

        N = x_beta.shape[0]
        freqs = rfftfreq(N)

        if isinstance(bpm_indices, int):
            spectrumx = _np.abs(rfft(x_beta))
            spectrumy = _np.abs(rfft(y_beta))
        else:
            spectrumx = _np.abs(rfftn(x_beta, axes=[0]))
            spectrumy = _np.abs(rfftn(y_beta, axes=[0]))

        return spectrumx, spectrumy, freqs

    def naff_tunes(self, dn=None, window_param=1, bpm_indices=None):
        """Computes the tune evolution from the BPMs matrix with a moving
            window of length dn.
           If dn is not passed, the tunes are computed using all points."""

        if bpm_indices is not None:
            x = self.data['orbx'][:, bpm_indices]
            y = self.data['orby'][:, bpm_indices]
        else:
            x = self.data['orbx']
            y = self.data['orby']
        N = x.shape[0]

        if dn is None:
            return self.tune_by_naff(x, y)
        else:
            tune1_list = []
            tune2_list = []
            slices = _np.arange(0, N, dn)
            for idx in range(len(slices)-1):
                idx1, idx2 = slices[idx], slices[idx+1]
                sub_x = x[idx1:idx2, :]
                sub_y = y[idx1:idx2, :]
                tune1, tune2 = self.tune_by_naff(sub_x, sub_y, window_param=1,
                                                 decimal_only=True)
                tune1_list.append(tune1)
                tune2_list.append(tune2)

        return _np.array(tune1_list), _np.array(tune2_list)

    def spectrogram(self, dn=None, overlap=True, bpm_indices=None):
        """."""
        if bpm_indices is not None:
            x = self.data['orbx'][:, bpm_indices]
            y = self.data['orby'][:, bpm_indices]
        else:
            x = self.data['orbx']
            y = self.data['orby']

        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
        N = x.shape[0]

        if dn is None:
            dn = N//50

        dn = int(dn)

        freqs = rfftfreq(dn)
        tune1_matrix = _np.zeros([freqs.size, N-dn])
        tune2_matrix = tune1_matrix.copy()

        if not overlap:
            slices = _np.arange(0, N, dn)
            for idx in range(len(slices)-1):
                idx1, idx2 = slices[idx], slices[idx+1]
                sub_x = x[idx1:idx2, :]
                sub_y = y[idx1:idx2, :]

                espectra_by_bpm_x = _np.abs(rfftn(sub_x, axes=[0]))
                espectra_by_bpm_y = _np.abs(rfftn(sub_y, axes=[0]))
                tune1_matrix[:, idx1:idx2] = _np.mean(espectra_by_bpm_x,
                                                      axis=1)[:, None]
                tune2_matrix[:, idx1:idx2] = _np.mean(espectra_by_bpm_y,
                                                      axis=1)[:, None]
        else:
            for n in range(N-dn):
                sub_x = x[n:n+dn, :]
                sub_y = y[n:n+dn, :]
                espectra_by_bpm_x = _np.abs(rfftn(sub_x, axes=[0]))
                espectra_by_bpm_y = _np.abs(rfftn(sub_y, axes=[0]))

                tune1_matrix[:, n] = _np.mean(espectra_by_bpm_x, axis=1)
                tune2_matrix[:, n] = _np.mean(espectra_by_bpm_y, axis=1)

        tune_matrix = tune1_matrix + tune2_matrix

        # normalizing this matrix to get a better heatmap plot:
        tune_matrix = (tune_matrix - tune_matrix.min()) / \
            (tune_matrix.max() - tune_matrix.min())

        # plots spectogram
        _plot_heatmap(tune_matrix, freqs)

        return tune1_matrix, tune2_matrix, freqs

    @staticmethod
    def tune_by_naff(x, y, window_param=1, decimal_only=True):
        """."""
        M = x.shape[1]
        beta_osc_x = x - _np.mean(x, axis=0)
        beta_osc_y = y - _np.mean(y, axis=0)

        Ax = beta_osc_x.ravel()
        Ay = beta_osc_y.ravel()

        freqx, _ = _pa.naff.naff_general(Ax, is_real=True, nr_ff=1,
                                         window=window_param)
        freqy, _ = _pa.naff.naff_general(Ay, is_real=True, nr_ff=1,
                                         window=window_param)
        tune1, tune2 = M*freqx, M*freqy

        if decimal_only is False:
            return _np.abs(tune1), _np.abs(tune2)
        else:
            tune1, tune2 = _np.abs(tune1 % 1), _np.abs(tune2 % 1)
            if tune1 > 0.5:
                tune1 = _np.abs(1-tune1)
            if tune2 > 0.5:
                tune2 = _np.abs(1-tune2)
            return tune1 % 1, tune2 % 1


def _plot_heatmap(tune_matrix, freqs):
    N = tune_matrix.shape[1]
    N_list = _np.arange(N)
    N_mesh, freqs_mesh = _np.meshgrid(N_list, freqs)
    _plt.pcolormesh(N_mesh, freqs_mesh, tune_matrix,
                    shading='auto', cmap='hot')
    _plt.colorbar().set_label(label="Relative Amplitude")
    _plt.ylabel('Frac. Frequency')
    _plt.xlabel('Turns')
    _plt.tight_layout()
    _plt.show()
