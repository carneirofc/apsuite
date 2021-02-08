"""."""
import re as _re
from collections import OrderedDict as _OrderedDict
import numpy as _np
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply
from siriuspy.search import PSSearch as _PSSearch
import pymodels as _pymod
from .base import BaseClass as _BaseClass


class _Utils(_BaseClass):
    """."""

    def get_strengths(self, magname_filter):
        """."""
        mags = self._get_magnet_names(magname_filter)
        stren = _np.asarray([self.devices[mag].strength for mag in mags])
        return mags, stren

    def apply_delta_strengths(
            self, delta_strengths, magname_filter=None,
            percentage=0, apply=False, print_change=False):
        """."""
        mags, init = self.get_strengths(magname_filter)
        dstren = _np.asarray(delta_strengths)
        if len(dstren) != len(mags):
            raise ValueError(
                'delta strength vector length is incompatible with \
                number of magnets')
        implem = init + dstren * (percentage/100)
        if print_change:
            self._print_current_status(
                magnets=mags, goal_strength=init + dstren)
            print()
            print('percentage of application: {:5.1f} %'.format(percentage))
            print()
            self._print_strengths_to_be_implemented(
                percentage=percentage, magnets=mags,
                init_strength=init, implem_strength=implem)
        if apply:
            self._implement_changes(magnets=mags, strengths=implem)
        return init

    def apply_strengths(
            self, strengths, magname_filter=None,
            percentage=0, apply=False, print_change=False):
        """."""
        _, init = self.get_strengths(magname_filter)
        dstren = _np.asarray(strengths) - init
        self.apply_delta_strengths(
            delta_strengths=dstren, magname_filter=magname_filter,
            percentage=percentage, apply=apply, print_change=print_change)
        return init

    def apply_factor(
            self, magname_filter=None, factor=1, apply=False):
        """."""
        mags, init = self.get_strengths(magname_filter)
        implem = factor * init
        print(
            'Factor {:9.3f} will be applied in {:10s} magnets'.format(
                magname_filter, factor))
        if apply:
            self._implement_changes(magnets=mags, strengths=implem)
        return init

    def change_average_strengths(
            self, magname_filter=None,
            average=None, percentage=0, apply=False):
        """."""
        mags, init = self.get_strengths(magname_filter)
        curr_ave = _np.mean(init)
        # If average is None, the applied average kicks will be unchanged
        goal_ave = curr_ave if average is None else average
        diff = (goal_ave - curr_ave) * percentage/100
        implem = init + diff
        print('           actual average: {:+.4f}'.format(curr_ave))
        print('             goal average: {:+.4f}'.format(goal_ave))
        print('percentage of application: {:5.1f} %'.format(percentage))
        if apply:
            self._implement_changes(magnets=mags, strengths=implem)
        return init

    def _print_current_status(self, magnets, goal_strength):
        """."""
        diff = []
        print(
            '{:17s}  {:9s}  {:9s}  {:9s}%'.format(
                '', ' applied', ' goal', ' diff'))
        for mag, stren in zip(magnets, goal_strength):
            diff.append((self.devices[mag].strength-stren)/stren)
            print('{:17s}: {:9.6f}  {:9.6f}  {:9.6f}%'.format(
                self.devices[mag].devname, self.devices[mag].strength,
                stren, diff[-1]*100))
        print()

    def _print_strengths_to_be_implemented(
            self, percentage, magnets, init_strength, implem_strength):
        """."""
        print('-- to be implemented --')
        print('percentage: {:5.1f}%'.format(percentage))
        for mag, ini, imp in zip(magnets, init_strength, implem_strength):
            perc = (imp - ini) / ini * 100
            print(
                '{:17s}:  {:9.4f} -> {:9.4f}  [{:7.4}%]'.format(
                    self.devices[mag].devname, ini, imp, perc))

    def _implement_changes(self, magnets, strengths):
        """."""
        for mag, stren in zip(magnets, strengths):
            self.devices[mag].strength = stren
            print('\n applied!')

    def _create_devices(self, devices_names):
        """."""
        for mag in devices_names:
            if mag not in self.devices:
                self.devices[mag] = _PowerSupply(mag)

    def _get_magnet_names(self, magname_filter=None):
        if magname_filter is None:
            mags = self.devices.keys()
        else:
            regex = _re.compile(magname_filter)
            mags = [mag for mag in self.devices if regex.match(mag)]
        return mags


class SetOpticsMode(_Utils):
    """."""

    TB_QUADS = [
        'LI-Fam:PS-QF2',
        'LI-01:PS-QD2',
        'LI-01:PS-QF3',
        'TB-01:PS-QD1',
        'TB-01:PS-QF1',
        'TB-02:PS-QD2A',
        'TB-02:PS-QD2B',
        'TB-02:PS-QF2A',
        'TB-02:PS-QF2B',
        'TB-03:PS-QD3',
        'TB-03:PS-QF3',
        'TB-04:PS-QD4',
        'TB-04:PS-QF4',
        ]

    TS_QUADS = [
        'TS-01:PS-QF1A',
        'TS-01:PS-QF1B',
        'TS-02:PS-QD2',
        'TS-02:PS-QF2',
        'TS-03:PS-QF3',
        'TS-04:PS-QD4A',
        'TS-04:PS-QD4B',
        'TS-04:PS-QF4',
        ]

    def __init__(self, acc, optics_mode=None):
        """Apply strengths of families to the machine for a given optics.

        Arguments:
        - acc: TB, BO, TS or SI.
        - optics_mode: available modes in pymodels. If None, default
        optics_mode for the accelerator will be used (optional).
        """
        super().__init__()
        self.acc = acc.upper()
        self.data = dict()
        self.data['optics_mode'] = optics_mode
        self.model = None
        self.quad_list = []
        self.sext_list = []
        self.devices = _OrderedDict()
        self._select_model()
        self._select_magnets()
        self._create_devices(
            devices_names=self.quad_list + self.sext_list)
        self.model = self._pymodpack.create_accelerator()
        self._create_optics_data()
        self.famdata = self._pymodpack.get_family_data(self.model)
        # convert to integrated strengths
        for key in self.data['optics_data']:
            if key in self.famdata:
                # NOTE: This is incorrect for magnets with more than one
                # segment
                idx = self.famdata[key]['index'][0][0]
                self.data['optics_data'][key] *= self.model[idx].length

    def get_goal_optics_vector(self, magname_filter=None, optics_data=None):
        """."""
        optics_data = optics_data or self.data['optics_data']
        maglist = self._get_magnet_names(magname_filter)
        stren = []
        for mag in maglist:
            stren.append(optics_data[mag.dev])
        return _np.asarray(stren)

    def _select_model(self):
        if self.acc == 'TB':
            self._pymodpack = _pymod.tb
        elif self.acc == 'BO':
            self._pymodpack = _pymod.bo
        elif self.acc == 'TS':
            self._pymodpack = _pymod.ts
        elif self.acc == 'SI':
            self._pymodpack = _pymod.si

    def _select_magnets(self):
        slist = []
        pvstr = ''
        if self.acc == 'TB':
            qlist = SetOpticsMode.TB_QUADS
        elif self.acc == 'TS':
            qlist = SetOpticsMode.TS_QUADS
        else:
            pvstr = self.acc + '-Fam:PS-'
            qlist = self._pymodpack.families.families_quadrupoles()
            slist = self._pymodpack.families.families_sextupoles()
        self.quad_list = [_PVName(pvstr+mag) for mag in qlist]
        self.sext_list = [_PVName(pvstr+mag) for mag in slist]

    def _create_optics_data(self):
        optmode = self.data['optics_mode']
        if optmode is None:
            optmode = self._pymodpack.default_optics_mode
        self.data['optics_mode'] = optmode
        self.data['optics_data'] = self._pymodpack.lattice.get_optics_mode(
            optics_mode=optmode)
        if 'T' in self.acc:
            self.data['optics_data'] = self.data['optics_data'][0]
            self.model = self.model[0]

    def _get_magnet_names(self, magname_filter=None):
        maname = magname_filter
        if isinstance(maname, str) and maname.lower().startswith('quad'):
            maglist = self.quad_list
        elif isinstance(maname, str) and maname.lower().startswith('sext'):
            maglist = self.sext_list
        else:
            maglist = super()._get_magnet_names(maname)
        return maglist


class SetCorretorsStrengths(_Utils):
    """."""

    def __init__(self, acc):
        """."""
        super().__init__()
        self.acc = acc.upper()
        self._get_corr_names()
        self.devices = _OrderedDict()
        self._create_devices(
            devices_names=self.ch_list+self.cv_list)

    def _get_corr_names(self):
        ch_names = _PSSearch.get_psnames(
            {'sec': self.acc, 'dis': 'PS', 'dev': 'CH'})
        cv_names = _PSSearch.get_psnames(
            {'sec': self.acc, 'dis': 'PS', 'dev': 'CV'})
        self.ch_list = [_PVName(mag) for mag in ch_names]
        self.cv_list = [_PVName(mag) for mag in cv_names]


class SISetTrimStrengths(_Utils):
    """."""

    def __init__(self, model=None):
        """."""
        super().__init__()
        self.model = _pymod.si.create_accelerator() or model
        self.fam_data = _pymod.si.get_family_data(self.model)
        self.devices = _OrderedDict()
        self.quad_names = list()
        self.skewquad_names = list()
        self._get_quad_names()
        self._get_skewquad_names()
        self._create_devices(
            devices_names=self.quad_names+self.skewquad_names)

    def apply_model_strengths(
            self, magname_filter, goal_model, ref_model=None,
            percentage=0, apply=False, print_change=True):
        """."""
        mags, init = self.get_strengths(magname_filter)
        refmod = ref_model or self.model
        cond = len(goal_model) != len(refmod)
        cond |= goal_model.length != refmod.length
        if cond:
            raise ValueError(
                'Reference and goal models are incompatible.')
        if magname_filter.lower() == 'quadrupole':
            if ref_model is None:
                magidx = self.quads_idx
            else:
                fam = _pymod.si.get_family_data(refmod)
                magidx = fam['QN']['index']
                # NOTE: This is incorrect for magnets with more than one
                # segment
                magidx = _np.asarray([idx[0] for idx in magidx])
            stren_ref = _np.asarray([refmod[idx].KL for idx in magidx])
            stren_goal = _np.asarray([goal_model[idx].KL for idx in magidx])
        elif magname_filter.lower() == 'skew_quadrupole':
            if ref_model is None:
                magidx = self.skewquads_idx
            else:
                fam = _pymod.si.get_family_data(refmod)
                magidx = fam['QS']['index']
                # NOTE: This is incorrect for magnets with more than one
                # segment
                magidx = _np.asarray([idx[0] for idx in magidx])
            stren_ref = _np.asarray([refmod[idx].KsL for idx in magidx])
            stren_goal = _np.asarray([goal_model[idx].KsL for idx in magidx])
        diff = (stren_goal - stren_ref) * (percentage/100)
        implem = init + diff
        if print_change:
            self._print_current_status(magnets=mags, goal_strength=stren_goal)
            print()
            print('percentage of application: {:5.1f} %'.format(percentage))
            print()
            self._print_strengths_to_be_implemented(
                percentage=percentage, magnets=mags,
                init_strength=init, implem_strength=implem)
        if apply:
            self._implement_changes(magnets=mags, strengths=implem)
        return init

    # private methods
    def _get_quad_names(self):
        """."""
        idcs = self.fam_data['QN']['index']
        # NOTE: This is incorrect for magnets with more than one segment
        self.quads_idx = _np.asarray([idx[0] for idx in idcs])

        for qidx in self.quads_idx:
            name = self.model[qidx].fam_name
            idc = self.fam_data[name]['index'].index([qidx, ])
            sub = self.fam_data[name]['subsection'][idc]
            inst = self.fam_data[name]['instance'][idc]
            qname = f'SI-{sub}:PS-{name}-{inst}'
            self.quad_names.append(qname.strip('-'))

    def _get_skewquad_names(self):
        """."""
        idcs = self.fam_data['QS']['index']
        # NOTE: This is incorrect for magnets with more than one segment
        self.skewquads_idx = _np.asarray([idx[0] for idx in idcs])

        for qidx in self.skewquads_idx:
            name = self.model[qidx].fam_name
            idc = self.fam_data[name]['index'].index([qidx, ])
            sub = self.fam_data[name]['subsection'][idc]
            inst = self.fam_data[name]['instance'][idc]
            qname = f'SI-{sub}:PS-QS-{inst}'
            self.skewquad_names.append(qname.strip('-'))

    def _get_magnet_names(self, magname_filter=None):
        maname = magname_filter
        if isinstance(maname, str) and maname.lower().startswith('quad'):
            maglist = self.quad_names
        elif isinstance(maname, str) and maname.lower().startswith('skew'):
            maglist = self.skewquad_names
        else:
            maglist = super()._get_magnet_names(maname)
        return maglist
