import numpy as _np
from numpy.polynomial.polynomial import polyfit as _polyfit, polyval as\
    _polyval
import matplotlib.pyplot as _plt
from mathphys.constants import vacuum_permeability
from mathphys.beam_optics import beam_rigidity


class Wire:
    """Simulates the magnetic field of a infinite wire.
    """
    def __init__(self, position=(0, 0), current=1):
        """ Initialize a Wire object.
        Parameters
        ----------
        position : array_like of size 2, optional
            Cartesian coordinates (x,y) of the wire position in [m], by
            default (0,0).

        current : float, optional
            Current of the wire, in units of [A], by default 1.
        """
        self._pos = None
        self._curr = None
        self.position = position  # [m]
        self.current = current    # [A]

    def calc_magnetic_field(self, r):
        """Computes the magnetic field (in units of [T]) generated by the wire
        in a set of points in space.

        Parameters
        ----------
        r : numpy.array
            Array of shape (2 x N) with the coordinates [[x1, x2, ... , xN],
            [y1, y2, ... , yN]] where the field will be returned.

        Returns
        -------
        numpy.array
            Array with the same shape of r, containing the magnetic field
            components at each point in r.

        Raises
        ------
        ValueError
            If input r hasn't shape of type (2 X N).
        """
        if r.shape[0] != 2:
            raise ValueError(
                "The input position vector r must have shape of type (2, N)")
        r = r.reshape(2, -1)
        mu0 = vacuum_permeability
        r_w = self.position[:, None]  # Wire positions
        rc = r - r_w  # Cursive r
        theta = _np.arctan2(rc[1], rc[0])
        theta_vec = _np.array([-_np.sin(theta), _np.cos(theta)])
        rc_norm = _np.linalg.norm(rc, axis=0)[None, :]
        mag_field = mu0 * self.current/(2*_np.pi*rc_norm)*theta_vec
        return mag_field

    @property
    def position(self):
        """."""
        return self._pos.copy()

    @position.setter
    def position(self, position):
        """."""
        position = self._check_r_size(position)
        self._pos = position

    @property
    def current(self):
        """."""
        return self._curr

    @current.setter
    def current(self, current):
        """."""
        self._curr = current

    @staticmethod
    def _check_r_size(r):
        """."""
        r = _np.asarray(r)
        if r.size == 2:
            return r.ravel()
        else:
            raise ValueError('r must be a numpy array of size 2.')


class NLK:
    """Simulates the magnetic field generated by the nonlinear kicker.
    """
    def __init__(self, positions=None, current=1850):
        """Initialize a NLK object

        Parameters
        ----------
        positions : numpy.array, optional
            Array of shape (2, 8) with the wires positions [[x1, ... , xN],
            [y1, ... , yN]], by default loads the nominal wire positions.

        current : float, optional.
            Current of the wires in Amperes, by default 1850 A.
        """
        if positions is not None:
            wire_positions = positions
        else:
            wire_positions = _np.zeros([2, 8])
            s1 = _np.array([1, -1, 1, -1])
            s2 = _np.array([1, 1, -1, -1])
            wire_positions[:, ::2] = _np.array([s1*7, s2*5.21])
            wire_positions[:, 1::2] = _np.array([s1*10, s2*5.85])
            wire_positions *= 1e-3

        # Creating wires
        currents = _np.zeros(8)
        currents[::2] = -current
        currents[1::2] = current

        self._wires = [Wire() for i in range(8)]
        self.positions = wire_positions
        self.currents = currents

    @property
    def wires(self):
        """List with the Wires objects that compose the NLK.
        Returns
        -------
        list of Wires objects.
        """
        return self._wires

    @property
    def positions(self):
        """Position coordinates of the wires.

        Returns
        -------
        numpy.array
            Returns a numpy.array of shape (2, 8) with the wires coordinates.
        """
        return _np.array([wi.position for wi in self.wires]).T

    @positions.setter
    def positions(self, wires_positions):
        """."""
        for i in range(wires_positions.shape[1]):
            self.wires[i].position = wires_positions[:, i]

    @property
    def currents(self):
        """Currents of the NLK wires in [A].

        Returns
        -------
        numpy.array
            Single dimension array with the currents of each wire in NLK.
        """
        return _np.array([wi.current for wi in self._wires])

    @currents.setter
    def currents(self, wires_currents):
        for i, current in enumerate(wires_currents):
            self._wires[i].current = current

    def calc_magnetic_field(self, r):
        """Computes the magnetic field generated by NLK in a set of points.

        Parameters
        ----------
        r : numpy.array
            Array of shape (2 x N) with the coordinates [[x1, x2, ... , xN],
            [y1, y2, ... , yN]] where the field will be returned.

        Returns
        -------
        numpy.array
            Array with the same shape of r, containing the magnetic field
            components in [T] at each point in r.

        Raises
        ------
        ValueError
            If the input r hasn't the shape of type (2 X N).
        """
        if r.shape[0] != 2:
            raise ValueError(
                "The input position vector r must have shape of type (2, N)")
        mag_field = _np.zeros(r.shape)
        for wire in self.wires:
            mag_field += wire.calc_magnetic_field(r)
        return mag_field

    def get_vertical_magnetic_field(self, y0=0):
        """NLK vertical field at a horizontal plane y=y0 for x ∈ [-12, 12] mm.

        Parameters
        ----------
        y0 : float, optional
            y coordinate of the horizontal plane.

        Returns
        -------
        x_pos: numpy.array
            Values of x where the field was computed.

        fieldy: numpy.array
            Vertical field in [T] at the plane y=y0.
        """
        x_pos = _np.linspace(-12, 12)*1e-3
        y_pos = _np.ones(x_pos.shape) * y0
        r = _np.vstack([x_pos, y_pos])
        fieldy = self.calc_magnetic_field(r)[1]
        return x_pos, fieldy

    def get_horizontal_magnetic_field(self, x0=0):
        """NLK horizontal field at a vertical plane x=x0 for y ∈ [-12, 12] mm.

        Parameters
        ----------
        x0 : float, optional
            x coordinate of the horizontal plane.

        Returns
        -------
        y_pos: numpy.array
            Values of y where the field was computed.

        fieldy: numpy.array
            Horizontal field in [T] at the plane x=x0.
        """
        y_pos = _np.linspace(-12, 12)*1e-3
        x_pos = _np.ones(y_pos.shape) * x0
        r = _np.vstack([x_pos, y_pos])
        fieldy = self.calc_magnetic_field(r)[0]
        return y_pos, fieldy

    @staticmethod
    def si_nlk_kick(scale=1, fit_monomials=None, plot_flag=False, r0=0.0):
        """Generates the NLK polynom_b, its horizontal kick and the integrated
        field at y=0. Useful for set the NLK pulse in the Sirius model.

        Parameters
        ----------
        scale : float, optional
            Multiplier of the NLK max field, by default 1.

        fit_monomials : array like or list, optional
            The multipoles indices or degrees of the terms included in the
            polynom_b, by default is numpy.arange(10, dtype=int)

        plot_flag : bool, optional
            If True, plots the field profile, by default False.

        r0 : float, optional
            Closed orbit x coordinate, by default 0.0.

        Returns
        -------
        x : numpy.array
            Values of x where the field was computed.

        integ_field : numpy.array
            Integrated NLK field in [T.m].

        kickx : numpy.array
            Horizontal kick in [rad].

        polynom_b : numpy array of shape (fit_monomials,)
            Polynom B that describes the NLK kick.
        """

        nlk_length = 0.45
        if fit_monomials is None:
            fit_monomials = _np.arange(10, dtype=int)
        else:
            fit_monomials = _np.asarray(fit_monomials)

        nlk = NLK()
        x, mag_field = nlk.get_vertical_magnetic_field()

        brho, *_ = beam_rigidity(energy=3)  # energy in [GeV]
        integ_field = scale * mag_field * nlk_length  # [T.m]
        kickx = integ_field / brho  # [rad]

        coeffs = _polyfit(x=x-r0, y=kickx, deg=fit_monomials)
        fit_kickx = _polyval(x=x-r0, c=coeffs)
        if plot_flag:
            _plt.figure()
            _plt.scatter(1e3*(x-r0), 1e3*kickx, label="data points")
            _plt.plot(
                1e3*(x-r0), 1e3*fit_kickx, c=[0, 0.6, 0], label="fitted curve")
            _plt.xlabel('X [mm]')
            _plt.ylabel("Kick @ 3 GeV [mrad]")
            _plt.title("NLK Profile")
            _plt.legend()

        polynom_b = _np.zeros(1 + fit_monomials.max())
        polynom_b[fit_monomials] = -coeffs/nlk_length
        return x, integ_field, kickx, polynom_b
