Solver Strategies
=================

``radiosim.simulator`` holds the swappable solver strategies.
``VisibilitySimulator`` is the abstract contract; ``RIMESimulator`` is the
direct RIME summation with cost
``O(N_sources x N_baselines x N_frequencies)``, and ``MModeSimulator`` is the
``SCI-004`` m-mode full-sidereal harmonic forward model.

``execution.simulator`` accepts exactly the keys of the simulator registry,
which are ``rime`` and ``mmode``. A new algorithm arrives as a registry entry
that the one selector already honours, never as a second unread configuration
field.

The whole-sky strategy boundary
-------------------------------

Both strategies are selected through one immutable ``SkySolveRequest`` carrying
the **whole** resolved ``SkyModel`` together with the point arrays, instrument
view, beam system, location, time and frequency coordinates, receptors, Jones
inventory, backend and worker policy; each returns one ``SkySolveOutcome`` with
the backend-native ``(time, baseline, frequency, 2, 2)`` receptor cube.

``RIMESimulator.solve`` is a thin wrapper around the maintained
point/HEALPix/hybrid path in ``radiosim.core.hybrid``: its arithmetic, component
order, source reduction, result bytes and fingerprints are unchanged.
``MModeSimulator.solve`` consumes the same whole request and never calls the
direct point or HEALPix kernels.

m-mode scope and capability truth
---------------------------------

``MModeSimulator`` is a **second complete forward model**, not a Jones term, a
point-source optimization, a map maker, or a new name for the direct sum. Its
observing regime is deliberately narrow: the array, beams, receptors and
accepted instrumental terms are fixed in the terrestrial frame, the phase centre
and boresight are the existing fixed zenith, the sky is sidereal and fixed over
one Earth rotation, and the sample centres are a complete, unflagged, uniformly
spaced Earth Rotation Angle cycle with no duplicated endpoint. Noise modelling,
inverse map making, pseudo-inverses, KL filtering, power-spectrum estimation,
calibration, tracking, drift-and-shift, missing samples and partial-day windows
are outside its scope.

The forward model is **full Stokes**: ``MModeSimulator.supports_polarization``
is explicitly ``True``. Capability truth here is phase-local and is stated
together with the unchanged ``RIMESimulator.supports_polarization``, so the
override is a statement about the m-mode phase rather than a weakening of the
direct solver.
``MModeSimulator.supports_gpu`` is ``False``: no end-to-end accelerator run of
this solver has been measured, and the recorded transform execution policy
``host_harmonics_backend_native_dense_v1`` describes where the work runs -- the
Astropy frame work, IERS mapping, beam sampling, HEALPix geometry and harmonic
transforms are host-side NumPy for every backend -- rather than claiming an
accelerator advantage. A polarized capability is not a speed claim. Register
row ``PERF-001`` governs every RadioSim performance statement.

Polarization in the harmonic forward model
------------------------------------------

The m-mode kernel is the *same* reference-phase response the direct RIME
builds, expanded in harmonics rather than summed over sources:

.. code-block:: text

   K^X_pqfc(n) = [J_p(n) P^X J_q^H(n)]_c  K_pq(n)  H(n)

one cell per Stokes component ``X`` in ``(I, Q, U, V)``. Three properties of
that line are load-bearing and are not free choices.

*The receptor matrix is part of the kernel.* An antenna's Jones matrix in the
celestial (North, East) tangent basis is ``J_p(n) = M_p E_p(n)``, where
``E_p`` is the sampled beam response and ``M_p = H_p C_p`` is the resolved,
direction-independent receptor and basis-transform pair -- built from the same
``radiosim.core.jones.receptor`` code objects the direct chain uses, so a
receptor, feed rotation or output basis cannot mean one thing to one solver and
something else to the other. Dropping ``M_p`` is invisible in Stokes ``I``,
because ``M P^I M^H = (1/2) M M^H = (1/2) I2`` for a unitary receptor, and
wrong for every polarized component.

*The Shaw convention enters exactly once.* The spin expansions are written in
the spherical ``(theta, phi)`` basis, whose ``theta`` points South, while
RadioSim's brightness matrix is ordered (North, East). The bridge is the single
matrix ``D = diag(-1, 1)``: the kernel uses ``D P^X D`` and the sky uses the
matching field relabelling ``U_H = -U``. There is no second, fitted or
configurable sign, and ``D`` is *not* the SCI-006 east-X permutation, which
stays inside the antenna Jones matrix and is antidiagonal.

*The constant cells act in the celestial tangent basis.* The coherency is built
in the celestial (North, East) basis of each direction; the chain's
direction-independent terms right-multiply it as constant matrices in that same
basis; and every mount-dependent tangent rotation belongs to the ``P`` term,
which is exactly the identity for the shipped ``fixed`` and unspecified mounts.
Constant coefficients on spin-weighted fields preserve the integrand's spin
weight, so the spin-``±2`` Gauss-Legendre quadrature stays spectrally exact. A
genuinely ground-anchored, direction-dependent response would need a measured
tangent transport instead; transporting a *constant* matrix into the rotating
local basis is the identity re-expression of a zenith-singular field, not an
alternative convention.

The transfer is expanded per science field in the fixed order
``("I", "+2", "-2", "V")``,

.. code-block:: text

   B^(+2)_lm = integral( (K^Q - i K^U)  {+2}Y_lm  dOmega )
   B^(-2)_lm = integral( (K^Q + i K^U)  {-2}Y_lm  dOmega )

with ``I`` and ``V`` carrying the scalar harmonics, and the forward per-``m``
product weights the two spin terms by ``1/2`` each. Those halves are a theorem
rather than a normalization: substituting a delta sky's coefficients collapses
the pair to exactly ``K^Q Q_H + K^U U_H``, and dropping either factor doubles
that contribution.

A run whose resolved payload has no non-zero ``Q``, ``U`` or ``V`` takes the
scalar execution path, records ``execution_path: "scalar"``, and evaluates only
the ``I`` field -- not as an approximation, but because the other three
contribute exactly zero.

Truncation is gated, not assumed
--------------------------------

Every m-mode production run executes a **two-tier gate** before any result or
output path is created. A run that fails it has no result: the limits are fixed
and are never widened to admit a run.

*Tier 1a* gates the harmonic pipeline itself at ``1e-8``. The complete pipeline
is evaluated once more with the horizon factor removed -- everything else
identical, through the same code path -- on both the production and the
``qcheck`` quadrature, and the two cubes must agree. That integrand is smooth,
so Gauss-Legendre is spectrally exact through the band and any sign,
normalization, weight, packing or dropped-mode defect fails sharply.

*Tier 1b* records the with-horizon quadrature shell. It carries no universal
limit, because the strict ``alt > 0`` factor makes no finite quadrature exact:
that residual converges only as ``nside**-2``. Its bound is a reviewed
per-fixture budget in the phase evidence.

*Tier 2* compares against the complete frozen-frame direct oracle and reports a
**truncation deficit**, never an agreement. The forward product reconstructs the
*band-limited* transfer kernel, so the deficit against the exact direct sum is a
property of the method rather than a defect; for a delta sky it is exactly
``S*K_L(n_s)``. The obligation is convergence and disclosure: the run recomputes
the deficit at a quarter and a half of its ``lmax`` -- as exact block-table
projections of the retained vectors, never re-transforms -- and requires strict
monotone decrease with a quarter-to-full factor of at least two. The measured
deficit enters the result's provenance record, so no consumer can read an m-mode
result without it.

Because tier 2 gates on convergence rather than equality, an acceptance fixture
must sit in the convergent regime. The governing conditions are geometric: every
payload direction must stay well clear of the horizon over the whole cycle, and
``lmax`` is pinned by measurement, because the quarter-to-full factor is *not*
monotone in ``lmax``. A fixture is qualified by measuring its three-level
deficit sequence and adopting it only with real margin; a predicate is never
widened to admit a fixture.

The ``SCI-004`` register row remains ``ROADMAP``: the design gate is accepted,
the production phases are separately gated, and no phase of it has closed the
row.

m-mode forward model
--------------------

.. automodule:: radiosim.simulator.mmode
   :members:
   :undoc-members:
   :show-inheritance:

Registry
--------

.. automodule:: radiosim.simulator
   :no-members:
   :no-special-members:

.. autofunction:: radiosim.simulator.get_simulator

.. autofunction:: radiosim.simulator.list_simulators

Solver contract
---------------

.. automodule:: radiosim.simulator.base
   :members:
   :undoc-members:
   :show-inheritance:

Direct RIME summation
---------------------

.. automodule:: radiosim.simulator.rime
   :members:
   :undoc-members:
   :show-inheritance:
