..  _libraries:

****************
Scenic Libraries
****************

One of the strengths of Scenic is its ability to reuse functions, classes, and behaviors
across many scenarios, simplifying the process of writing complex scenarios. This page
describes the libraries built into Scenic to facilitate scenario writing by end users.

Simulator Interfaces
====================

Many of the simulator interfaces provide utility functions which are useful when writing
scenarios for particular simulators. See the documentation for each simulator on the
:ref:`simulators` page, as well as the corresponding module under `scenic.simulators`.

.. _domains:

Abstract Domains
================

To enable cross-platform scenarios which are not specific to one simulator, Scenic
defines *abstract domains* which provide APIs for particular application domains like
driving scenarios. An abstract domain defines a protocol which can be implemented by
various simulator interfaces so that scenarios written for that domain can be executed in
those simulators. For example, a scenario written for our
:ref:`driving domain <driving_domain>` can be run in both LGSVL and CARLA.

A domain provides a Scenic :term:`world model` which defines Scenic classes for the various types
of objects that occur in its scenarios. The model also provides a simulator-agnostic way
to access the geometry of the simulated world, by defining regions, vector fields, and
other objects as appropriate (for example, the driving domain provides a `Network` class
abstracting a road network). For domains which support dynamic scenarios, the model will
also define a set of simulator-agnostic actions for dynamic agents to use.

..  _driving_domain:

Driving Domain
--------------

The driving domain, `scenic.domains.driving`, is designed to support scenarios taking
place on or near roads. It defines generic classes for cars and pedestrians, and provides
a representation of a road network that can be loaded from standard map formats (e.g.
`OpenDRIVE <https://www.asam.net/standards/detail/opendrive/>`_). The domain supports
dynamic scenarios, providing actions for agents which can drive and walk as well as
implementations of common behaviors like lane following and collision avoidance. See the
documentation of the `scenic.domains.driving` module for further details.

MPCC Collision Avoidance
~~~~~~~~~~~~~~~~~~~~~~~~

The MPCC variants of the driving behaviors support optional trajectory-level
collision avoidance. For example::

	ego = new Car with behavior FollowLaneBehaviorMPCC(
		target_speed=10,
		collision_avoidance="all",
		collision_margin=0.25
	)

The ``collision_avoidance`` argument has three modes:

``"none"``
	Preserve the original unconstrained MPCC behavior. This is the default.

``"all"``
	Constrain the controlled agent against every other physical object in the
	scenario, including static objects and agents using non-MPCC behaviors.

``"ego_asymmetric"``
	Constrain the ego against every other object. A non-ego agent ignores the
	ego, so it may collide with ego, but still constrains itself against static
	objects and other non-ego agents.

Each agent solves a decentralized optimization problem. Other objects are
predicted over the MPCC horizon using their current planar velocity and yaw
rate; their controls are not jointly optimized and future plans are not shared.
Each rectangular footprint is conservatively covered by three oriented circles,
giving smooth constraints suitable for the IPOPT solver while allowing closer
side-by-side driving than a single circumcircle.

Safety separation is implemented with heavily penalized slack variables so that
an unavoidable or initially unsafe encounter does not make the optimization
problem infeasible. Therefore, collision avoidance is best-effort rather than a
formal guarantee. The controller applies emergency braking if the collision-aware
optimization itself fails. Solver cost grows with the number of physical objects,
the prediction horizon, and the number of footprint-circle pairs. These modes
require the optional CasADi dependency, installable through ``scenic[mpcc]``.
