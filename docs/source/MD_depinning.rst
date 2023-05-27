MD of a depinning transition
============================

The previous examples dealt more with static properties of the interface.
The ``MD_rigid_rototrasl.py`` module allows you to integrate the overdamped equation of motion (see introduction) and follow the trajectory of a cluster in time. See the example folder molecular_dynamics for more details and inputs.

An example of interesting time evolution is the depinning of a cluster under the action of a torque and force applied on its center of mass (CM).
If these drivers are above a critical threshold, the cluster will depin and starting to translate and rotate.

The animation below shows the evolution of the cluster during the depining, with each particle coloured according to its potential energy. The applied torque is in the counter-clockwise direction and the force along :math:`x`.

.. figure:: _static/trajectory.gif
            :height: 400px

            Depinning cluster under the action of torque and force. The color of each particle indicates the substrate potential energy, increasing from blue to yello. Arrows in dicate the force acting on each particle (the legth is proportional to the magnitude).

From the output file, we can also plot the evolution of the total energy of the cluster as a function of time.

.. figure:: _static/energy.pdf
           :height: 400px

           Energy per particle as a function of time


As the drivers in this example are not much larger than the critical value, the motion is characterised by an *intermittent* dynamics reminiscent of `stick-slip <https://en.wikipedia.org/wiki/Stick-slip_phenomenon>`_ (strincly speaking, there cannot be stick-slip in a rigid system under a constant force), as highlight by the evolution of the :math:`x` coordinate of the CM below.

.. figure:: _static/x_trajectory.pdf
           :height: 400px

           Position of the cluster CM along x as a function of time
