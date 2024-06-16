# Directional Brownian Motion

This project attempts to illustrate the directional motion of Brownian particles simulating thermophoresis, as a personal project for a "Physics Of Life" course. The simulation, written in Python, currently uses a Lennard-Jones potential.

## Implementation

As of now, the simulation is running relatively slow. The thermostat used in the simulation is not physically accurate and may violate the fluctuation-dissipation theorem. Initial velocities are normally distributed, and each particle is placed on a lattice at t=0. The system is allowed to thermalize before activating the thermostat.

### Challenges Encountered

- The thermostat introduces a potential bias to the system. Plans are in place to recode it using the Nosé-Hoover algorithm for better accuracy.
- Computational speed is a limiting factor. Adding too many particles impacts the simulation's performance and doesn't favor statistical mechanics.
- Choosing physical parameters is challenging as there is no defined ground truth in the absence of real-world parameters.

## Results

As of now, there are no quantitative results available. The simulation is a work in progress, and my first attempt at molecular dynamics has led to the identification of various interesting challenges.
Will be recoded in C++ in the following months!
