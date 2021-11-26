#
# import basic atomistic simulation modules
#
import numpy as np
import ase
from ase.build import bulk
from ase.data.pubchem import pubchem_atoms_search
from ase.visualize import view
#
# now import the modules we need to run molecular dynamics
#
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.optimize.precon import PreconLBFGS
from ase.constraints import FixAtoms

# Import modules to store our results
import pandas as pd
import os
import statsmodels.tsa.stattools as sts

# import an energy model
import sys

sys.path.insert(0, "ANI")
import ani

T = 600  # Temperature of the simulation in Kelvin
time_step = 0.5  # in units fs.


def ex1():
    isoprene = pubchem_atoms_search(smiles="CC=C(C)C")
    isoprene.calc = ani.calculator

    MaxwellBoltzmannDistribution(isoprene, temperature_K=T)

    dynamics = Langevin(isoprene, temperature_K=T, timestep=time_step * units.fs,
                        friction=0.01)  # units.fs means femto seconds so time step is 0.5x10^(-15)s

    steps = 10  # Number of times we record the properties - We need our array to be sized = steps + 1 as we record t = 0.
    number_of_timesteps = 2000  # Overall the simulation will run this many steps but we will only record our data every number of 'steps'
    dist = []  # Stores distances
    time = []  # Stores times

    xyzfile = open('isoprene_t=600_s=10.xyz',
                   'w')  # the file we are going to record the structures to, the visualiser application "Ovito" can read such XYZ files.

    def report():
        dr = (isoprene.get_positions()[8, :] - isoprene.get_positions()[13, :])
        r = np.linalg.norm(dr)  # Calculates the L2 norm of the displacement vector = sqrt(x^2+y^2+z^2)
        t = dynamics.get_time() / units.fs

        # print("time: ", t, " distance: ", r)
        dist.append(r)
        time.append(t)

        ase.io.write(xyzfile, isoprene, format="extxyz")

    dynamics.attach(report,
                    interval=steps)  # the interval argument specifies how many steps of dynamics are run between each call to record the trajectory
    dynamics.run(number_of_timesteps)
    xyzfile.close()
    del dynamics
    print("done")


# EXERCISE 2

def ex2():
    polyisoprene = pubchem_atoms_search(smiles="CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C")

    fixed_atom_1 = 27
    fixed_atom_2 = 29
    polyisoprene.set_constraint(FixAtoms(indices=[fixed_atom_1, fixed_atom_2]))

    # Now we want to check they are fixed by calculating the distance between them and making sure it stays constant with time.

    # Setup the energy model.
    polyisoprene.calc = ani.calculator
    polyisoprene.get_potential_energy()

    # Get distribution
    MaxwellBoltzmannDistribution(polyisoprene, temperature_K=T)

    poly_time_step = 0.5  # fs

    # Set up dynamics object
    dynamics = Langevin(polyisoprene, temperature_K=T, timestep=poly_time_step * units.fs,
                        friction=0.01)  # units.fs means femto seconds so time step is 0.5x10^(-15)s

    poly_steps = 10  # Number of times we record the properties - We need our array to be sized = steps + 1 as we record t = 0.
    number_of_timesteps = 100  # Overall the simulation will run this many steps but we will only record our data every number of 'steps'

    distance_polyisoprene = []

    def report():
        dr = (polyisoprene.get_positions()[fixed_atom_1, :] - polyisoprene.get_positions()[fixed_atom_2, :])
        r = np.linalg.norm(dr)  # Calculates the L2 norm of the displacement vector = sqrt(x^2+y^2+z^2)
        t = dynamics.get_time() / units.fs
        distance_polyisoprene.append(r)

        print("time: {:.3f} fs | distance: {:.3f}".format(t, r))

    dynamics.attach(report,
                    interval=poly_steps)  # the interval argument specifies how many steps of dynamics are run between each call to record the trajectory
    dynamics.run(number_of_timesteps)
    del dynamics
    print("standard deviation in distances: ", np.std(
        distance_polyisoprene))  # This will be = 0 if there is no fluctuation in the distance between the 2 molecules verifying that they arent moving.


ex1()
