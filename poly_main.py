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

# import an energy model
import sys
sys.path.insert(0, "ANI")
import ani

T = 600 # Temperature of the simulation in Kelvin
poly_time_step = 1  # in units fs.

# we now create a polyisoprene molecule
polyisoprene = pubchem_atoms_search(smiles="CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C")

# Fix 2 carbon atoms at opposite ends of the carbon chain
fixed_atom_1 = 27
fixed_atom_2 = 29
polyisoprene.set_constraint(FixAtoms(indices=[fixed_atom_1,fixed_atom_2]) )

# Setup the energy model.
polyisoprene.calc = ani.calculator
polyisoprene.get_potential_energy()

# Get distribution
MaxwellBoltzmannDistribution(polyisoprene, temperature_K=T)

# Set up dynamics object
dynamics = Langevin(polyisoprene, temperature_K=T, timestep=poly_time_step*units.fs, friction=0.01) # units.fs means femto seconds so time step is 0.5x10^(-15)s

poly_number_of_timesteps = 100000 # 100,000 for the final sim. This is the number of time steps: if time step is 0.5 ns then the total simulation time is 0.5 * this number.
number_of_stretches = 10 # In our final plot this will be the number of data points we have.
sample_step = 5 # will be 100 in the final sim? Every this number of time steps we take a sample and report the positions, forces, etc.

num_of_samples_per_stretch = poly_number_of_timesteps/sample_step

equilibrium_steps = 200 # number of timesteps we run in between steps to ensure out molecule doesnt become unstable. 750
big_delta_step = 3 # But this is to big to take in 1 go 2.5
small_delta_step = 0.02 # This will be our small stretch 0.05
num_of_delta_steps = int(big_delta_step/small_delta_step)

maximim_force = 0.5 # This is the max force when we optimise the molecules position.

def stretch_atoms():

    for i in range(num_of_delta_steps):
        dr = (polyisoprene.get_positions()[fixed_atom_1,:]-polyisoprene.get_positions()[fixed_atom_2,:])
        r = np.linalg.norm(dr)  # Calculates the L2 norm of the displacement vector = sqrt(x^2+y^2+z^2)
        dru = dr/r # Unit vecotr between the 2 atoms

        new_pos = polyisoprene.get_positions()[fixed_atom_1,:] + dru * small_delta_step
        # Set new position
        polyisoprene.positions[fixed_atom_1] = new_pos

        # Now run equilibrium steps
        dynamics.run(equilibrium_steps)


    # Finally after the stretch we want to optimise the molecule so that it is in its lowest state.
    # opt = PreconLBFGS(polyisoprene)
    # opt.run(fmax=maximim_force) # fmax gives the maximum force tolerance. lower the fmax the longer this takes.

    print("Stretch performed")


xyzfile = open('polyisoprene_trajectory_t=600_10.xyz', 'w') # the file we are going to record the structures to, the visualiser application "Ovito" can read such XYZ files.

def report():
    t = dynamics.get_time()/units.fs

    ase.io.write(xyzfile, polyisoprene, format="extxyz")

    print("time: {:.3f} fs ".format(t))

number_of_samples = int(poly_number_of_timesteps/sample_step)

# First I want to optimise the position.
opt = PreconLBFGS(polyisoprene)
opt.run(fmax=maximim_force) # fmax gives the maximum force tolerance. lower the fmax the longer this takes.

# This loop runs the whole simulation
for i in range(number_of_stretches):
    # For each fixed distance
    for j in range(number_of_samples):
        dynamics.run(sample_step)
        report()

    stretch_atoms() # We stretch the atoms.


xyzfile.close()
del dynamics
# Simulation finishes
