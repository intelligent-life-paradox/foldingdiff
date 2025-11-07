import os
from tempfile import TemporaryFile, NamedTemporaryFile

import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, Modeller
import pdbfixer
import mdtraj as md


def calculate_protein_energy(pdb_file):
    traj = md.load(pdb_file)

    # Select only backbone atoms (N, CA, C, O)
    backbone_atoms = traj.topology.select('backbone')
    backbone_traj = traj.atom_slice(backbone_atoms)

    # Save backbone-only structure to temporary file
    with NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
        backbone_traj.save(tmp.name)
        temp_pdb = tmp.name

    try:
        # Now load with OpenMM - backbone should have no terminal issues
        pdb = PDBFile(temp_pdb)
        modeller = Modeller(pdb.topology, pdb.positions)

        forcefield = app.ForceField('amber14-all.xml')
        modeller.addHydrogens(forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )

        integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        simulation.minimizeEnergy(maxIterations=1000)
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()

        return energy

    finally:
        # Clean up temporary file
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)