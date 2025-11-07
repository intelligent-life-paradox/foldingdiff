import io
import gzip
import nglview as nv
from Bio.PDB import PDBParser
from foldingdiff.angles_and_coords import canonical_distances_and_dihedrals, create_new_chain_nerf
from foldingdiff.tmalign import run_tmalign
from moldyn.free_energy import calculate_protein_energy
import ipywidgets as widgets

def extract_pdb_gz(file_path):
    with gzip.open(file_path, "rt") as f:
        buff = f.read()
    return buff

file_path = '/home/samir/pdb-search/pdb_files/3NYK.pdb.gz'

pdb_content = extract_pdb_gz(file_path)
parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", io.StringIO(pdb_content))

EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]
angles = canonical_distances_and_dihedrals(file_path, distances=EXHAUSTIVE_DISTS, angles=EXHAUSTIVE_ANGLES)

print(angles)