import os
import random
import warnings
from enum import Enum
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import rdkit.Chem.AllChem as Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina


class ForceFields(str, Enum):
    MMFF94 = "mmff"
    UFF = "uff"


class ConformerGenerator:
    """Class that enables the generation of conformers for molecules
    using RDKit. By default, uses the ETKDGv3 algorithm for constraining
    the distances.
    """

    def __init__(
        self,
        smiles: str,
        force_field: str = ForceFields.MMFF94,
        num_conformers_returned: int = 20,
        num_conformers_generated: int = 200,
        num_attempts: int = 5,
        prune_threshold: float = 0.1,
        cluster_rmsd_tol: float = 2.0,
        threads: int = 1,
        random_coords: bool = False,
        random_seed: int = None,
    ):
        """Initializes the ConformerGenerator for a given `mol` and sets it up based on several options.

        Args:
            smiles (str): SMILES string for the molecules for which conformers will be generated.
            force_field (str): name of the force field to be used. Supported values: "MMFF94" and "UFF".
            num_conformers_returned (int): maximum number of conformers returned by ConformerGenerator.
            num_conformers_generated (int): maximum number of conformers generated by RDKit.
            num_attempts (int): maximum number of attempts when generating the conformers.
            prune_threshold (float): RMS threshold for considering conformers equal to each other.
            cluster_rmsd_tol (float): maximum RMSD to use when clustering conformations.
            threads (int): number of CPU threads to use when generating conformers.
            random_coords (bool): if True, starts with random coordinates when generating conformers.
            random_seed (int): random seed for conformer generation.
        """
        self._smiles = self.get_canonical_smiles(smiles)
        self.mol = self.get_mol()
        self.force_field = force_field
        self.num_conformers_returned = num_conformers_returned
        self.num_conformers_generated = num_conformers_generated
        self.num_attempts = num_attempts
        self.prune_threshold = prune_threshold
        self.cluster_rmsd_tol = cluster_rmsd_tol
        self.threads = threads
        self.random_coords = random_coords
        self.random_seed = (
            random.randint(1, 10000000) if random_seed is None else random_seed
        )

    @property
    def smiles(self) -> str:
        return self._smiles

    def get_canonical_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    def get_mol(self, add_hs=True) -> Chem.Mol:
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        return mol

    def run(self) -> Tuple[Chem.Mol, List[float]]:
        conf_ids = self.generate()
        energies = self.optimize()
        self.align()
        clusters = self.cluster()

        newmol, energies = self.downselect(energies, clusters)

        return newmol, energies

    def generate(self) -> List[int]:
        params = rdDistGeom.ETKDGv3()
        params.maxAttempts = self.num_attempts
        params.pruneRmsThres = self.prune_threshold
        params.randomSeed = self.random_seed
        params.useRandomCoords = self.random_coords
        params.threads = self.threads

        conformers = Chem.EmbedMultipleConfs(
            self.mol,
            self.num_conformers_generated,
            params,
        )

        return conformers

    def get_force_field(self) -> Callable:
        if self.force_field == ForceFields.MMFF94:
            molprops = Chem.MMFFGetMoleculeProperties(self.mol)
            ff_gen = lambda conf_id: Chem.MMFFGetMoleculeForceField(
                self.mol, molprops, confId=conf_id
            )

            if ff_gen(0) is not None:
                return ff_gen

        # the FF is either UFF or MMFF94 cannot be used for this molecule
        self.force_field = ForceFields.UFF
        ff_gen = lambda conf_id: Chem.UFFGetMoleculeForceField(self.mol, confId=conf_id)
        return ff_gen

    def optimize(self) -> Dict[int, float]:
        ff_gen = self.get_force_field()

        energies = {}
        try:
            for conf_id in range(self.mol.GetNumConformers()):
                molff = ff_gen(conf_id)
                molff.Minimize()
                energies[conf_id] = molff.CalcEnergy()

        except RuntimeError as e:
            warnings.warn(f"Error minimizing the molecule: {e}")

            # creates some fake energies to proceed
            energies = {i: 0 for i in range(self.mol.GetNumConformers())}

        return energies

    def align(self):
        return rdMolAlign.AlignMolConformers(self.mol)

    def cluster(self):
        dm = Chem.GetConformerRMSMatrix(self.mol, prealigned=False)
        clusters = Butina.ClusterData(
            dm,
            self.mol.GetNumConformers(),
            self.cluster_rmsd_tol,
            isDistData=True,
            reordering=True,
        )
        return clusters

    def downselect(
        self,
        energies: Dict[int, float],
        clusters: List[List[int]],
    ):
        def get_lowest_energy(ids: List[int], energies: Dict[int, float]):
            subset = [(i, energies[i]) for i in ids]
            subset = sorted(subset, key=lambda x: x[1])
            return subset[0]

        selected_ids = []
        for cluster in clusters:
            conf_id, energy = get_lowest_energy(cluster, energies)
            selected_ids.append(conf_id)

        selected_ids = sorted(selected_ids, key=lambda id_: energies[id_])
        selected_ids = selected_ids[: self.num_conformers_returned]
        selected_energies = [energies[i] for i in selected_ids]

        return self.duplicate_mol(self.mol, selected_ids), selected_energies

    def duplicate_mol(self, mol: Chem.Mol, conformer_ids: List[int] = None):
        if conformer_ids is None:
            conformer_ids = list(range(mol.GetNumConformers()))

        newmol = self.get_mol(add_hs=True)

        for i in conformer_ids:
            conformer = mol.GetConformer(i)
            newmol.AddConformer(conformer)

        return newmol
