import os
import random
import time
from typing import List

from mkite_conformer.runners.rdkit import ConformerGenerator
from mkite_conformer.runners.rdkit import ForceFields
from mkite_core.external.rdkit import RdkitInterface
from mkite_core.models import ConformerInfo
from mkite_core.models import EnergyForcesInfo
from mkite_core.models import JobResults
from mkite_core.models import NodeResults
from mkite_core.recipes import BaseOptions
from mkite_core.recipes import EnvSettings
from mkite_core.recipes import PythonRecipe
from pydantic import Field

import rdkit.Chem.AllChem as Chem


class ConformerGenerationOptions(BaseOptions):
    force_field: ForceFields = Field(
        ForceFields.MMFF94,
        description="Force field to use when generating the conformer",
    )
    num_conformers_returned: int = Field(
        20,
        description="Number of optimized, low-energy conformers to return",
    )
    num_conformers_generated: int = Field(
        200,
        description="Number of conformers generated in the initial guess",
    )
    num_attempts: int = Field(
        5,
        description="Number of attempts to try embedding",
    )
    prune_threshold: float = Field(
        0.1,
        description="RMSD tolerance in pruning the generated conformers",
    )
    cluster_rmsd_tol: float = Field(
        2.0,
        description="RMSD tolerance to use during clustering stage",
    )
    threads: int = Field(
        1,
        description="Number of threads to use during embedding process",
    )
    random_seed: int = Field(
        default_factory=lambda: random.randint(1, 10000000),
        description="random seed to use in the embedding process",
    )


class ConformerGenerationRecipe(PythonRecipe):
    _PACKAGE_NAME = "mkite_conformer"
    _METHOD = "GEN"
    SETTINGS_CLS = EnvSettings
    OPTIONS_CLS = ConformerGenerationOptions

    def get_inputs(self) -> Chem.Mol:
        inputs = super().get_inputs()
        smiles = inputs[0]["smiles"]
        return smiles

    def run(self):
        start_time = time.process_time()

        smiles = self.get_inputs()
        opts = self.get_options()

        confgen = ConformerGenerator(smiles, **opts)
        newmol, energies = confgen.run()

        end_time = time.process_time()
        duration = end_time - start_time

        return self.postprocess(newmol, energies, duration=duration)

    def postprocess(
        self, mol: Chem.Mol, energies: List[float], duration: float
    ) -> JobResults:
        interface = RdkitInterface(mol)
        nodes = []
        for conf, e in zip(interface.conformer_info, energies):
            nres = NodeResults(
                chemnode=self.create_chemnode(conf),
                calcnodes=[self.create_calcnode(e)],
            )
            nodes.append(nres)

        runstats = self.get_run_stats(duration)

        jobres = JobResults(
            job=self.get_done_job(),
            runstats=runstats,
            nodes=nodes,
        )

        jobres.to_json(os.path.join(".", JobResults.file_name()))
        return jobres

    def create_chemnode(self, conformer_info: ConformerInfo) -> dict:
        conformer_info.mol = {
            "inchikey": conformer_info.mol.inchikey,
            "smiles": conformer_info.mol.smiles,
        }
        return conformer_info.as_dict()

    def create_calcnode(self, energy: float) -> List[dict]:
        return EnergyForcesInfo(energy=energy).as_dict()
