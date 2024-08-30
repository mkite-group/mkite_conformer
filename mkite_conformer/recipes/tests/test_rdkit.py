import json
import uuid
import unittest as ut
from pkg_resources import resource_filename

import rdkit.Chem.AllChem as Chem

from mkite_core.models import JobInfo, JobResults, RunStatsInfo, ConformerInfo
from mkite_core.tests.tempdirs import run_in_tempdir
from mkite_conformer.recipes.rdkit import ConformerGenerationRecipe, ConformerGenerationOptions


EXAMPLE_FILE = resource_filename("mkite_conformer.tests.files", "molecule.json")

with open(EXAMPLE_FILE, "r") as f:
    EXAMPLE_MOL = json.load(f)


class TestConformerGeneration(ut.TestCase):
    def setUp(self):
        self.recipe = ConformerGenerationRecipe(self.get_info())

    def get_info(self):
        return JobInfo(
            job={"id": 1, "uuid": str(uuid.uuid4())},
            recipe={},
            options={
                "num_conformers_generated": 10,
                "random_seed": 142857,
            },
            inputs=[EXAMPLE_MOL],
        )

    def get_mol(self):
        return Chem.MolFromSmiles(EXAMPLE_MOL["smiles"])

    def test_get_inputs(self):
        smi = self.recipe.get_inputs()
        self.assertIsInstance(smi, str)

    @run_in_tempdir
    def test_run(self):
        results = self.recipe.run()

        self.assertIsInstance(results, JobResults)
        self.assertEqual(len(results.nodes), 1)
        self.assertIsInstance(results.nodes[0].chemnode, dict)
        self.assertIsInstance(results.nodes[0].calcnodes, list)
        self.assertIsInstance(results.runstats, RunStatsInfo)
