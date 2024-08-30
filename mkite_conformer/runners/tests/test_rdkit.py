import os
import json
import uuid
import unittest as ut
from pkg_resources import resource_filename

import rdkit.Chem.AllChem as Chem

from mkite_conformer.runners.rdkit import ConformerGenerator


class TestConformerGenerator(ut.TestCase):
    def setUp(self):
        self.smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.recipe = ConformerGenerator(
            self.smiles,
            random_seed=142857,
        )

    @property
    def mol(self):
        return self.recipe.mol

    def test_generate(self):
        confs = self.recipe.generate()
        self.assertEqual(len(confs), self.mol.GetNumConformers())

    def test_optimize(self):
        confs = self.recipe.generate()
        energies = self.recipe.optimize()

        self.assertIsInstance(energies, dict)
        self.assertEqual(len(energies), self.mol.GetNumConformers())

    def test_cluster(self):
        confs = self.recipe.generate()
        energies = self.recipe.optimize()
        self.recipe.align()
        clusters = self.recipe.cluster()

        self.assertEqual(len(clusters), 1)

    def test_downselect(self):
        confs = self.recipe.generate()
        energies = self.recipe.optimize()
        self.recipe.align()
        clusters = self.recipe.cluster()
        new, energies = self.recipe.downselect(energies, clusters)

        self.assertEqual(new.GetNumConformers(), len(clusters))

    def test_run(self):
        new, energies = self.recipe.run()


class TestSeveralSmiles(ut.TestCase):
    def test_several(self):
        smiles = ["[H]", "[C]", "[C-]#[O+]"]

        for smi in smiles:
            recipe = ConformerGenerator(
                smi,
                random_seed=142857,
            )

            mol, energies = recipe.run()
            self.assertTrue(len(energies) > 0)
            self.assertTrue(mol.GetNumConformers() > 0)
