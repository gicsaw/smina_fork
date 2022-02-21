#!/usr/bin/env python
import os
import sys
import numpy as np
from openbabel import pybel
#import openbabel.openbabel as ob
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
import copy
import pandas as pd
import subprocess
import re
from pdbtools import pdbtools
from pdbtools import ligand_tools
import argparse

reaction_list = [
    ['Nucleophilic substitution',
     '[C&H2:1](-F)-[C:2](=[O:3])',
     '[C&H2:1]-[C:2](=[O:3])',
     'C(=O)(OCc1ccccc1)N[C@H](C(=O)N[C@H](C(=O)CF)C)Cc1ccc(cc1)O'],
    ['Nucleophilic substitution',
     '[C&H2:1](-Cl)-[C:2](=[O:3])',
     '[C&H2:1]-[C:2](=[O:3])',
     'C(=O)(OCc1ccccc1)N[C@H](C(=O)N[C@H](C(=O)CCl)COCc1ccccc1)CCCNC(=[NH2+])N'],
    ['Nucleophilic substitution',
     '[C&H2:1](-F)-[C:2](=[N:3])',
     '[C&H2:1]-[C:2](=[N:3])',
     'FCC(=N)NCCC[C@H](NC(=O)c1c(C(=O)O)cccc1)C(=O)N'],  # no reference...
    ['Nucleophilic substitution',
     '[c:1](-Cl)(:[n:2]):[c:3]:[c:4]',
     '[c:1](:[n:2]):[c:3]:[c:4]',
     'c12c(cccc1)cc(c(n2)Cl)Cn1c2c(cc(c(c2)F)C)c(c2c(=O)[nH]ccc2)c1C(=O)O'],
    ['Nucleophilic substitution',
     '[c:1](-Cl):[c:2]:[c:3]:[c:4]-[N+1:5](=[O:6])-[O-:7]',
     '[c:1]:[c:2]:[c:3]:[c:4]-[N+1:5](=[O:6])-[O-:7]',
     'Clc1ccc([N+](=O)[O-])cc1C(=O)Nc1ccccc1'],
    ['Nucleophilic substitution',
     '[c:1](-Cl):[c:2]:[c:3]:[n:4]',
     '[c:1]:[c:2]:[c:3]:[n:4]',
     'C(c1nccc(c1)Cl)O'],

    ['Addition to aldehyde',
     '[C&H1:1](=[O:2])-[C:3]',
     '[C&H2:1](-[O&H1:2])-[C:3]',
     'c1c(ccc2c1cccc2)C(=O)N[C@H](C(=O)N1[C@H](C(=O)N[C@@H](CC(=O)NS(=O)(=O)C)C=O)CCC1)C(C)C'],

    ['Addition to ketone',
     '[C&H0:1](=[O:2])(-[#6:3])-[#6:4]',
     '[C&H1:1](-[O&H1:2])(-[#6:3])-[#6:4]',
     'c1ccc(cc1)COC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H]1CN(C(=O)[C@H](CC(C)C)NC(=O)OCc2ccccc2)CC1=O'],
    ['Addition to nitrile',
     '[C&H0:1](#[N:2])-[#6:3]',
     '[C&H1:1](=[N:2])-[#6:3]',
     'C(=O)(N1CCOCC1)N[C@@H](CC(C)C)C(=O)N[C@@H](COCc1ccccc1)C#N'],
    ['Disulfide formation',
     '[S&H1:1]-[C:2][#6:3]',
     '[S&H1:1]-[C:2][#6:3]',
     'SCCNC(=O)[C@@H]1N(S(=O)(=O)c2ccc(cc2)C)CCC1'],
    ['Michael addition',
     '[#6&H1:1]1=[#6:2]-[#6:3](=[O:4])-[#6:5](=[O:6])-[#6:7]=[#6:8]-1',
     '[#6&H1:1]1:[#6:2]:[#6:3](-[O:4]):[#6:5](-[O:6]):[#6:7]:[#6:8]:1',
     'CN(N=O)C1=CC(=O)C(=O)C=C1'],

    ['Michael addition',
     '[#6&H1:1]1=[#6:2]-[#6:3](=[O:4])-[#6:5]=[#6:6]-[#6:7](=[O:8])-1',
     '[#6&H1:1]1:[#6:2]:[#6:3](-[O:4]):[#6:5]:[#6:6]:[#6:7](-[O:8]):1',
     'CC[C@H](C1=CC(=O)C(OC)=C(OC)C1=O)c1ccccc1'],
    ['Michael addition',
     '[C:1]=[C:2]-[C:3]=[O:4]',
     '[C:1]-[C:2]-[C:3]=[O:4]',
     'Cc1onc(c1)C(=O)N[C@@H](C(C)C)C(=O)C[C@@H](CCC(C)C)C(=O)N[C@H](C=CC(=O)OCC)C[C@@H]1CCNC1=O'],
    ['Michael addition',
     '[C:1]=[C:2]-[S:3](=[O:4])=[O:5]',
     '[C:1]-[C:2]-[S:3](=[O:4])=[O:5]',
     'C(=O)(OCc1ccccc1)N[C@H](C(=O)N[C@@H](CCc1ccccc1)C=CS(=O)(=O)O)Cc1ccccc1'],

    ['Ring opening (aziridine)',
     '[C:1]1(-[C:2]-[N:3]-1)-[C:4]=[O:5]',
     '[C:1](-[C:2]-[N:3])-[C:4]=[O:5]',
     'c1c2c(c(c3c(Cl)cc4c(ncnc4c3F)N3CCN(C(=O)C4CN4)CC3)c(C)c1)cn[nH]2'],
    ['Ring opening (aziridine)',
     '[C:1]1-[C:2](-[N:3]-1)-[C:4]=[O:5]',
     '[C:1]-[C:2](-[N:3])-[C:4]=[O:5]',
     'c1c2c(c(c3c(Cl)cc4c(ncnc4c3F)N3CCN(C(=O)C4CN4)CC3)c(C)c1)cn[nH]2'],
    ['Ring opening (epoxide)',
     '[C:1]1(-[C:2]-[O:3]-1)-[C:4]=[O:5]',
     '[C:1](-[C:2]-[O:3])-[C:4]=[O:5]',
     'CN(C)C(=O)[C@H](Cc1ccccc1)NC(=O)C1OC1C(=O)NCCc1ccccn1'],
    ['Ring opening (epoxide)',
     '[C:1]1-[C:2](-[O:3]-1)-[C:4]=[O:5]',
     '[C:1]-[C:2](-[O:3])-[C:4]=[O:5]',
     'CN(C)C(=O)[C@H](Cc1ccccc1)NC(=O)C1OC1C(=O)NCCc1ccccn1']
]


def pdbqt_to_covalent_receptor(in_file, out_file, chain_id0, cys_idx):

    fp = open(in_file)
    lines = fp.readlines()
    fp.close()

    fp_out = open(out_file, 'w')

    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            atom_name = line[12:16]
            residue_name = line[17:20]
            residue_idx = int(line[22:26])
            chain_id = line[21]
            if cys_idx != residue_idx:
                fp_out.write(line)
                continue
            if chain_id != chain_id:
                fp_out.write(line)
                continue
            if residue_name != 'CYS':
                print('error:', residue_name, cys_idx)
                break
            if atom_name == ' SG ':
                atom_type = line[77:79]
                if atom_type == 'SA':
                    new_atom_type = 'SC'
                else:
                    print('error:', atom_name, atom_type)
                line_out = '%s%s\n' % (line[0:77], new_atom_type)
            elif atom_name == ' CB ':
                atom_type = line[77:79]
                if atom_type == 'C ':
                    new_atom_type = 'CN'
                else:
                    print('error:', atom_name, atom_type)
                line_out = '%s%s\n' % (line[0:77], new_atom_type)
            else:
                line_out = line
            fp_out.write(line_out)

        else:
            fp_out.write(line)
    return


def read_cov_near(in_file):

    fp = open(in_file)
    lines = fp.readlines()
    fp.close()
    cov_near_dict = dict()

    conect_dict = dict()
    for line in lines:
        if line[0:6] == 'CONECT':
            conect_list = []
            for i in range(0, 8):
                ini = i * 5 + 6
                fin = (i + 1) * 5 + 6
                atom_number = line[ini:fin].strip()
                if len(atom_number) > 0:
                    conect_list += [int(atom_number)]
            conect_idx = conect_list[0]
            if conect_idx not in conect_dict:
                conect_dict[conect_idx] = conect_list[1:]
            else:
                conect_dict[conect_idx] = conect_dict[conect_idx] + \
                    conect_list[1:]

    conect_list = conect_dict[1]

    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            atom_number = int(line[6:11])
            atom = line[12:16]
            if atom[0] == 'H' or atom[1] == 'H':
                continue
            if atom_number == 1:
                cov_near_dict[atom] = 'covalent'
            if atom_number in conect_list:
                cov_near_dict[atom] = 'near'

    return cov_near_dict


def mol_with_atom_index(mol):
    atoms = mol.GetAtoms()
    atoms[0].SetAtomMapNum(1)
#    num_atoms = len(atoms)
#    for i in range(num_atoms):
#        atom = atoms[i]
#        idx = atom.GetIdx()+1
#        atom.SetAtomMapNum(idx)


def pdbqt_to_covalent(in_file, out_file, cov_near_dict):

    fp = open(in_file)
    lines = fp.readlines()
    fp.close()

    fp_out = open(out_file, 'w')

    keys = cov_near_dict.keys()

    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            atom_name = line[12:16]
            if atom_name in cov_near_dict:
                atom_type = line[77:79]
                mod_type = cov_near_dict[atom_name]
                if mod_type == 'covalent':
                    if atom_type == 'C ':
                        new_atom_type = 'CC'
                    elif atom_type == 'A ':
                        new_atom_type = 'AC'
                    elif atom_type[0] == 'S':
                        new_atom_type = 'SC'
                    else:
                        print('error: atom_type:', atom_type)

                    line_out = '%s%s\n' % (line[0:77], new_atom_type)
                elif mod_type == 'near':
                    if atom_type == 'C ':
                        new_atom_type = 'CN'
                    elif atom_type == 'A ':
                        new_atom_type = 'AN'
                    elif atom_type == 'N ':
                        new_atom_type = 'NN'
                    elif atom_type == 'NA':
                        new_atom_type = 'NM'
                    elif atom_type == 'O ':
                        new_atom_type = 'ON'
                    elif atom_type == 'OA':
                        new_atom_type = 'OM'
                    else:
                        print('error: atom_type:', atom_type)
                    line_out = '%s%s\n' % (line[0:77], new_atom_type)
                else:
                    print('strange mode', mod_type)

            else:
                line_out = line
        else:
            line_out = line
        fp_out.write(line_out)

    return


def covalent_to_pdbqt(in_file, out_file):

    fp = open(in_file)
    lines = fp.readlines()
    fp.close()

    fp_out = open(out_file, 'w')

    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            atom_name = line[12:16]
            atom_type = line[77:79]
            if atom_type == 'CC':
                new_atom_type = 'C '
            elif atom_type == 'AC':
                new_atom_type = 'A '
            elif atom_type[0] == 'SC':
                new_atom_type = 'S '
            elif atom_type == 'CN':
                new_atom_type = 'C '
            elif atom_type == 'AN':
                new_atom_type = 'A '
            elif atom_type == 'NN':
                new_atom_type = 'N '
            elif atom_type == 'NM':
                new_atom_type = 'NA'
            elif atom_type == 'ON':
                new_atom_type = 'O '
            elif atom_type == 'OM':
                new_atom_type = 'OA'
            else:
                new_atom_type = atom_type
            line_out = '%s%s\n' % (line[0:77], new_atom_type)
            fp_out.write(line_out)
        else:
            fp_out.write(line)

    return


def sim_docking(config_file, input_file, output_file):

    run_line = 'smina_fork --config %s' % config_file
    run_line += ' --ligand %s' % input_file
    run_line += ' --out %s' % output_file
    run_line += ' --exhaustiveness 8'
    run_line += ' --cpu 8'
    result = subprocess.check_output(run_line.split(),
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)


def gen_config(docking_config_file, ref_config_file, reaction_type, receptor_file):
    fp = open(ref_config_file)
    lines = fp.readlines()
    fp.close()
    fp_out = open(docking_config_file, 'w')
    line_out = 'receptor=%s\n' % (receptor_file)
    fp_out.write(line_out)
    for line in lines:
        if line.startswith('receptor'):
            continue
        line_out = line
        fp_out.write(line_out)
    if reaction_type == 'Disulfide formation':
        line = 'custom_scoring=custom_scoring_ss.txt\n'
    else:
        line = 'custom_scoring=custom_scoring.txt\n'
    fp_out.write(line)
    fp_out.close()


def main():

    parser = argparse.ArgumentParser(description='cov_dock')
    parser.add_argument('-r', '--receptor', type=str, required=True,
                        help='receptor.pdbqt')
    parser.add_argument('-c', '--chain_id', type=str, required=False,
                        default=' ',
                        help='chain_id, default is " "')
    parser.add_argument('--cys_idx', type=int, required=True,
                        help='cys_idx')
    parser.add_argument('-i', '--input_smi', type=str, required=True,
                        help='input SMILES')
    parser.add_argument('-l', '--lig_id', type=str, required=True,
                        help='lig1')
    parser.add_argument('-o', '--output_pdb', type=str, required=False,
                        default='out.pdb',
                        help='out.pdb')
    parser.add_argument('--config', type=str, required=False,
                        default='config.txt',
                        help='config.txt')
    parser.add_argument('-t', '--tmp_dir', type=str, required=False,
                        default='tmp',
                        help='tmp')

    args = parser.parse_args()
    receptor_pdbqt_file = args.receptor

    chain_id0 = args.chain_id
    cys_idx = args.cys_idx
    lig_id = args.lig_id
    smi = args.input_smi
    tmp_dir = args.tmp_dir
    output_pdb = args.output_pdb

    ref_config_file = args.config
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    docking_dir = tmp_dir

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(smi, 'is strong')
        sys.exit()

    rr = 0
    r_atoms_list = list()
    product_list = list()
    reaction_product_list = list()
    for reaction_d in reaction_list:
        reaction_type = reaction_d[0]
        reactent_smarts = reaction_d[1]
        product_smarts = reaction_d[2]
        reactent_patt = Chem.MolFromSmarts(reactent_smarts)
        r_atoms = mol.GetSubstructMatches(reactent_patt)
        num_rrr = len(r_atoms)
        if num_rrr == 0:
            continue
        rr += 1
        r_atoms_list += list(r_atoms)
        reaction_smarts = reactent_smarts + '>>' + product_smarts
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        products = rxn.RunReactants([mol])

        for product in products:
            mol_product = product[0]
            mol_product.UpdatePropertyCache()
            product_patt = Chem.MolFromSmarts(product_smarts)
            p_atoms = mol_product.GetSubstructMatches(product_patt)
            if len(p_atoms) == 0:
                print('error:', smi)
            if p_atoms[0][0] != 0:
                print('error:')
            mol_with_atom_index(mol_product)
            smi_product = Chem.MolToSmiles(mol_product, rootedAtAtom=0)

            if smi_product not in product_list:
                product_list += [smi_product]
                reaction_product_list += [[reaction_type,
                                           reactent_smarts, product_smarts, smi_product]]
        if len(product_list) > 0:
            break
#        if reaction_type == 'Michael addition':
#            break

    num_product = len(reaction_product_list)
    num_product = 1
    for i_p in range(num_product):
        reaction_product = reaction_product_list[i_p]
        reaction_type, reactent_smarts, product_smarts, smi_product = reaction_product

        pdb_file = '%s/%s_%d.pdb' % (tmp_dir, lig_id, i_p)
        pdb_file2 = '%s/%s_%d.pdb' % (docking_dir, lig_id, i_p)
        pdbqt_file = '%s/%s_%d.pdbqt' % (tmp_dir, lig_id, i_p)
        pdbqt_file2 = '%s/%s_%d.pdbqt' % (docking_dir, lig_id, i_p)
        dock_pdbqt_file = '%s/dock0_%s_%d.pdbqt' % (docking_dir, lig_id, i_p)
        dock_pdbqt_file2 = '%s/dock_%s_%d.pdbqt' % (docking_dir, lig_id, i_p)
        dock_pdb_file = output_pdb
#        dock_pdb_file = '%s/dock_%s_%d.pdb' % (docking_dir, lig_id, i_p)
        cov_receptor_pdbqt_file = '%s/receptor.pdbqt' % (docking_dir)

        pdbqt_to_covalent_receptor(
            receptor_pdbqt_file, cov_receptor_pdbqt_file, chain_id0, cys_idx)

        config_file = '%s/config.txt' % (docking_dir)
        gen_config(config_file, ref_config_file,
                   reaction_type, cov_receptor_pdbqt_file)

        ee = ligand_tools.gen_3d(smi_product, pdb_file)
        if ee is not None:
            print(smi_product, 'is strange')
            continue
        cov_near_dict = read_cov_near(pdb_file2)
#        print(cov_near_dict)
        ee = ligand_tools.pdb_to_pdbqt(pdb_file2, pdbqt_file)
        if ee is not None:
            print(lig_id, ee)

        pdbqt_to_covalent(pdbqt_file, pdbqt_file2, cov_near_dict)

        sim_docking(config_file, pdbqt_file2, dock_pdbqt_file)
        covalent_to_pdbqt(dock_pdbqt_file, dock_pdbqt_file2)
        ligand_tools.pdbqt_to_pdb_ref(
            dock_pdbqt_file2, dock_pdb_file, pdb_file)


if __name__ == '__main__':
    main()
