# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Process Structure Data."""

from alphafold3.constants import atom_types
from alphafold3.constants import residue_names
from alphafold3.constants import side_chains
from alphafold3.constants import mmcif_names
import numpy as np


NUM_DENSE = atom_types.DENSE_ATOM_NUM
NUM_AA = len(residue_names.PROTEIN_TYPES)
NUM_AA_WITH_UNK_AND_GAP = len(
    residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP
)
NUM_RESTYPES_WITH_UNK_AND_GAP = (
    residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
)


def _make_restype_rigidgroup_dense_atom_idx():
  """Create Mapping from rigid_groups to dense_atom indices."""
  # Create an array with the atom names.
  # shape (num_restypes, num_rigidgroups, 3_atoms):
  # (31, 8, 3)
  base_atom_indices = np.zeros(
      (NUM_RESTYPES_WITH_UNK_AND_GAP, 8, 3), dtype=np.int32
  )

  # 4,5,6,7: 'chi1,2,3,4-group'
  for restype, restype_letter in enumerate(
      residue_names.PROTEIN_TYPES_ONE_LETTER
  ):
    resname = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]

    dense_atom_names = atom_types.ATOM14[resname]
    # 0: backbone frame
    base_atom_indices[restype, 0, :] = [
        dense_atom_names.index(atom) for atom in ['C', 'CA', 'N']
    ]

    # 3: 'psi-group'
    base_atom_indices[restype, 3, :] = [
        dense_atom_names.index(atom) for atom in ['CA', 'C', 'O']
    ]
    for chi_idx in range(4):
      if side_chains.CHI_ANGLES_MASK[restype][chi_idx]:
        atom_names = side_chains.CHI_ANGLES_ATOMS[resname][chi_idx]
        base_atom_indices[restype, chi_idx + 4, :] = [
            dense_atom_names.index(atom) for atom in atom_names[1:]
        ]
  dense_atom_names = atom_types.DENSE_ATOM['A']
  nucleic_rigid_atoms = [
      dense_atom_names.index(atom) for atom in ["C1'", "C3'", "C4'"]
  ]
  for nanum, _ in enumerate(residue_names.NUCLEIC_TYPES):
    # 0: backbone frame only.
    # we have aa + unk + gap, so we want to start after those
    resnum = nanum + NUM_AA_WITH_UNK_AND_GAP
    base_atom_indices[resnum, 0, :] = nucleic_rigid_atoms

  return base_atom_indices


RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX = _make_restype_rigidgroup_dense_atom_idx()


def _make_restype_pseudobeta_idx():
  """Returns indices of residue's pseudo-beta."""
  restype_pseudobeta_index = np.zeros(
      (NUM_RESTYPES_WITH_UNK_AND_GAP,), dtype=np.int32
  )
  for restype, restype_letter in enumerate(
      residue_names.PROTEIN_TYPES_ONE_LETTER
  ):
    restype_name = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]
    atom_names = list(atom_types.ATOM14[restype_name])
    if restype_name in {'GLY'}:
      restype_pseudobeta_index[restype] = atom_names.index('CA')
    else:
      restype_pseudobeta_index[restype] = atom_names.index('CB')
  for nanum, resname in enumerate(residue_names.NUCLEIC_TYPES):
    atom_names = list(atom_types.DENSE_ATOM[resname])
    # 0: backbone frame only.
    # we have aa + unk , so we want to start after those
    restype = nanum + NUM_AA_WITH_UNK_AND_GAP
    if resname in {'A', 'G', 'DA', 'DG'}:
      restype_pseudobeta_index[restype] = atom_names.index('C4')
    else:
      restype_pseudobeta_index[restype] = atom_names.index('C2')
  return restype_pseudobeta_index


RESTYPE_PSEUDOBETA_INDEX = _make_restype_pseudobeta_idx()


def _make_aatype_dense_atom_to_atom37():
  """Map from dense_atom to atom37 per residue type."""
  restype_dense_atom_to_atom37 = []  # mapping (restype, dense_atom) --> atom37
  for rt in residue_names.PROTEIN_TYPES_ONE_LETTER:
    atom_names = list(
        atom_types.ATOM14_PADDED[residue_names.PROTEIN_COMMON_ONE_TO_THREE[rt]]
    )
    atom_names.extend([''] * (NUM_DENSE - len(atom_names)))
    restype_dense_atom_to_atom37.append(
        [(atom_types.ATOM37_ORDER[name] if name else 0) for name in atom_names]
    )
  # Add dummy mapping for restype 'UNK', '-' (gap), and nucleics [but not DN].
  for _ in range(2 + len(residue_names.NUCLEIC_TYPES_WITH_UNKNOWN)):
    restype_dense_atom_to_atom37.append([0] * NUM_DENSE)

  restype_dense_atom_to_atom37 = np.array(
      restype_dense_atom_to_atom37, dtype=np.int32
  )
  return restype_dense_atom_to_atom37


PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37 = _make_aatype_dense_atom_to_atom37()


def _make_nucleic_aatype_dense_atom_to_atom29(target_chain_poly_type: str):
  """Create mapping from (aatype, dense_idx) to ATOM29_IDX for nucleic acids."""
  # Initialize map with zeros. Shape: (NUM_RESTYPES_WITH_UNK_AND_GAP, NUM_DENSE)
  restype_dense_atom_to_atom29 = np.zeros(
      (NUM_RESTYPES_WITH_UNK_AND_GAP, NUM_DENSE), dtype=np.int32
  )

  # Create a reverse mapping from aatype_idx to the residue name string
  # used in POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP.
  idx_to_resname_key = {
      v: k
      for k, v in residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP.items()
  }

  for restype_idx in range(NUM_RESTYPES_WITH_UNK_AND_GAP):
    resname_key = idx_to_resname_key.get(restype_idx)
    if resname_key is None:
      # This case should ideally not be reached if idx_to_resname_key is complete.
      continue

    actual_resname_for_dense_atom_lookup = None
    if target_chain_poly_type == mmcif_names.RNA_CHAIN:
      if resname_key in residue_names.RNA_TYPES:  # 'A', 'G', 'C', 'U'
        actual_resname_for_dense_atom_lookup = resname_key
      elif resname_key == residue_names.UNK_NUCLEIC_ONE_LETTER:  # 'N'
        # Map generic nucleic unknown to RNA unknown for DENSE_ATOM keys
        actual_resname_for_dense_atom_lookup = residue_names.UNK_RNA # 'N'

    elif target_chain_poly_type == mmcif_names.DNA_CHAIN:
      if resname_key in residue_names.DNA_TYPES:  # 'DA', 'DG', 'DC', 'DT'
        actual_resname_for_dense_atom_lookup = resname_key
      elif resname_key == residue_names.UNK_NUCLEIC_ONE_LETTER:  # 'N'
        # Map generic nucleic unknown to DNA unknown for DENSE_ATOM keys
        actual_resname_for_dense_atom_lookup = residue_names.UNK_DNA # 'DN'

    if actual_resname_for_dense_atom_lookup:
      # Get the canonical list of atoms for this specific residue type.
      # This list has <= NUM_DENSE (24) atoms.
      canonical_atoms_for_res = list(
          atom_types.DENSE_ATOM.get(actual_resname_for_dense_atom_lookup, [])
      )
      # Pad this list to NUM_DENSE with empty strings.
      padded_atom_names = canonical_atoms_for_res + [''] * (
          NUM_DENSE - len(canonical_atoms_for_res)
      )
      # Map these NUM_DENSE atom names to their indices in ATOM29_ORDER.
      # If atom name is empty or not in ATOM29_ORDER, map to 0 (dummy index).
      indices_in_atom29 = [
          (atom_types.ATOM29_ORDER.get(name, 0) if name else 0)
          for name in padded_atom_names
      ]
      restype_dense_atom_to_atom29[restype_idx] = indices_in_atom29
  return restype_dense_atom_to_atom29

RNA_AATYPE_DENSE_ATOM_TO_ATOM29 = _make_nucleic_aatype_dense_atom_to_atom29(mmcif_names.RNA_CHAIN)
DNA_AATYPE_DENSE_ATOM_TO_ATOM29 = _make_nucleic_aatype_dense_atom_to_atom29(mmcif_names.DNA_CHAIN)
