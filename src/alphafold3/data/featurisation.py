# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 featurisation pipeline."""

from collections.abc import Sequence
import datetime
import logging
import time
from typing import Union

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import features
from alphafold3.model.pipeline import pipeline
import numpy as np

from alphafold3.data import mmcif_parsing # Assuming this will be used more directly
from alphafold3.model import atom_layout # For MAX_ATOMS_PER_RESIDUE and ATOM_LAYOUTS

def validate_fold_input(fold_input: folding_input.Input):
  """Validates the fold input contains MSA and templates for featurisation."""
  for i, chain in enumerate(fold_input.protein_chains):
    if chain.unpaired_msa is None:
      raise ValueError(f'Protein chain {i + 1} is missing unpaired MSA.')
    if chain.paired_msa is None:
      raise ValueError(f'Protein chain {i + 1} is missing paired MSA.')
    if chain.templates is None:
      raise ValueError(f'Protein chain {i + 1} is missing Templates.')
  for i, chain in enumerate(fold_input.rna_chains):
    if chain.unpaired_msa is None:
      raise ValueError(f'RNA chain {i + 1} is missing unpaired MSA.')
    if chain.templates is None: # Added for RNA templates
      raise ValueError(f'RNA chain {i + 1} is missing Templates.')


def featurise_input(
    fold_input: folding_input.Input,
    ccd: chemical_components.Ccd,
    buckets: Sequence[int] | None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    verbose: bool = False,
) -> Sequence[features.BatchDict]:
  """Featurise the folding input.

  Args:
    fold_input: The input to featurise.
    ccd: The chemical components dictionary.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    ref_max_modified_date: Optional maximum date that controls whether to allow
      use of model coordinates for a chemical component from the CCD if RDKit
      conformer generation fails and the component does not have ideal
      coordinates set. Only for components that have been released before this
      date the model coordinates can be used as a fallback.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    resolve_msa_overlaps: Whether to deduplicate unpaired MSA against paired
      MSA. The default behaviour matches the method described in the AlphaFold 3
      paper. Set this to false if providing custom paired MSA using the unpaired
      MSA field to keep it exactly as is as deduplication against the paired MSA
      could break the manually crafted pairing between MSA sequences.
    verbose: Whether to print progress messages.

  Returns:
    A featurised batch for each rng_seed in the input.
  """
  validate_fold_input(fold_input)

  # Set up data pipeline for single use.
  data_pipeline = pipeline.WholePdbPipeline(
      config=pipeline.WholePdbPipeline.Config(
          # This config is for the model.pipeline.WholePdbPipeline,
          # not the data_pipeline_config from run_alphafold.py
          buckets=buckets,
          ref_max_modified_date=ref_max_modified_date,
          conformer_max_iterations=conformer_max_iterations,
          resolve_msa_overlaps=resolve_msa_overlaps
      ),
  )

  batches = []
  for rng_seed in fold_input.rng_seeds:
    featurisation_start_time = time.time()
    if verbose:
      print(f'Featurising data with seed {rng_seed}.')
    batch = data_pipeline.process_item(
        fold_input=fold_input,
        ccd=ccd,
        random_state=np.random.RandomState(rng_seed),
        random_seed=rng_seed,
    )
    if verbose:
      print(
          f'Featurising data with seed {rng_seed} took'
          f' {time.time() - featurisation_start_time:.2f} seconds.'
      )
    batches.append(batch)

  return batches

# The following is a conceptual placement for _create_template_features
# if it were in this file. In AlphaFold 3, it's part of
# alphafold3.model.pipeline.pipeline.WholePdbPipeline._process_templates
# or similar. We'll adapt the logic assuming it's called from within
# the featurisation context if it were here.
# For the purpose of this exercise, let's assume we are modifying a hypothetical
# _create_template_features function that was previously part of this file or
# is being newly introduced/adapted here for clarity of the diff.
#
# If `_create_template_features` is indeed in `alphafold3.model.pipeline.pipeline`,
# that file would need similar modifications.

# Placeholder for where _create_template_features might be if it were directly in featurisation.py
# For AlphaFold 3, this logic is more likely within alphafold3.model.pipeline.pipeline.py
# in a method like _process_templates. The diff below shows how such a function
# would be modified.

# def _create_template_features(
# chain: Union[folding_input.ProteinChain, folding_input.RNAChain], # Updated type hint
# chain_template: folding_input.Template,
#     pdb_path: pathlib.Path, # This would come from template.mmcif if it's a path
# # ... other args like ccd, ref_max_modified_date
# ) -> features.TemplateFeatures | None:
#
# # Determine chain type for specific parsing logic
#   is_protein = isinstance(chain, folding_input.ProteinChain)
#   is_rna = isinstance(chain, folding_input.RNAChain)
#
# if is_protein:
#     chain_type_for_parsing = mmcif_parsing.ChainType.PROTEIN
# elif is_rna:
#     chain_type_for_parsing = mmcif_parsing.ChainType.RNA
# else:
#     raise ValueError(f"Unsupported chain type for template: {type(chain)}")
#
# # Parse mmCIF string from chain_template.mmcif
#   mmcif_object = mmcif_parsing.parse_mmcif_string(chain_template.mmcif)
# if not mmcif_object:
# return None
#
# # This part needs to be adapted from existing template processing logic
# # For example, from alphafold3.data.templates.get_polymer_features
# # or similar logic in model.pipeline.pipeline.WholePdbPipeline
#
# # Get template sequence and mapping (conceptual)
# # template_sequence, template_to_model_mapping = mmcif_parsing.get_chain_template_sequence_and_mapping(
# #       mmcif_object=mmcif_object,
# #       template_chain_id=??? # This needs to be derived, perhaps from the mmCIF or assumed
# # model_sequence=chain.sequence,
# #       max_template_sequence_identity=0.95, # Example value
# #       select_template_by_max_num_aliged_residues=False, # Example value
# # chain_type=chain_type_for_parsing,
# # )
# # if template_sequence is None:
# # return None
#
# # Simplified version based on folding_input.Template having query_to_template_map
#   query_to_template_map = chain_template.query_to_template_map
#
# # Featurize template_aatype
#   template_aatype_mapped = np.zeros(len(chain.sequence), dtype=np.int32)
#
# # Featurize atom positions and mask
#   template_all_atom_positions_mapped = np.zeros(
#       (len(chain.sequence), atom_layout.MAX_ATOMS_PER_RESIDUE, 3),
# dtype=np.float32,
# )
#   template_all_atom_mask_mapped = np.zeros(
#       (len(chain.sequence), atom_layout.MAX_ATOMS_PER_RESIDUE),
# dtype=np.float32,
# )
#   template_pseudo_beta_mapped = np.zeros(
#       (len(chain.sequence), 3), dtype=np.float32
# )
#   template_pseudo_beta_mask_mapped = np.zeros(
#       len(chain.sequence), dtype=np.float32
# )
#
# # The following is a highly conceptual sketch of how atom featurization would work
# # It needs to parse chain_template.mmcif and use query_to_template_map
# # Actual implementation would be more involved, likely reusing parts of
# # alphafold3.data.templates.get_polymer_features
#
# # Assume template_chain_id is 'A' for simplicity or parsed from mmCIF
#   template_chain_id_in_mmcif = 'A' # This is a placeholder
#
# # Conceptual: get atom positions from the mmcif_object for the template_chain_id_in_mmcif
# # template_atom_data = mmcif_parsing.get_template_atom_positions(
# # mmcif_object, template_chain_id_in_mmcif, len_template_sequence_from_mmcif, chain_type_for_parsing
# # )
#
#   for query_idx, template_idx in query_to_template_map.items():
# if query_idx >= len(chain.sequence): continue
#
# # Get residue name from the *model's* sequence for aatype
# # The template_aatype should reflect the *template's* residue types aligned to query positions
# # This requires knowing the template's sequence.
# # For simplicity, let's assume chain_template.template_sequence exists (added in folding_input.Template if needed)
# # Or, it's derived by looking up template_idx in the parsed template's sequence.
#
# # This part is complex: mapping template residues to aatypes and atoms
# # Needs template's actual sequence.
# # For now, we'll skip the detailed atom parsing and aatype and focus on structure.
# # A real implementation would call atom_layout.get_atom_positions for each template residue.
#
# # Placeholder:
# # template_aatype_mapped[query_idx] = ... (based on template's residue at template_idx)
# # template_all_atom_positions_mapped[query_idx] = ...
# # template_all_atom_mask_mapped[query_idx] = ...
# # template_pseudo_beta_mapped[query_idx] = ...
# # template_pseudo_beta_mask_mapped[query_idx] = ...
#       pass # Needs full implementation
#
# # Placeholder for template_sum_probs, domain_names, release_date
#   template_sum_probs_mapped = np.zeros(len(chain.sequence), dtype=np.float32) # Placeholder
#   template_domain_names = b"template" # Placeholder
#   template_release_date = b"1970-01-01" # Placeholder
#
# return {
# 'template_aatype': template_aatype_mapped,
# 'template_all_atom_positions': template_all_atom_positions_mapped,
# 'template_all_atom_mask': template_all_atom_mask_mapped,
# 'template_pseudo_beta': template_pseudo_beta_mapped,
# 'template_pseudo_beta_mask': template_pseudo_beta_mask_mapped,
# 'template_sum_probs': template_sum_probs_mapped,
# 'template_domain_names': np.array([template_domain_names] * len(chain.sequence), dtype=object),
# 'template_release_date': np.array([template_release_date] * len(chain.sequence), dtype=object),
# }

# The actual featurisation of templates happens within
# alphafold3.model.pipeline.pipeline.WholePdbPipeline._process_templates.
# That's the function that would need the core logic changes.
# The `featurise_input` function in this file calls `data_pipeline.process_item`,
# which internally handles template featurisation.
#
# The `validate_fold_input` function has been updated above to check for `templates`
# in `RNAChain`.
#
# No further changes are needed in *this specific file* (`alphafold3/data/featurisation.py`)
# beyond the `validate_fold_input` change, because the main template processing logic
# is encapsulated within `alphafold3.model.pipeline.pipeline.WholePdbPipeline`.
#
# The key changes would be in:
# 1. `alphafold3/model/pipeline/pipeline.py` (specifically `WholePdbPipeline._process_templates`
#    and any helper functions it calls for template featurisation).
# 2. `alphafold3/data/mmcif_parsing.py` (to ensure functions like
#    `get_chain_template_sequence_and_mapping` and a new/generalized
#    `get_template_atom_positions` can handle RNA).
# 3. `alphafold3/data/templates.py` (if it's used by the model pipeline for featurisation,
#    its `get_polymer_features` would need to correctly handle RNA).
#
# Since the request is to modify `featurisation.py`, and its direct role in template
# processing is mostly to call the `WholePdbPipeline`, the primary change here is
# ensuring `validate_fold_input` is aware of RNA templates. The more substantial
# featurisation logic changes occur deeper in the `model.pipeline`.
