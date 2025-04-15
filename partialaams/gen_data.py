from rdkit import Chem
from typing import Optional

from synkit.IO.graph_to_mol import GraphToMol
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc

from synkit.IO.chem_converter import smiles_to_graph


def _get_partial_aam(smart) -> str:
    """
    Generate a partial atom-atom mapping SMILES string from
    a reactant graph and an ITS graph.

    This function performs the following steps:
      1. Deep copies the ITS graph to avoid modifying the original.
      2. Iterates through each node in the copied ITS graph; if a node is not present
         in the reactant graph (rc), its 'atom_map' attribute is set to 0.
      3. Decomposes the modified graph into two subgraphs using `its_decompose`
         (one for retained mapping and one for partial mapping).
      4. Converts the resulting subgraphs into RDKit molecule objects with hydrogen count
         consideration using GraphToMol.
      5. Generates SMILES strings from both molecules and concatenates them in the format:
         "retained_smiles>>partial_smiles".

    Parameters:
      rc (nx.Graph): The reactant graph.
      its (nx.Graph): The integrated transition state (ITS) graph to decompose.

    Returns:
      str: A SMILES string in the format "retained_smiles>>partial_smiles".

    Raises:
      RuntimeError: If an error occurs during graph decomposition, molecule conversion,
                    or SMILES generation.
    """
    r, p = smart.split(">>")
    r_graph = smiles_to_graph(r, use_index_as_atom_map=True)
    p_graph = smiles_to_graph(p, use_index_as_atom_map=True)
    its = ITSConstruction.ITSGraph(r_graph, p_graph)
    rc = get_rc(its)
    # Create a set of reactant nodes for efficient membership checking.
    rc_nodes = set(rc.nodes())

    # Update the 'atom_map' attribute for nodes not present in the reactant graph.
    for node, _ in r_graph.nodes(data=True):
        if node not in rc_nodes:
            r_graph.nodes[node]["atom_map"] = 0
    # Update the 'atom_map' attribute for nodes not present in the reactant graph.
    for node, _ in p_graph.nodes(data=True):
        if node not in rc_nodes:
            p_graph.nodes[node]["atom_map"] = 0

    converter = GraphToMol()
    try:
        retained_mol = converter.graph_to_mol(r_graph, use_h_count=True)
        partial_mol = converter.graph_to_mol(p_graph, use_h_count=True)
    except Exception as e:
        raise RuntimeError(
            "Error converting graphs to RDKit molecules: " + str(e)
        ) from e

    # Generate SMILES strings for both molecules.
    try:
        retained_smiles = Chem.MolToSmiles(retained_mol)
        partial_smiles = Chem.MolToSmiles(partial_mol)
    except Exception as e:
        raise RuntimeError(
            "Error generating SMILES from RDKit molecules: " + str(e)
        ) from e

    # Return the concatenated SMILES string representing the mapping.
    return f"{retained_smiles}>>{partial_smiles}"


def _remove_small_smiles(smiles: str) -> str:
    """
    Given a SMILES string, returns the SMILES string corresponding to its largest fragment.

    This function performs the following steps:
      1. Converts the input SMILES string to an RDKit molecule without immediate sanitization.
      2. Attempts to sanitize the molecule.
      3. Extracts all fragments of the molecule.
      4. Selects the largest fragment based on heavy atom count.
      5. Optionally sanitizes the largest fragment and converts it back to a canonical SMILES string.

    Parameters:
      smiles (str): The input SMILES string.

    Returns:
      str: The canonical SMILES of the largest fragment.

    Raises:
      ValueError: If the input SMILES is invalid, sanitization fails, or no fragments are found.
    """
    # Convert the SMILES string to an RDKit molecule without sanitization.
    mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Sanitize the molecule to ensure chemical consistency.
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"Sanitization failed for SMILES '{smiles}': {e}")

    # Get all fragments as individual molecule objects.
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        raise ValueError(f"No fragments found for SMILES '{smiles}'.")

    # Select the fragment with the maximum number of heavy atoms.
    largest_fragment = max(fragments, key=lambda m: m.GetNumHeavyAtoms())

    # Optional: Sanitize the largest fragment to ensure it is valid.
    try:
        Chem.SanitizeMol(largest_fragment)
    except Exception as e:
        raise ValueError(f"Sanitization failed for the largest fragment: {e}")

    # Convert the largest fragment back to a canonical SMILES string and return.
    return Chem.MolToSmiles(largest_fragment)


def _create_unbalanced_aam(rsmi: str, side: str = "right") -> str:
    """
    Processes a reaction SMILES (rsmi) by removing small fragments from one or both sides, 
    generating an unbalanced atom-atom mapping (AAM) reaction SMILES.

    The input reaction SMILES should be in the format "reactant_smiles>>product_smiles". 
    The parameter `side` controls which side(s) to process:
      - "left"  : Process and clean the reactant (left) fragment.
      - "right" : Process and clean the product (right) fragment.
      - "both"  : Process and clean both the reactant and product fragments.

    This function is useful when a reaction SMILES contains multiple fragments on one side
    and only the largest (dominant) fragment is of interest for further analysis.

    Parameters:
      rsmi (str): The reaction SMILES string separated by ">>".
      side (str): Which side to process. Must be one of "left", "right", or "both" (default is "right").

    Returns:
      str: A new reaction SMILES string in the format "reactant>>product" with the specified side(s) processed.

    Raises:
      ValueError: If the reaction SMILES does not contain exactly one ">>" separator,
                  if an invalid value for side is provided,
                  or if an error occurs while processing the specified side(s).
    """
    # Validate that the reaction SMILES contains exactly one '>>'
    parts = rsmi.split(">>")
    if len(parts) != 2:
        raise ValueError(f"Invalid reaction SMILES format: {rsmi}. Expected exactly one '>>' separator.")

    # Strip extraneous whitespace.
    r, p = [part.strip() for part in parts]

    # Ensure the side parameter is valid.
    side_lower = side.lower()
    if side_lower not in ("left", "right", "both"):
        raise ValueError(f"Invalid side value '{side}'. Expected one of 'left', 'right', or 'both'.")

    # Process the reactant side if requested.
    if side_lower in ("left", "both"):
        try:
            r = _remove_small_smiles(r)
        except Exception as e:
            raise ValueError(f"Error processing reactant SMILES: {e}")

    # Process the product side if requested.
    if side_lower in ("right", "both"):
        try:
            p = _remove_small_smiles(p)
        except Exception as e:
            raise ValueError(f"Error processing product SMILES: {e}")

    # Return the new reaction SMILES string.
    return f"{r}>>{p}"
