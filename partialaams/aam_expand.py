from partialaams.ilp_expand import extend_aam_from_rsmi
from partialaams.extender import Extender
from synkit.Graph.ITS.its_expand import ITSExpand

try:
    from partialaams.gm_expand import gm_extend_aam_from_rsmi
except Exception:
    gm_extend_aam_from_rsmi = None


def partial_aam_extension_from_smiles(rsmi: str, method: str = "ilp"):
    """
    Extends atom mappings (AAMs) in a reaction represented in RSMI format.

    Parameters:
    - rsmi (str): The Reaction SMILES (RSMI) string that contains atom mappings.
    - method (str): The method to use for atom mapping extension. Options are:
                     - 'ilp': Uses ILP-based atom mapping extension (default).
                     - 'gm': Uses GM-based atom mapping extension.
                     - 'syn': Uses SynAAM-based atom mapping extension.

    Returns:
    - str: The SMILES string with extended atom mappings, based on the chosen method.

    Example:
    - rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"
    - extended_smiles = partial_aam_extension_from_smiles(rsmi, method="gm")
    """
    if method == "ilp":
        # ILP-based extension using aamutils
        return extend_aam_from_rsmi(rsmi)
    elif method == "gm":
        # GM-based extension using partialaams
        return gm_extend_aam_from_rsmi(rsmi)
    elif method == "syn":
        # SynAAM-based extension using synutility
        p = ITSExpand()
        return p.expand_aam_with_its(rsmi)
    elif method == "extend":
        ext = Extender()
        return ext.fit(rsmi)
    elif method == "extend_g":
        ext = Extender()
        return ext.fit(rsmi, use_gm=True)
    else:
        # Raise an error if the provided method is not valid
        raise ValueError("Invalid method. Choose from 'ilp', 'gm', or 'syn'.")
