# PartialAAMs

**PartialAAMs** is a library designed to provide a benchmarking framework for evaluating effectiveness of  Partial Atom-Atom Mappings (AAMs) extension. It simplifies the generation, manipulation, and testing of AAMs for chemical reactions.


## Installation

Follow the steps below to set up the environment and install the library:

1. Create a conda environment with Python 3.11.10:
   ```bash
   conda create -n partialaams python=3.11.10 -y
   ```

2. Activate the environment:
   ```bash
   conda activate partialaams
   ```

3. Clone the GranMapache repository:
   ```bash
   git clone https://github.com/MarcosLaffitte/GranMapache.git
   ```

4. Navigate to the GranMapache directory:
   ```bash
   cd GranMapache
   ```

5. Install GranMapache:
   ```bash
   pip install .
   ```

6. Return to the original directory:
   ```bash
   cd ..
   ```

7. Remove the GranMapache directory:
   ```bash
   rm -rf GranMapache
   ```

8. Install additional requirements:
   ```bash
   pip install -r requirements.txt
   ```

# Usage

### Example Workflow

```python
from partialaams.gm_expand import gm_extend_from_graph
from partialaams.ilp_expand import extend_aam_from_graph
from aamutils.utils import smiles_to_graph

# Define reaction SMILES
rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"

# Convert SMILES to graphs
G, H = smiles_to_graph(rsmi)

# Extend the Partial AAM and generate reaction SMILES by gm
extended_smiles_gm = gm_extend_from_graph(G, H)
print("Extended Reaction SMILES:", extended_smiles_gm)

# Extend the Partial AAM and generate reaction SMILES by ILP solver
extended_smiles_ilp = extend_aam_from_graph(G, H)
print("Extended Reaction SMILES:", extended_smiles_ilp)
>>"[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"

```