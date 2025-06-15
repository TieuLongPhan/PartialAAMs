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
## Usagebenchmark

### Python Interface
Extend a partial atom-atom mapping (AAM) directly from reaction SMILES. Choose one of five strategies:

```python
from partialaams.aam_expand import partial_aam_extension_from_smiles

# Define your (partial) reaction SMILES
rsmi = "[CH3][CH:1]=[CH2:2].[H:3][H:4]>>[CH3][CH:1]([H:3])[CH2:2][H:4]"

# Expected fully-mapped SMILES
expected = (
    "[CH2:1]=[CH:2][CH3:3].[H:4][H:5]>>"
    "[CH2:1]([CH:2]([CH3:3])[H:5])[H:4]"
)

# Try all five extension methods:
for method in ("ilp", "gm", "syn", "extend", "extend_g"):
    result = partial_aam_extension_from_smiles(rsmi, method=method)
    print(f"{method:8} →", result)
    # You can validate with AAMValidator, e.g.:
    # assert AAMValidator.smiles_check(result, expected)

```
### Supported methods

- **`gm`**  Graph-matching extension. Please refer to [https://github.com/MarcosLaffitte/GranMapache](https://github.com/MarcosLaffitte/GranMapache) 
- **`ilp`**  ILP-based extension  
- **`syn`**  Gluing Graph extension  
- **`extend`**  Color reorder extension  
- **`extend_g`**  Color reorder extension using gm isomorphism

### Command-Line Interface

Once you’ve installed `partialaams` (e.g. via `pip install .` in your project root), you can run the CLI with:

```bash
python -m partialaams [OPTIONS] [RSMI]
```

### Options

| Flag                   | Description                                                                                                                   | Default  |
|:-----------------------|:------------------------------------------------------------------------------------------------------------------------------|:---------|
| `-i`, `--input`        | Path to a file with one reaction SMILES per line, **or** a single SMILES string if the file does not exist.                   | —        |
| `RSMI`                 | Reaction SMILES string (partial AAM). Use this instead of `-i` when processing a single reaction directly.                    | —        |
| `-m`, `--method`       | Extension strategy. One of: `gm`, `ilp`, `syn`, `extend`, `extend_g`.                                                        | `gm`     |
| `-o`, `--output`       | File path to write the extended SMILES to. If omitted, results stream to **stdout**.                                         | stdout   |
| `-l`, `--list-methods` | List all supported extension methods and exit.                                                                                | —        |

### Examples

```bash
$ python -m partialaams -i "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"
>> [Cl:1][CH2:5][CH2:4][CH3:3].[N:2]>>[Cl:1].[N:2][CH2:5][CH2:4][CH3:3]
```


## Benchmarking

```bash
$ python benchmarking.py
```

## Publication

[Extension of Partial Atom-to-Atom Maps: Uniqueness and Algorithms]()


### Citation

TODO


## Contributing
- [Marcos E. González Laffitte](https://github.com/MarcosLaffitte)
- [Tieu-Long Phan](https://tieulongphan.github.io/)


## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

This project has received funding from the European Unions Horizon Europe Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No 101072930 ([TACsy](https://tacsy.eu/) -- Training Alliance for Computational)