#!/usr/bin/env python3
"""
partialaams.__main__
CLI for extending partial atom–atom mappings (AAM) from reaction SMILES.

Examples
--------
Single SMILES (positional):
    python -m partialaams "[CH3][CH:1]=[CH2:2]>>..."

Single SMILES via -i / --input:
    python -m partialaams -i "[CH3][CH:1]=[CH2:2]>>..."

Batch file:
    python -m partialaams -i reactions.txt -o extended.txt
"""
from __future__ import annotations

import argparse
import os
import sys

# Relative import for `python -m partialaams`, absolute as fallback for script use
try:
    from .aam_expand import partial_aam_extension_from_smiles  # type: ignore
except ImportError:  # pragma: no cover
    from partialaams.aam_expand import partial_aam_extension_from_smiles  # noqa: E402

SUPPORTED_METHODS: list[str] = ["gm", "ilp", "syn", "extend", "extend_g"]


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="partialaams",
        description="Extend partial AAMs directly from reaction SMILES.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Either a positional SMILES *or* -i/--input (file **or** SMILES)
    p.add_argument(
        "rsmi",
        nargs="?",
        help="Reaction SMILES string (optional if -i is given).",
    )
    p.add_argument(
        "-i",
        "--input",
        metavar="PATH|SMILES",
        help=(
            "Path to a file with one reaction SMILES per line, "
            "or a single SMILES string if the path does not exist."
        ),
    )

    p.add_argument(
        "-m",
        "--method",
        choices=SUPPORTED_METHODS,
        default="gm",
        help="Extension method to use (default: gm).",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write extended SMILES to file (default: stdout).",
    )
    p.add_argument(
        "-l",
        "--list-methods",
        action="store_true",
        help="List supported extension methods and exit.",
    )
    return p


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def list_methods_and_exit() -> None:
    print("Supported extension methods:")
    for m in SUPPORTED_METHODS:
        print(f"  - {m}")
    sys.exit(0)


def iter_smiles(args: argparse.Namespace):
    """Yield reaction SMILES strings according to CLI arguments."""
    if args.input is not None:
        # Treat as file if it exists; otherwise as literal SMILES
        if os.path.exists(args.input):
            try:
                with open(args.input, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield line
            except OSError as exc:
                sys.stderr.write(f"Failed to read '{args.input}': {exc}\n")
                sys.exit(1)
        else:
            yield args.input.strip()
    elif args.rsmi is not None:
        yield args.rsmi.strip()
    else:
        sys.stderr.write(
            "Error: provide a reaction SMILES (positional) or use -i/--input.\n"
        )
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        list_methods_and_exit()

    # Prepare output sink
    if args.output:
        try:
            out_fh = open(args.output, "w", encoding="utf-8")
            close_out = True
        except OSError as exc:
            sys.stderr.write(f"Failed to open '{args.output}' for writing: {exc}\n")
            sys.exit(1)
    else:
        out_fh = sys.stdout
        close_out = False

    # Process SMILES lines
    for rsmi in iter_smiles(args):
        try:
            extended = partial_aam_extension_from_smiles(rsmi, method=args.method)
            out_fh.write(extended + "\n")
        except ValueError as exc:
            sys.stderr.write(f"Error processing '{rsmi}': {exc}\n")

    if close_out:
        out_fh.close()


if __name__ == "__main__":  # pragma: no cover
    main()
