#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Executable file for the project.

For encapsulation reasons, ``run`` acts as a high-level builder. Having the builder control multiple processes
that abstract the dataset analysis and/or visualisation of file(s) also makes debugging signficantly easier.

Constants:
    TESTING_FILENAME_BASE (str): Quick access to set a known, working file to test execution of package.
    TESTING_INPUT_DIR (str): Quick access to set the path to ``TESTING_FILENAME_BASE``.
    TESTING_OUTPUT_DIR (str): Quick access to output path for saving any generated figures.

Examples:
    (Here, place useful implementations of the contents of test_file.py). Note that leading symbol '>>>' includes the
    code in doctests, while '$' does not.)::

        >>> run(filename_base='1723', batch_processing=True, has_numeric_suffixes=True)

Todo:
    - Check expressions: For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi))
                                                     * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    - Convert code in file to a class-based structure.

References:
    Style guide: `Google Python Style Guide`_

Notes:
    File version
        0.3.1
    Project
        SpinChains-PythonAnalysis
    Path
        Core Files/main.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        06 Mar 2022
    IDE
        PyCharm

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""
__all__ = ['generate_filenames', 'increment_suffix', 'run']

# Standard library imports
import logging as log
import string
from typing import Optional

# Local application imports
from data_analysis import AnalyseData, PlotEigenmodes
from system_preparation import SystemSetup

# Module-level constants
TESTING_FILENAME_BASE: str = '1723'
"""Quick access to set a known, working file to test execution of package."""

TESTING_INPUT_DIR: str = '2024-08-22'
"""Quick access to set the path to TESTING_FILENAME_BASE"""

TESTING_OUTPUT_DIR: str = '2025-02-12'
"""Quick access to output path for saving any generated figures."""


def generate_filenames(system_setup: SystemSetup, filename_base: str, is_suffix_numeric: bool) -> None:
    """Helper function for extracting data from sequentially generated C++ data files.

    Currently, assume that users will edit the literals of this function before executing `main.py`.
    """
    suffix: int | str = 1 if is_suffix_numeric else 'a'

    break_conditions = ['aaa', 1000]
    while suffix != any(break_conditions):
        # 1) Update filename to find next dataset to load.
        filename = filename_base + (f'_{suffix}' if is_suffix_numeric else suffix)

        # 2) Load, process, and plot dataset. Note - currently no automatic preset available for AnalyseData().
        analyse_simulation_dataset = AnalyseData()
        analyse_simulation_dataset.import_data(
            file_descriptor=filename,
            input_dir_path=system_setup.input_dir(),
            output_dir_path=system_setup.output_dir(),
        )
        analyse_simulation_dataset.process_data()
        analyse_simulation_dataset.call_methods(
            override_function="te",
            override_method="pf",
            override_site=3501,
            early_exit=True,
            loop_function=True,
            mass_produce=True,
            interactive_mode=False
        )

        increment_suffix(suffix)


def increment_suffix(suffix: int | str) -> int | str:
    """Helper function for extracting data from sequentially generated C++ data files."""
    if isinstance(suffix, int):
        return suffix + 1

    # Convert suffix from numeric to alphabetic.
    alphabet = string.ascii_lowercase
    suffix_as_numeric = 0
    for i, char in enumerate(reversed(suffix)):
        suffix_as_numeric += (alphabet.index(char) + 1) * (26 ** i)

    # Increment to obtain suffix of next dataset.
    suffix_as_numeric += 1

    # Convert suffix back from numeric to alphabetic
    suffix = ''
    while suffix_as_numeric > 0:
        suffix_as_num, remainder = divmod(suffix_as_numeric - 1, 26)
        suffix = alphabet[remainder] + suffix

    return suffix


def run(
    filename_base: Optional[str] = None,
    batch_processing: bool = False,
    has_numeric_suffixes: bool = True,
    should_analyse_eigenvalues: bool = False
) -> None:
    """Primary access point for the package which generates a single output for a single input data file.

    Initialises all classes before calling sequentially invoking their methods.

    Args:
        filename_base:
            The base (i.e. root) filename. Defaults to accepting a user input.
        batch_processing:
            If true, will generate a single output for each file sequentially labelled according to
            ``has_numeric_suffixes``.
        has_numeric_suffixes:
            If true, processes batch files named with sequential numeric suffices (0, 1, 2, ...), else processes for
            sequential alphabetic suffixes (a, b, c, ...).
        should_analyse_eigenvalues:
            If true, extracts eigenfrequency information from dataset instead of visualising it.
    """
    log.info(f"Program start...")

    if filename_base is None:
        filename_base = str(input("Enter the unique identifier of the file: "))

    # Generate file system/structure information
    system_setup = SystemSetup()
    system_setup.detect_os(
        use_default=False,
        download_for_onedrive=False,
        download_for_pcloud=False,
        custom_input_dir_name=TESTING_INPUT_DIR,
        custom_output_dir_name=TESTING_OUTPUT_DIR
    )

    # Begin processing
    if should_analyse_eigenvalues:
        dataset2 = PlotEigenmodes(filename_base, system_setup.input_dir(), system_setup.output_dir())
        dataset2.import_eigenmodes()
        dataset2.plot_eigenmodes()  # only use this line if raw files don't need imported or converted
        return

    if batch_processing:
        generate_filenames(system_setup=system_setup,
                           filename_base=filename_base,
                           is_suffix_numeric=has_numeric_suffixes)
    else:
        dataset1 = AnalyseData()
        dataset1.import_data(
            file_descriptor=filename_base,
            input_dir_path=system_setup.input_dir(),
            output_dir_path=system_setup.output_dir(),
        )
        dataset1.process_data()
        dataset1.call_methods(
            override_method="pf",
            override_function="hd",
            override_site=40,
            early_exit=True,
            loop_function=True,
            interactive_mode=True
        )
        return

    return


if __name__ == '__main__':
    run(filename_base=TESTING_FILENAME_BASE)
    exit(0)
