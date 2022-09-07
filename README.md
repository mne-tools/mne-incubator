# MNE-Python incubator

[Questions? Use the MNE Forum!](https://mne.discourse.group)

This is a repository for experimental code for new techniques and ideas that may or may not make it into the official MNE-Python package. All of this is considered as work-in-progress.

## How this works
Contributions are welcome in the form of pull requests. Once the implementation of a piece of functionality is considered to be bug free and properly documented (both API docs and an example script), it can be incorporated into the master branch. Once it is in the master branch, it can be used by users of MNE-Python while the functionality awaits verification in a scientific manner (for new techniques, this means a paper). After the functionality has been verified, it can be integrated into MNE-Python.

## Code organization

The directory structure of this repository mirrors the one of MNE-Python. When you add new functionality, place it in the location where you would expect it to end up in the MNE-Python repository. Your code may depend on the development version of MNE-Python and other submodules of MNE-incubator. At least one example script should be placed in the `mne_incubator/examples` folder.
