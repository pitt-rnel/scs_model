# Changelog

## [1.1.0] - 2021-10-01
### Changed
- update installation of jupyter-commons with "jupyter-minimal" optional requirements
- changed base image to jupyter-base 
## [1.0.7] - 2021-06-22
### Changed
- updates simcore service library to reduce cpu usage when extracting archives
## [1.0.6] - 2021-04-06
### Fixed
- fixed simcore.service.settings docker labels concerning RAM/CPU limits

## [1.0.4] - 2021-03-22
### Fixed
- fixed requirements

## [1.0.2] - 2021-03-22
### Changed
- updated simcore-sdk to 0.3.2

## [1.0.1] - 2020-03-09
### Fixed
- state_puller no longer causes out of memory errors causing the entire app to crash
- state puller will no longer allow the notebook to boot upon error when recovering the status
### Changed
- upgraded simcore-sdk and dependencies

## [1.0.0] - 2021-02-19
### Added
- jupyter notebook with octave and python kernels
- python kernel installs math and NEURON libraries (see [requirements.txt](requirements.txt))
- jupyterlab installs the following extensions:
  - [git extensions](https://github.com/jupyterlab/jupyterlab-git#readme)
  - [interactive matplotlib](https://github.com/matplotlib/ipympl#readme)


---
All notable changes to this service will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the release numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


<!-- Add links here -->

<!-- [1.0.0]:https://git.speag.com/oSparc/sparc-internal/-/tags/jneuron_v1.0.0  -->


<!-- HOW TO WRITE  THIS CHANGELOG

- Guiding Principles
  - Changelogs are for humans, not machines.
  - There should be an entry for every single version.
  - The same types of changes should be grouped.
  - Versions and sections should be linkable.
  - The latest version comes first.
  - The release date of each version is displayed.
  - Mention whether you follow Semantic Versioning.
  -
- Types of changes
  - Added for new features.
  - Changed for changes in existing functionality.
  - Deprecated for soon-to-be removed features.
  - Removed for now removed features.
  - Fixed for any bug fixes.
  - Security in case of vulnerabilities.

SEE https://keepachangelog.com/en/1.0.0/
-->
