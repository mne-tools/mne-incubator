#! /usr/bin/env python
from setuptools import setup

descr = """Experimental code for MEG and EEG data analysis."""

DISTNAME = 'mne-incubator'
DESCRIPTION = descr
MAINTAINER = 'Alexandre Gramfort'
MAINTAINER_EMAIL = 'alexandre.gramfort@telecom-paristech.fr'
URL = 'https://mne.tools/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-incubator'
VERSION = 'unstable'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=[
              'mne_incubator',
              'mne_incubator.preprocessing',
              'mne_incubator.connectivity',
              'mne_incubator.externals',
              'mne_incubator.externals.pacpy',
          ],
      )
