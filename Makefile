# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

sample_data:
	@python -c "import mne; mne.datasets.sample.data_path(verbose=True);"

testing_data:
	@python -c "import mne; mne.datasets.testing.data_path(verbose=True);"

test: in
	rm -f .coverage
	$(NOSETESTS) -a '!ultra_slow_test' mne_incubator

test-fast: in
	rm -f .coverage
	$(NOSETESTS) -a '!slow_test' mne_incubator

test-full: in
	rm -f .coverage
	$(NOSETESTS) mne_incubator

test-no-network: in
	sudo unshare -n -- sh -c 'MNE_SKIP_NETWORK_TESTS=1 nosetests mne_incubator'

test-no-testing-data: in
	@MNE_SKIP_TESTING_DATASET_TESTS=true \
	$(NOSETESTS) mne_incubator

test-no-sample-with-coverage: in testing_data
	rm -rf coverage .coverage
	$(NOSETESTS) --with-coverage --cover-package=mne_incubator --cover-html --cover-html-dir=coverage

test-doc: sample_data testing_data
	$(NOSETESTS) --with-doctest --doctest-tests --doctest-extension=rst doc/

test-coverage: testing_data
	rm -rf coverage .coverage
	$(NOSETESTS) --with-coverage --cover-package=mne_incubator --cover-html --cover-html-dir=coverage

test-profile: testing_data
	$(NOSETESTS) --with-profile --profile-stats-file stats.pf mne_incubator
	hotshot2dot stats.pf | dot -Tpng -o profile.png

test-mem: in testing_data
	ulimit -v 1097152 && $(NOSETESTS)

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne_incubator; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

codespell:
	# The *.fif had to be there twice to be properly ignored (!)
	codespell.py -w -i 3 -S="*.fif,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.coverage,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii" ./dictionary.txt -r .
