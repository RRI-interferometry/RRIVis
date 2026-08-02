# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
# Warnings are errors. Tier 8C fixed all sixteen warnings the build carried and
# installed this default so `ci.yml`'s `make -C docs html` inherits the gate
# without a workflow change. `--keep-going` reports every warning in one build
# rather than stopping at the first, so a contributor fixes them in one pass.
# Override on the command line (`make -C docs html SPHINXOPTS=`) only to
# inspect a broken build; never to land one.
SPHINXOPTS    ?= -W --keep-going
# Keep the documented `make -C docs html` command reproducible by running
# Sphinx from the checked-in Pixi environment.
SPHINXBUILD   ?= pixi run -- python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
clean:
	rm -rf $(BUILDDIR)/*

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
