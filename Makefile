VENV			= .venv
VENV_BIN		= $(VENV)/bin
VENV_PYTHON		= $(VENV_BIN)/python3
SYSTEM_PYTHON	= $(or $(shell which python3), $(shell which python))
DOCKER = sudo docker


help:  ## Display this help output
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

$(VENV_PYTHON):
	$(SYSTEM_PYTHON) -m venv --system-site-packages $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip


local_packages: $(VENV_PYTHON) ## Install required packages
	$(VENV_PYTHON) -m pip install --no-cache-dir --verbose -r requirements.txt

hooks: $(VENV) ## Install git pre-commit hooks
	$(VENV_PYTHON) -m pip install pre-commit
	$(VENV_BIN)/pre-commit install

OPTS = --gui
run: ## Run the script
	wget --no-clobber https://github.com/chthomos/video-media-samples/raw/refs/heads/master/big-buck-bunny-480p-30sec.mp4
	$(VENV_PYTHON) -m src.run --input big-buck-bunny-480p-30sec.mp4 $(OPTS)
