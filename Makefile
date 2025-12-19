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

# OPTS = --show
run: ## Run the script
# 	$(VENV_PYTHON) -m src.zone_process --image example.jpg --points points_garage.json $(OPTS)
	$(VENV_PYTHON) -m eulerian_video_magnification guitar.mp4
	$(VENV_PYTHON) -m eulerian_video_magnification baby.mp4


setup:
	$(VENV_PYTHON) -m src.polygon_picker

mesh:
	$(VENV_PYTHON) -m src.mesh_test

