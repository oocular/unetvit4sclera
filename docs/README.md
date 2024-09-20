# Docs

## Install uv (An extremely fast Python package manager)
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
## Create venv
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment:
#uv venv 2nd_env --python 3.13 #create with a diff python version

deactivate
#rm -rf 2nd_env # to remove 2nd_env
```

## Install python package deps
```
uv pip install --editable . # Install the package in editable mode
uv pip install .[test]
uv pip install .[learning]
#uv pip uninstall ready

uv pip install PyQt6
```


