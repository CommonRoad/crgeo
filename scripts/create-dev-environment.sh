# This script was last tested on a fresh Ubuntu 18.04.6 LTS installation in 05/2022
# Before running the script, install python-pip via `sudo apt install python-pip` and anaconda via downloading its .sh file from `https://www.anaconda.com/products/distribution` and running `bash Anaconda3-2021.11-Linux-x86_64.sh` (replace anaconda version with your specific ..sh file name)
# Then download (only) this script into your future working directory that should contain the commonroad-geometric repository (e.g. `/home/youruser/git/`, i.e. `~/git/`)
# Don't execute the script right away, but go to l. 32-48 and adapt the torch installation versions to your specific hardware!
# After selecting the right torch versions, finally run the script by sourcing it,
# i.e. `source create-dev-environment.sh`
# Please see `https://superuser.com/questions/176783/what-is-the-difference-between-executing-a-bash-script-vs-sourcing-it` for more details

# Make sure that you have a clean start:
ENV_NAME="commonroad-3.8"
BUILD_JOBS=8

# Remove the environment `commonroad-3.8` if it exists:
if conda info --envs | grep $ENV_NAME 
then
    echo "Removing existing ${ENV_NAME}..."
    conda deactivate
    conda env remove -y --name $ENV_NAME 
    echo "Done!"
fi

# Delete pip cache, otherwise you might have some version conflicts during installation:
pip cache purge

# Create & activate a conda environment for CommonRoad development:
conda create --name $ENV_NAME 'python==3.8.*' -y
conda activate commonroad-3.8

# Update pip:
pip install --upgrade pip --no-input

# Install PyTorch via pip:
# Open `https://pytorch.org/get-started/locally/` to select the right installation version in the selection matrix.
# Here, the LTS version 1.8.2 of PyTorch is used as it is compatible with the code base.
# The right version for you is heavily machine-dependent (CUDA version available etc.), if you have CUDA11.1 on your Nvidia GPU, the command will be:
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html --no-input
# If you don't have a GPU, replace the command above with the CPU-only variant:
# pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html --no-input

# Install PyTorch-Geometric via pip:
# Open `https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html` to select the right installation version in the selection matrix.
# Make sure your PyTorch-Geometric version matches your installed PyTorch version from above.
# Here, the selected packagee versions are selected to be compatible with PyTorch 1.8.2 and the code base.
# For PyTorch 1.8.2+cu111, the PyTorch-Geometric versions from https://data.pyg.org/whl/torch-1.8.1%2Bcu111.html (i.e. 1.8.1) have to be used.
# The right version for you is heavily machine-dependent (cuda version available etc.), if you have CUDA11.1 on your NVidia GPU, the command will be:
pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html --no-input
# If you don't have a GPU, replace the command above with the CPU-only variant:
# pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.2 -f https://data.pyg.org/whl/torch-1.8.1+cpu.html --no-input

# Installing via conda does not work because of package version conflicts:
# conda install pyg -c pyg -c conda-forge

# Install SUMO via the Python package
# `https://sumo.dlr.de/docs/Downloads.php#python_packages_virtual_environments`:
pip install --upgrade eclipse-sumo --no-input

# Register file path of the SUMO Python package inside the ~/.bashrc (to be tested on a fresh system):
echo 'export SUMO_HOME=\"$(which sumo)\"' >> ~/.bashrc
# Please make sure that the path of the SUMO Python package has ended up in your ~/.bashrc, e.g. something like ```export SUMO_HOME="/home/YOURUSER/anaconda3/envs/commonroad-3.8/lib/python3.8/site-packages/sumo/"```

# Install CommonRoad Geometric:
# Clone repo from `https://gitlab.lrz.de/cps/commonroad-geometric`:
git clone https://gitlab.lrz.de/cps/commonroad-geometric.git
pushd commonroad-geometric
# Install in editable mode:
# Install with sumo
pip install --editable .[sumo] --no-input --user
# Install without sumo
pip install --editable . --no-input --user
popd
