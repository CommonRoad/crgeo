# This script was last tested on a fresh Ubuntu 22.04.2 LTS installation on 2023/04/28
# STEP 1
# Before running the script download the Anaconda installer from `https://www.anaconda.com/products/distribution`
# Then run it with `bash Anaconda3-2021.11-Linux-x86_64.sh` (replace anaconda version with your specific .sh file name)
# Answer the prompt 'Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]' with yes

# STEP 2
# Clone the commonroad-geometric repository with e.g. 'git clone git@gitlab.lrz.de:cps/commonroad-geometric.git'

# STEP 3
# Run this script from the root of the repository, e.g after running 'cd commonroad-geometric' 
# Run the script by sourcing it, i.e. `source scripts/create-crgeo-cpu-environment.sh`
# Please see `https://superuser.com/questions/176783/what-is-the-difference-between-executing-a-bash-script-vs-sourcing-it` for more details

# Install dependencies necessary for building commonroad-drivability-checker (https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker/-/blob/master/build.sh#L227)
# This command requires superuser privileges and will prompt you for your password
sudo apt-get install build-essential cmake git git-lfs wget unzip libboost-dev libboost-thread-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev libomp-dev
BUILD_JOBS=8

# Make sure that you have a clean start:
ENV_NAME="crgeo-3.10-cpu"

# Warn and exit if the environment `ENV_NAME` exists:
if conda info --envs | grep $ENV_NAME 
then
    echo "Warning: Environment ${ENV_NAME} exists already, before proceeding and running this script again:"
    echo "  1. Deactivate the environment ${ENV_NAME} with 'conda deactivate'"
    echo "  2. Remove the environment ${ENV_NAME} with 'conda env remove --name ${ENV_NAME}'"
    return 1
fi

# Install Git LFS and pull
git lfs install
git lfs pull

# Create conda development environment (see environment.yml for installation of pip dependencies)
conda env create -f environment_cpu.yml

# Fix issue with "libGL error: failed to load driver: swrast" (see: https://askubuntu.com/a/1405450)
rm ~/anaconda3/envs/$ENV_NAME/lib/libstdc++.so.6
