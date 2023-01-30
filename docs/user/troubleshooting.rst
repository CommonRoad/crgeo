.. _troubleshooting:

Troubleshooting
======

This page serves as a pool of bug reports and solutions.

Dependencies
-----------

1. `traci.exceptions.TraCIException: MoveToXY vehicle should obtain: edgeID, lane, x, y, angle and optionally keepRouteFlag`
   - This exception is due to version mismatch of traci and sumo. For instance, it's important to have traci=1.12.0 and all sumo components in version 1.12.0. Make sure to do `sudo add-apt-repository ppa:sumo/stable` `sudo apt-get update` `sudo apt-get install sumo sumo-tools sumo-doc` to install the up-to-date sumo.
2. `OSError: libcusparse.so.11: cannot open shared object file: No such file or directory`
   - This error is related to cuda install. If you have libcusparse.so.11 but torch-geometric can't find it, add the folder contains it to LD_LIBRARY_PATH by adding e.g. `export LD_LIBRARY_PATH="your_miniconda_path/lib:$LD_LIBRARY_PATH"` in `\~/.bashrc` as discussed in https://github.com/pyg-team/pytorch_geometric/issues/2040. If you can't find it in either locally or in your virtual env, the reason can be mismatched version of nvcc and the cudatoolkit version you installed together with pytorch.
3. `OSError: libcudart.so.11.0: cannot open shared object file: No such file or directory` when having a torch CPU-only installation
   - This error is also related to cuda install. If you have no GPU available, make sure you installed ALL torch packages (torch, torchvision, torchaudio, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, torch-geometric) in their respective CPU variants. The man pages of the packages give advice on how to select the right package variant and version for the given CUDA version, if any.
   - If all torch packages are installed as CPU versions, torch will no longer search for the non-existent libcudart file.
4. `ImportError: Library "GLU" not found.` `sudo apt install freeglut3-dev freeglut3 libgl1-mesa-dev libglu1-mesa-dev libxext-dev libxt-dev` `sudo apt install python3-opengl libgl1-mesa-glx libglu1-mesa`
5. `libGL error: MESA-LOADER: failed to open iris: /home/maurice/anaconda3/envs/commonroad-isometric-embedding/bin/../lib/libstdc++.so.6: version GLIBCXX_3.4.29' not found (required by /usr/lib/dri/iris_dri.so) (search paths /usr/lib/dri, suffix \\\\\\\\\\\\\\\_dri) libGL error: failed to load driver: iris`

- Do:
  - `rm anaconda3/envs/$env-name/lib/libstdc++.so`
  - `rm anaconda3/envs/$env-name/lib/libstdc++.so.6`
  - `ln -s /usr/lib64/libstdc++.so.6.0.29 anaconda3/envs/$env-name/lib/libstdc++.so`
  - `ln -s /usr/lib64/libstdc++.so.6.0.29 anaconda3/envs/$env-name/lib/libstdc++.so.6`
- Reason for issue: Anaconda links to its own libstdc++.so.6.0.28 which is incompatible with the system libstdc++.
- Version 1.5.27 of pyglet works well while 2.0.0 can't display the render.