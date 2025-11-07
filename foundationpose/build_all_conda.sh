PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Changes
echo This is the PYTHONPATH
echo $PYTHONPATH

echo This is the CONDA_PREFIX
echo $CONDA_PREFIX

echo This is the CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH

echo Is torch installed?
python -c "import torch; print(torch.__version__)"

# This is the PYTHONPATH
# /home/stois/miniconda3/envs/Any6D/lib/python3.9/site-packages
# This is the CONDA_PREFIX
# /home/stois/miniconda3/envs/Any6D
# This is the CMAKE_PREFIX_PATH
# /home/stois/miniconda3/envs/Any6D/lib/python3.9/site-packages/pybind11/share/cmake/pybind11


# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
#python -m pip install -e .
python -m pip install .

cd ${PROJ_ROOT}
