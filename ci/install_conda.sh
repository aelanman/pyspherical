# This script also borrowed from github.com/RadioAstronomySoftwareGroup/pyuvsim

set -xe

if [[ ! $OS == 'macos-latest' ]]; then
  sudo apt-get update
  sudo apt-get install -y gcc g++
fi

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --name=${ENV_NAME}  python=$PYTHON --quiet
conda env update -f ci/${ENV_NAME}.yaml
source activate ${ENV_NAME}
conda list -n ${ENV_NAME}
# check that the python version matches the desired one; exit immediately if not
PYVER=`python -c "import sys; print('{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor))"`
if [[ $PYVER != $PYTHON ]]; then
  exit 1;
fi
