

### Installation of Dependencies

- Conda

 Download anaconda with python 2.7 support from the [anaconda website](https://docs.continuum.io/anaconda/) (avalaible here: [Anaconda2-4.3.1.sh](https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh)), then install with

 ```bash
    bash Anaconda2-4.3.1-Linux-x86_64.sh
 ```

- Create a python 2.7 environment in conda

```bash
  conda create -n py27 python=2.7 anaconda
```

- Source conda py2.7

```bash
  source activate py27
```

- Opencv Conda with HighGUI Supoport. See [this](http://stackoverflow.com/questions/24400935/how-could-we-install-opencv-on-anaconda).


```bash
  conda install -c loopbio ffmpeg-feature gtk2-feature opencv
```
<!-- Install from the conda 3rd party repos

```bash
  conda install -c https://conda.binstar.org/menpo opencv
```

You may like to add the menpo site permanently

```bash
  conda config --add channels menpo
```

**UPDATE**

-  Install opencv from sources

  - Grap the opencv 2.4.13 tag from my github profile.

  - Temporarily move anaconda to a directory that bash recognizes e,g, ~/Downloads

  - Then compile opencv cmake as follows

  ```bash
      cmake -DCMAKE_BUILD_TYPE=RELEASE  \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_TBB=ON \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -DWITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_TIFF=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D PYTHON_EXECUTABLE=~/anaconda2/envs/py27/python \
      -D PYTHON_LIBRARY=~/anaconda2/envs/py27/lib/python2.7 \
      -D PYTHON_INCLUDE_DIR=~/anaconda2/envs/py27/include/python2.7 \
      -D INSTALL_C_EXAMPLES=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GTK=ON \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_OPENNI=ON \
      -D WITH_OPENMP=ON \
      -D WITH_OPENCL=ON \
      -D CUDA_GENERATION=Kepler ..
  ```

  We seem good to go. Note the prefix of where the python libraries and include headers are stored in conda environment.

  This might be different on your system. The important thing is to install python headers and libs in the correct anaconda environment

- Next, we'll build the intermediate files and then install the executables. Do this:

```bash
  make -j$(nproc) && sudo make install
```
-->

**UPDATE **

  ```bash
    pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp27-none-linux_x86_64.whl

    pip install torchvision
  ```

  That too fails. remove ~/anaconda2 from path and install pytorch with pip and see what happens

- Install ros barebones package

  ```bash
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  ```

  - Set up ros keys

  ```bash
    sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
  ```

  ```bash
    sudo apt-get update
  ```

  Then install ros bare bones

  ```bash
    sudo apt-get install ros-indigo-ros-base
  ```

- Pytorch

  - with Cuda support

  ```bash
    conda install pytorch torchvision cuda80 -c soumith
  ```

  - without cuda support

   please folow the instructions [here](http://pytorch.org/)

### FAQs

- Python environments mixed up

Explicitly add the conda python2.7 executable path as an alias into your bash, e.g.,

`alias py27='~/anaconda2/envs/py27/bin/python2'`

and then run the code as

```bash
  py27 main_detect.py
```
