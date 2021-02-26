

# Instalar Cuda

sudo apt install gcc

sudo apt -y install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8


Driver
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.39/NVIDIA-Linux-x86_64-460.39.run
sudo sh NVIDIA-Linux-x86_64-460.39.run 

Cuda 11.2
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run
sudo sh cuda_11.2.1_460.32.03_linux.run


Al bashrc 
export PATH=/usr/local/cuda/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   

# Instalar Cudnn
Descargar cuDNN Library for Linux (x86_64)
https://developer.nvidia.com/rdp/cudnn-download



# Instalar TensorFlow 2.4
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
pip3 install --user --upgrade tensorflow-gpu
https://www.tensorflow.org/install/pip?hl=es-419#system-install

sudo apt update
sudo apt install python3-dev python3-pip python3-venv

pip3 install --user --upgrade tensorflow-gpu
python3 -c 


# Instalar code
sudo apt update; sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt install code

# Instlar git
sudo apt install git

# Instalar Cudnn

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
