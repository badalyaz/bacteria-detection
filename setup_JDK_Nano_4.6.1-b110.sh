#!/bin/bash

#######  ############################################  #######
#######  THIS SCRIPT IS FOR NVIDIA-JETPACK 4.6.1-b110  #######
#######             RUN THIS WITH SUDO SU              #######
#######  ############################################  #######

path1='export PATH=/usr/local/cuda/bin:${PATH}'
path2='export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}'
path3='export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1"'
path4='export BUILD_VERSION=0.9.0'
for comm in 'apt update' 'apt install -y python3-pip' 'echo "$path1" >> ~/.bashrc' 'echo "$path2" >> ~/.bashrc' '/usr/bin/pip3 install Cython' 'echo "$path1"' 'echo "$path2"' '/usr/bin/pip3 install pycuda==2019.1.2' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' 'apt update' 'apt -y install python-scipy libopenblas-base libpng-dev gfortran libopenmpi-dev liblapack-dev libatlas-base-dev' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install --upgrade protobuf==3.19.4' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install --upgrade pandas' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install --upgrade scipy' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' 'python3 -m pip install --upgrade pip' 'python3 -m pip install --upgrade Pillow' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install sklearn scikit-image' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install pyyaml' '/usr/bin/pip3 install -U future psutil dataclasses typing-extensions tqdm seaborn' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' 'wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl' '/usr/bin/pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' 'apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' 'git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision' 'cd torchvision' 'echo "$path4" >> ~/.bashrc' 'echo $path4' 'python3 setup.py install --user' 'cd ..  #running torch from torchvision will fail' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install notebook' 'echo "$path3" >> ~/.bashrc' 'echo $path3' '/usr/bin/pip3 uninstall -y numpy' '/usr/bin/pip3 install -U numpy==1.19.4' '/usr/bin/pip3 install psutil' 'sudo -H pip install -U jetson-stats' 'pip install torchmetrics' 'pip install scikit-learn' 'echo "$path5" >> ~/.bashrc' 'echo $path5' 'echo "All installations had finished successfully!"';
do
	clear
	echo -e "\033[32mCommand:" $comm "\033[0m" && \
	eval $comm;

	es="$?";
	if [ $es -ne "0" ]
	then
		echo -e "\033[31m\033[6mError in command\033[0m" $comm;
		break;
	fi
done

if [ $es -ne "0" ]
then
	echo "setup ended unsuccessfully"
else
	echo "setup ended successfully"
fi
