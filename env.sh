mkdir tfapp
sudo apt install python-virtualenv
apt-get install python-tk
virtualenv --system-site-packages tfapp
cd tfapp
source bin/activate
pip install tensorflow
pip install numpy
pip install matplotlib
pip install jupyter
pip install scikit-image
pip install librosa
pip install nltk
pip install keras
pip install git+https://github.com/tflearn/tflearn.git

