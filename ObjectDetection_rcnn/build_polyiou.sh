sudo apt-get install swig

git clone git@github.com:CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit
swig -c++ -python polyiou.i
python3 setup.py build_ext --inplace