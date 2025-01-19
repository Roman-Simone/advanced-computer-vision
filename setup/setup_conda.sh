cd ..

git submodule init
git submodule update
conda create -n xfeat python=3.8
conda activate xfeat
pip install -r requirements.txt
pip install -e .

cd accelerated_features 

git submodule init
git submodule update

cd ..
echo "Packages installed, proceed with dataset installation!\nConfigure datasets.yaml and run download_dataset.py"
