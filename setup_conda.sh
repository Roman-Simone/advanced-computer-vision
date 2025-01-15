git submodule init
git submodule update
conda create -n xfeat python=3.8
conda activate xfeat
pip install -r requirements.txt
pip install -e .

mkdir data

cd accelerated_features 

git submodule init
git submodule update

python3 -m modules.dataset.download --megadepth-1500 --download_dir ../data
python3 -m modules.dataset.download --scannet-1500 --download_dir ../data

cd ..
echo "Packages installed, proceed with dataset installation!\nConfigure datasets.yaml and run download_dataset.py"
