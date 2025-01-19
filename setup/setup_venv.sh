cd ..

git submodule init
git submodule update
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

cd accelerated_features 

git submodule init
git submodule update

cd ..
echo "Packages installed, proceed with dataset installation!\nConfigure datasets.yaml and run download_dataset.py"
