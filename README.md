# Action recognition datasets tools

Tools to preprocess action recognition datasets.

* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)


## Setup

For Ubuntu 18.04 install the following packages:

```bash
sudo apt install python3-dev python3-virtualenv virtualenvwrapper p7zip-full p7zip-rar ffmpeg
```

Setup virtualenvwrapper adding the follwing to `.bashrc`:

```bash
# virtualenvwrapper
if [ `id -u` != '0' ]; then
  export WORKON_HOME=$HOME/.virtualenvs
  source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
  export PIP_VIRTUALENV_BASE=$WORKON_HOME
  export PIP_RESPECT_VIRTUALENV=true
fi
```

Create an enviroment:

```bash
mkvirtualenv dstools-ar -p /usr/bin/python3.6
```

Add the repository to the venv path:
```bash
add2virtualenv .
```

Install packages:
```bash
pip install -r requirements.txt
```

Create a [dotenv](https://pypi.org/project/python-dotenv/) file `dstools-ar/.env` with the `DATASETS` variable pointing to the datasets container directory.

```bash
DATASETS_DIR=/path/to/datasets_dir
```


## Run

Each dataset has a `snakefile`, to run the whole preprocess enter 
to its directory and execute:

```bash
cd ucf101
snakemake
```

Check `snakefile` for the available steps. There is also a `stats.py` script to plot statistics for the dataset.
