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

Enter to the repository diretory and vreate an enviroment:

```bash
cd dstools-ar
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

Finally, its necessary to define `DATASETS_DIR` environment variable pointing to the datasets container directory. It can be defined permanently in a [dotenv](https://pypi.org/project/python-dotenv/) file `.env` in the root directory of the repository with the following content:

```bash
DATASETS_DIR=/path/to/datasets_dir
```

also, definition can be done at command level:

```bash
DATASETS_DIR=/path/to/datasets_dir snakemake
```

Note that command level definition overrides dotenv one, this allows using diffents `DATASETS_DIR` directories per execution.


## Run

Each dataset has a `snakefile`, to run the whole preprocess enter 
to its directory and execute:

```bash
cd ucf101
snakemake
```

Check `snakefile` for the available steps. There is also a `stats.py` script to plot statistics for the dataset.
