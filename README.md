# casper
Implementation of Cascade Anytime Size Prediction via self-Exciting Regression model (CASPER)  

The model is described in paper:
```shell
[Anytime Information Cascade Popularity Prediction via Self-Exciting Processes](insert doi link)  
Xi Zhang, Akshay Aravamudan, Georgios C. Anagnostopoulos
In Proceedings of the 39th International Conference on Machine Learning, volume X of Proceedings of Machine Learning Research, pages xxx–xxx. PMLR, 17–23 Jul 2022. 
```

## Basic Usage

### Requirements

The code was tested with `python 3.9.12`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name casper python=3.9.12

# activate environment
conda activate casper

# install requirment packages
pip install -r requirements.txt
```

### Run the code
```shell

# generate information cascades
python main.py

```
