# Model Compra Comigo
![version](badges/version.svg)
![coverage](badges/coverage.svg)

An implementation of a model for steel price prediction. It contains experiments around time series and machine learning applied to time series, simplified dockerfied processes to train, retrain and predict using models, and other related functionalities .

# Run

If you run the docker commands, you only need docker or other compatible container runtime installed (you can skip installation steps, since they are run in docker images)

## Installation
We recommend using a virtual environment to run locally.
Example:
```
conda create -n model-compra-comigo python=3.9 -y && \
conda activate model-compra-comigo
```
Install anaconda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Install only required packages
To install required packages to experiment/predict:
```
make install
```

### Installation (development)
To install packages related to experiments (train/predicted and others are included):
```
make install-dev
```

> **Warning**
> 
> You may need to install development requirements to run examples or experiments.

## Predict Api
TODO impl (precisa de revisão por ter mudado a assinatura de alguns métodos, recolocarei em commits futuros)

## Predict Cli
TODO docs

## Docker
TODO impl (precisa de revisão por ter mudado a assinatura de alguns métodos, recolocarei em commits futuros)

# Docs
TODO regerar as docs (agora que alterei as assinaturas, antiga ficou obsoleta). Fazer ao final dos outros commits.

<!-- 
## Train

### Train Locally
To run locally:
```
```

### Train Containerized
To run with docker:
```
``` -->

## Test
To run unit tests and behaviour tests:
```
make test
```

To only run unit tests:
```
make unit-tests
```

To only run behaviour tests:
```
make behaviour-tests
```

## Docs
TODO
To build docs:
```
make build-docs
```
You can access the docs in the docs folder (html) .

## Utils

### Check
To check if package is ready for a commit:
```
make check
```

### Audit
To run static analysis and formatting:
```
make audit
```

> **Warning**
> 
> You may need to install poetry beforehand to generate the requirements .
> 
> Example:
> 
> ``` pip install poetry ```

### Generate badges
```
make badge
```

### Clean
To clean the repository up:
```
make clean
```
