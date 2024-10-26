# Fuzzy-DDI

![CLAT-DTA](https://github.com/zhanggit001/CLAT-DTA/blob/master/CLAT-DTA/src/CLAT-DTA.png))

**This is the data and code for our paper** `CLAT-DTA: A Collaborative Attention-based Approach for Drug-traget Binding Affinity Prediction`.

## Prerequisites

Make sure your local environment has the following installed:

* `pytorch == 2.4.1`
* `python == 3.9 or python == 3.8   `

## Datastes

We provide the dataset in the [data](data/) folder.

| Data | Description |
| --- | --- |
| [Davis](data/drugbank/) | A drug-target interaction network between  68 drugs with 442 targets interactions. |
| [KIBA](data/TWOSIDES/) | A drug-drug interaction network betweeen 2,111 drugs with 229 targets interactions. |

## Documentation

```
--data
  │--davis_train.csv
  |--davis_test.csv
  │--kiba_train.csv
  |--kiba_test.csv
--src
  │--README.md
  │--models
    |--DAT.py
    |--transformer.py
    |--layers
  |--utils

--training.py
--validating.py
--test.py
```

## Train

Training script example: `train.py`

## Validating

The trained model will be automatically stored under the folder `./saved_models`. The model name will be `DAT_best_davis.pkl`.

To test a trained model, you can use the following file:

```
validating.py
```



## Authors

** Yu Zhang** @github.com/zhanggit001 
**Email:** zhangyyy2023@163.com 
**Site:** [GitHub](https://github.com/zhanggit001)
