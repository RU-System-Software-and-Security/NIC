# NIC

An adversarial example detection tool.

---
## Dependencies and acknowledgement:
1. [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo) (Included as a folder in this project)
2. [thundersvm](https://github.com/Xtra-Computing/thundersvm)
3. sklearn

---
## Quick Start

- Step 0: dependencies.

We re-use code from [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo) repo (Thank you!) to generate adv examples etc., and thus to use it, you need to install the required dependencies.

- Step 1: Get internal values

```
$ python get_output.py (adv_get_output.py)
```

The output resutls will be in folder 'output' ('adv_output').

- Step 2.0: PCA

```
$ python PCA.py
```

The outputs of some layes are too large to be used to train an osvm. In this step, we use Principal Component Analysis (PCA) to reduce dimensionality. Currently, we use 5000 as the default value. This is not necessary if you have enough computing resources.
  
- Step 2.1: VI

```
$ python svm_2.1.py
```

All of the results will be stored in 'SVM' folder.

- Step 2.2: PI

```
$ python svm_2.2.py
```

All of the results will be stored in 'SVM_2' folder.

- Step 3: Detection

```
$ python svm_3.py
```

All of the results are stored in 'SVM_3' folder.

---
## Tuning and Speedup

- We recommend [thundersvm](https://github.com/Xtra-Computing/thundersvm) to accelerate the osvm training process.
- You can change the [PCA function parameters](https://github.com/Jethro85/NIC/blob/dfa45ea2d5f5d9fc2bc69b6e9a37dff4846313a7/PCA.py#L18) based on your computing resources.
- We recommend tuning each [PI](https://github.com/Jethro85/NIC/blob/e226c8d93352055561783ffc6fd766a811f81a63/svm_2.1.py#L21-L32) and [VI](https://github.com/Jethro85/NIC/blob/e226c8d93352055561783ffc6fd766a811f81a63/svm_2.2.py#L63-L75) before tuning the final detector.
