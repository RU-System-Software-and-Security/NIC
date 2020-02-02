# NIC

---
## Dependencies and acknoledgement:
1. [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo) (Included as a folder in this project)
2. [thundersvm](https://github.com/Xtra-Computing/thundersvm)
3. sklearn

---
## Quick Start
**Please follow the steps below, run the files one by one.**   
- Firstly, install the corresponding environment according to  [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo) repo. Run 'get_output.py' and 'adv_get_output.py' to get every layer's outputs of the model for benign dataset and adversarial test dataset. Please put results in 'output' and 'adv_output' packages, respectively.  

- Step 1: Get internal values

```bash
$ python get_output.py (adv_get_output.py(
```

The output resutls will be in folder 'output' ('adv_output').

- Step 2: Training

- Step 2.0: PCA

```bash
$ python PCA.py
```

The outputs of some layes are too large to be used to train a osvm. In this step, we use Principal component analysis (PCA) to reduce dimensionality. Currently, we use 5000 as the default value. This is not necessary if you have enough computing resources.
  
- Step 2.1: VI

```bash
$ python svm_2.1.py
```

All of the results will be stored in 'SVM' folder.

- Step 2.2: PI

```bash
$ python svm_2.2.py
```

- Step 3: Detection

```bash
$ python svm_3.py
```

All of the results are stored in 'SVM_3' folder.

---
## Tuning and Speedup

- We recommand [thundersvm](https://github.com/Xtra-Computing/thundersvm) to accelerate the osvm training process.
- You can change the PCA function parameters based on computing resources.
- We recommand tuning each PI and VI before tuning the final detector



