# NIC

This repo is an example of Paper: 
['NIC: Detecting Adversarial Samples with Neural Network Invariant Checking.'](https://www.cs.purdue.edu/homes/ma229/papers/NDSS19.pdf)  
Shiqing Ma, Yingqi Liu, Guanhong Tao, Wen-Chuan Lee, Xiangyu Zhang  
Proceedings of the 26th Network and Distributed System Security Symposium (NDSS 2019)

---
## Thanks To:
1. [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo): The dataset, adversarial test examples and trained carlini model we use come from their repo. 
2. [thundersvm](https://github.com/Xtra-Computing/thundersvm): We use their Fast SVM Library on GPUs to accelerate our one class svm (osvm) training process. 

---
## Files Introduction
- 'get_output.py' and 'adv_get_output.py' in the EvadeML-Zoo package are files used to generate every layer's outputs of the model for benign dataset and adversarial test dataset. Please install the corresponding environment according to  [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo) repo. All of the results should be stored in 'output' and 'adv_output' packages, respectively.  
- 'PCA.py' can be used to reduce dimensionality of specific layer's output.   
- 'svm_2.1.py' is used to train osvms according to every layer's output. All of the results, include the trained osvm, predict results, 'decision_function' results are stored in 'SVM' package.  
- 'svm_2.2.py' is used to train osvms according to every two conjunctive layers' output. All of the results, include the trained osvm, predict results, 'decision_function' results are stored in 'SVM_2' package.  
- 'svm_3.py' is used to train the final osvm using the 'decision_function' results from 'svm_2.1' and 'svm_2.2'. All of the results, include the trained osvm, predict results, decision_function results are stored in 'SVM_3' package.  

---
## Quick Start
**Please follow the steps below, run the files one by one.**   
- Firstly, install the corresponding environment according to  [EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo) repo. Run 'get_output.py' and 'adv_get_output.py' to get every layer's outputs of the model for benign dataset and adversarial test dataset. Please put results in 'output' and 'adv_output' packages, respectively.  
- Secondly, after flatten, the outputs of convolution layes are too large to be used to train a osvm. In this step, we use Principal component analysis (PCA) to reduce dimensionality of the flattened convolution layes' outputs to (60000, 5000) and (100, 5000), for benign dataset and adversarial testing dataset respectively. If your server is good enough, you can skip this step.  
**For the following steps, you can also use [thundersvm](https://github.com/Xtra-Computing/thundersvm) to accelerate the osvm training process.**   
- Thirdly, run 'svm_2.1.py' to train osvms according to every layer's output. All of the results will be stored in 'SVM' package.  
- Fourthly, run 'svm_2.2.py' to train osvms according to every two conjunctive layers' output. All of the results will be stored in 'SVM_2' package.  
- In the end, run 'svm_3.py' to train the final osvm using the 'decision_function' results from 'svm_2.1' and 'svm_2.2'. All of the results are stored in 'SVM_3' package.  

---
## Results:
According to our experiment, after running 'svm_3.py', we can get an osvm whose benign dataset predict accuracy can arrive at: 98% and adversarial dataset misclassification rate is only 1%. 


