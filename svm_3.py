import numpy as np
from sklearn import svm
import pickle

if __name__ == '__main__':
    files_name = ['lambda_1', 'lambda_2', 'activation_1', 'activation_2','max_pooling2d_1', 'activation_3',
                  'activation_4', 'max_pooling2d_2', 'flatten_1', 'activation_5','activation_6', 'activation_7']

    # benign results
    result1 = np.array([])
    for num in range(0, len(files_name)):
        i = files_name[num]
        result_name = './SVM/{0}_benign_result.npy'.format(i)
        output_1 = np.load(result_name)

        if result1.size == 0:
            result1 = output_1
        else:
            result1 = np.concatenate((result1, output_1), axis=1)
    print(result1.shape)

    result2 = np.array([])
    for num in range(0, len(files_name) - 1):
        i = num
        result_name = './SVM_2/{0}_{1}_benign_result.npy'.format(files_name[i], files_name[i+1])
        output_2 = np.load(result_name).reshape(-1, 1)

        if result2.size == 0:
            result2 = output_2
        else:
            result2 = np.concatenate((result2, output_2), axis=1)
    print(result2.shape)

    result = np.concatenate((result1, result2), axis=1)
    print(result.shape)

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
    clf.fit(result)
    print('Done\n')

    # benign predict
    predict_result = clf.predict(result)
    print(predict_result)
    m = 0
    for i in range(len(predict_result)):
        if predict_result[i] == 1:
            m += 1
    acc = m/len(predict_result)
    print(acc)
    result_name = './SVM_3/final_benign_predict_result.npy'
    np.save(result_name, predict_result)

    # benign decision
    final_result = clf.decision_function(result)
    result_name = './SVM_3/final_benign_result.npy'
    np.save(result_name, final_result)

    # adv results
    adv_result1 = np.array([])
    for num in range(0, len(files_name)):
        i = files_name[num]
        result_name = './SVM/{0}_adv_result.npy'.format(i)
        adv_output_1 = np.load(result_name)

        if adv_result1.size == 0:
            adv_result1 = adv_output_1
        else:
            adv_result1 = np.concatenate((adv_result1, adv_output_1), axis=1)
    print(adv_result1.shape)

    adv_result2 = np.array([])
    for num in range(0, len(files_name) - 1):
        i = num
        result_name = './SVM_2/{0}_{1}_adv_result.npy'.format(files_name[i], files_name[i+1])
        adv_output_2 = np.load(result_name).reshape(-1, 1)

        if adv_result2.size == 0:
            adv_result2 = adv_output_2
        else:
            adv_result2 = np.concatenate((adv_result2, adv_output_2), axis=1)
    print(adv_result2.shape)

    adv_result = np.concatenate((adv_result1, adv_result2), axis=1)
    print(adv_result.shape)


    # adv predict
    adv_predict_result = clf.predict(adv_result)
    print(adv_predict_result)
    m = 0
    for i in range(len(adv_predict_result)):
        if adv_predict_result[i] == 1:
            m += 1
    acc = m/len(adv_predict_result)
    print(acc)
    result_name = './SVM_3/final_adv_predict_result.npy'
    np.save(result_name, adv_predict_result)

    # adv decision
    adv_final_result = clf.decision_function(result)
    result_name = './SVM_3/final_adv_result.npy'
    np.save(result_name, adv_final_result)

    s = pickle.dumps(clf)
    f = open('./SVM_3/final_svm.model', "wb+")
    f.write(s)
    f.close()
    print("Done\n")



