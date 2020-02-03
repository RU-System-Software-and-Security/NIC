import numpy as np
from thundersvm import OneClassSVM
import pickle


if __name__ == '__main__':
    files_name = ['lambda_1', 'lambda_2', 'activation_1', 'activation_2',
                  'max_pooling2d_1', 'activation_3', 'activation_4', 'max_pooling2d_2', 'flatten_1', 'activation_5',
                  'activation_6', 'activation_7']

    # for num in range(2, len(files_name)):
    for num in range(5, 6):
    # for num in range(0, 1):
        i = files_name[num]
        # i = 'activation_7'
        output = np.load('./output/{0}_check_values.npy'.format(i))
        print('h layer', i, output.shape)
        output = output.reshape((60000, -1))
        print(output.shape)

        '''
        We recommend tuning each PI before tuning the final detector. 
        
        What we have done is try to make the classify accuracy of benign inputs as high as possible, and make the misclassification rate of adversarial pictures as low as possible. 
        The parameters we use to train the osvms are as follows:
        
        When i = 0 and 1, we use nu=0.01 and gamma=0.5; 
        When i = 2 to 6, we use nu=0.1 and gamma=0.1; We don't have enough computing resources. If you have, you can try nu=0.01 and gamma=0.5
        When i = 7 and 8, we use nu=0.01 and gamma=1;
        When i = 9 to 11, we use nu=0.1 and gamma=0.1;
        
        '''
        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(output)

        # benign predict
        predict_result = clf.predict(output)
        print(predict_result)
        m = 0
        for num_1 in range(len(predict_result)):
            if predict_result[num_1] == 1:
                m += 1
        acc = m / len(predict_result)
        print(acc)

        result_name = './SVM/{0}_benign_predict_result.npy'.format(i)
        np.save(result_name, predict_result)

        # decision_function
        result = clf.decision_function(output)
        result_name = './SVM/{0}_benign_result.npy'.format(i)
        np.save(result_name, result)

        # adv predict
        adv_output = np.load('./adv_output/{0}_check_values.npy'.format(i)).reshape((100, -1))
        print('h layer', i, adv_output.shape)

        adv_predict_result = clf.predict(adv_output)
        print(adv_predict_result)
        m = 0
        for num_2 in range(len(adv_predict_result)):
            if adv_predict_result[num_2] == 1:
                m += 1
        adv_acc = m / len(adv_predict_result)
        print(adv_acc)

        adv_result_name = './SVM/{0}_adv_predict_result.npy'.format(i)
        np.save(adv_result_name, adv_predict_result)

        # adv_decision_function
        adv_result = clf.decision_function(adv_output)
        adv_result_name = './SVM/{0}_adv_result.npy'.format(i)
        np.save(adv_result_name, adv_result)

        clf.save_to_file('SVM/{0}_svm.model'.format(i))


    







