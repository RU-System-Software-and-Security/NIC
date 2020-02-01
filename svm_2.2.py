import pickle
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import svm

# f2=open('./SVM/lambda_1_svm.model','rb')
# s2=f2.read()
# model=pickle.loads(s2)

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def dense(input_shape):
    model = Sequential()
    model.add(Dense(10, input_shape=(input_shape[1], )))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    files_name = ['lambda_1','lambda_2','activation_1','activation_2',
 'max_pooling2d_1','activation_3','activation_4','max_pooling2d_2','flatten_1','activation_5','activation_6','activation_7']

    for i in range(0, len(files_name)):

        name_1 = files_name[i]
        output_1 = np.load('./output/{0}_check_values.npy'.format(name_1)).reshape((60000, -1))
        input_shape_1 = output_1.shape
        model_1 = dense(input_shape_1)
        output1 = model_1.predict(output_1)

        j = i

        if files_name[j] == 'activation_6':
            name_2 = files_name[j+1]
            output2 = np.load('./output/{0}_check_values.npy'.format(name_2))

        else:
            name_2 = files_name[j+1]
            output_2 = np.load('./output/{0}_check_values.npy'.format(name_2)).reshape((60000, -1))
            input_shape_2 = output_2.shape
            model_2 = dense(input_shape_2)
            output2 = model_2.predict(output_2)

        output = np.concatenate((output1, output2), axis=1)
        print(output.shape)

        file_name = './SVM_2/{0}_{1}_output_values.npy'.format(files_name[i], files_name[j+1])
        np.save(file_name, output)

        clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.5)
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

        result_name = './SVM_2/{0}_{1}_benign_predict_result.npy'.format(files_name[i], files_name[j+1])
        np.save(result_name, predict_result)

        # decision_function
        result = clf.decision_function(output)
        result_name = './SVM_2/{0}_{1}_benign_result.npy'.format(files_name[i], files_name[j+1])
        np.save(result_name, result)

        #get adv
        adv_output_1 = np.load('./adv_output/{0}_check_values.npy'.format(name_1)).reshape((100, -1))
        adv_input_shape_1 = adv_output_1.shape
        adv_model_1 = dense(adv_input_shape_1)
        adv_output1 = adv_model_1.predict(adv_output_1)

        k = i

        if files_name[k] == 'activation_6':
            name_2 = files_name[k+1]
            adv_output2 = np.load('./adv_output/{0}_check_values.npy'.format(name_2))

        else:
            name_2 = files_name[k+1]
            adv_output_2 = np.load('./adv_output/{0}_check_values.npy'.format(name_2)).reshape((100, -1))
            adv_input_shape_2 = adv_output_2.shape
            adv_model_2 = dense(adv_input_shape_2)
            adv_output2 = adv_model_2.predict(adv_output_2)

        adv_output = np.concatenate((adv_output1, adv_output2), axis=1)
        print(adv_output.shape)

        file_name = './SVM_2/{0}_{1}_adv_output_values.npy'.format(files_name[i], files_name[k+1])
        np.save(file_name, output)

        # adv predict
        adv_predict_result = clf.predict(adv_output)
        print(adv_predict_result)
        m = 0
        for num_2 in range(len(adv_predict_result)):
            if adv_predict_result[num_2] == 1:
                m += 1
        adv_acc = m / len(adv_predict_result)
        print(adv_acc)

        result_name = './SVM_2/{0}_{1}_adv_predict_result.npy'.format(files_name[i], files_name[k+1])
        np.save(result_name, adv_predict_result)

        # adv_decision_function
        adv_result = clf.decision_function(adv_output)
        result_name = './SVM_2/{0}_{1}_adv_result.npy'.format(files_name[i], files_name[k+1])
        np.save(result_name, adv_result)

        s = pickle.dumps(clf)
        f = open('./SVM_2/{0}_{1}_svm.model'.format(files_name[i], files_name[k+1]), "wb+")
        f.write(s)
        f.close()
        print("Done_{0}_{1}\n".format(files_name[i], files_name[k+1]))







