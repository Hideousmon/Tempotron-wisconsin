# coding=utf-8
###################################################################################
## File Name: train_and_test.py
## Description: train and test for tempotron wisconsin
###################################################################################
from tempotronSNN import *
from tqdm import tqdm

if __name__ == '__main__':
        ## training
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        accuracy = 0
        x_train, x_test, y_train, y_test = load_data_from_wisconsin()
        train = convert_to_pulse_rate(x_train, y_train, 9, 0, 9, 1, 10, 500)
        test = convert_to_pulse_rate(x_test, y_test, 9, 0, 9, 1, 10, 500)

        #np.random.seed(0)
        #efficacies = 1.8 * np.random.random(9) - 0.50
        efficacies = 1.0 * np.random.random(9) -0.5
        print('synaptic efficacies:', efficacies, '\n')
        tempotron = Tempotron(0, 9, 2.5, efficacies)
        for n in tqdm(range(len(test))):
                inference = tempotron.test(test[n][0])
                if (test[n][1] == True) and (inference == True):
                        TP += 1
                        accuracy += 1
                if (test[n][1] == False) and (inference == True):
                        FN += 1
                if (test[n][1] == True) and (inference == False):
                        FP += 1
                if (test[n][1] == False) and (inference == False):
                        TN += 1
                        accuracy += 1
        print('TP_before:', TP/len(test))
        print('TN_before:', TN / len(test))
        print('FP_before:', FP / len(test))
        print('FN_before:', FN / len(test))
        print('accuracy_before:', accuracy / len(test))

        ## testing
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        accuracy = 0
        tempotron.train(train, 250, learning_rate=10e-5)
        print('synaptic efficacies:', tempotron.efficacies)
        for n in tqdm(range(len(test))):
                inference = tempotron.test(test[n][0])
                if (test[n][1] == True) and (inference == True):
                        TP += 1
                        accuracy += 1
                if (test[n][1] == False) and (inference == True):
                        FN += 1
                if (test[n][1] == True) and (inference == False):
                        FP += 1
                if (test[n][1] == False) and (inference == False):
                        TN += 1
                        accuracy += 1
        print('TP_after:', TP/len(test))
        print('TN_after:', TN / len(test))
        print('FP_after:', FP / len(test))
        print('FN_after:', FN / len(test))
        print('accuracy_after:', accuracy / len(test))
