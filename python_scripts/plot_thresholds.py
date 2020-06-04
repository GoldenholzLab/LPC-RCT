import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
import csv


def determine_hyperparameter_based_file_paths(model_storage_dir,
                                              model_storage_folder_name,
                                              model_type,
                                              dropout_rate=None,
                                              CNNet_1D_num_filters=None,
                                              CNNet_1D_kernel_size=None,
                                              Leaky_ReLU_neg_slope=None):

    if(model_type == 'single_perceptron'):

        hyperparameter_based_deepRCT_model_file_name = 'only_one_model'

    elif(model_type == '3_layer'):

        hyperparameter_based_deepRCT_model_file_name = 'dropout_' + str(int(100*dropout_rate)) + '%'

    elif(model_type == 'simple'):

        hyperparameter_based_deepRCT_model_file_name = \
            'dropout_'           + str(int(100*dropout_rate))         + \
            '%_num_filters_'     +     str(CNNet_1D_num_filters)      + \
            '_kernel_size_'      +     str(CNNet_1D_kernel_size)      + \
            '_leaky_relu_slope_' + str(int(10*Leaky_ReLU_neg_slope))

    elif(model_type == 'intermediate'):

        hyperparameter_based_deepRCT_model_file_name = \
            'dropout_'           + str(int(100*dropout_rate))         + \
            '%_num_filters_'     +     str(CNNet_1D_num_filters)      + \
            '_kernel_size_'      +     str(CNNet_1D_kernel_size)      + \
            '_leaky_relu_slope_' + str(int(10*Leaky_ReLU_neg_slope))

    elif(model_type == 'complicated'):
        
        hyperparameter_based_deepRCT_model_file_name = \
            'dropout_'           + str(int(100*dropout_rate))         + \
            '%_num_filters_'     +     str(CNNet_1D_num_filters)      + \
            '_kernel_size_'      +     str(CNNet_1D_kernel_size)      + \
            '_leaky_relu_slope_' + str(int(10*Leaky_ReLU_neg_slope))

    else:
        
        raise ValueError('The \'model_name\' parameter for \'DL_algo.py\' needs to be one of the following: \n\n' + 
                         'single_perceptron\n3_layer\nsimple\nintermediate\ncomplicated \n')
    
    results_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_results.json'

    thresholds_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_thresholds.json'

    return [results_file_path, 
            thresholds_file_path]


def plot_LPC_thresholds(NV_model):

    model_storage_dir = os.getcwd()

    if(NV_model == 'one'):
        NV_model_model_storage_folder_name = \
            'NV_model_one_weekly_regular_level_16_keras_models'
    elif(NV_model == 'two'):
        NV_model_model_storage_folder_name = \
            'NV_model_two_weekly_regular_level_16_keras_models'
    else:
        raise ValueError('uninterpretable NV model: ' + str(NV_model))

    [results_file_path, 
     thresholds_file_path] = \
        determine_hyperparameter_based_file_paths(model_storage_dir,
                                                  NV_model_model_storage_folder_name,
                                                  'single_perceptron')

    with open(results_file_path, 'r') as json_file:
        data = json.load(json_file)
        num_patients_per_trial_arm_array = np.array(data[0])

    with open(thresholds_file_path, 'r') as json_file:
        thresholds_array = np.array(json.load(json_file))

    plt.figure()
    plt.plot(2*num_patients_per_trial_arm_array, thresholds_array)
    plt.xticks(np.arange(40, 460 + 40, 40))
    plt.ylim([-0.05, 1.05])
    plt.xlabel('number of patients per trial')
    plt.ylabel('threshold value')
    plt.title('NV model ' + NV_model + ' LPC thresholds')
    plt.savefig('NV model ' + NV_model + ' LPC thresholds.png')
    
    with open('NV model ' + NV_model + ' LPC thresholds.csv', 'w+', newline='') as csv_file:

        threshold_writer = \
            csv.writer(csv_file, 
                       delimiter=' ',
                       quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)

        #for threshold in thresholds_array:
        threshold_writer.writerow(thresholds_array)


if(__name__=='__main__'):

    NV_model = sys.argv[1]
    plot_LPC_thresholds(NV_model)
    