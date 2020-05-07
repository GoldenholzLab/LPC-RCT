import keras.models as models
import numpy as np
import time
import json
import sys
import os


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

    model_storage_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '.h5'

    thresholds_storage_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_thresholds.json'
    
    results_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_results.json'

    testing_time_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_runtimes.txt'
    
    return [model_storage_file_path,
            thresholds_storage_file_path,
            results_file_path,
            testing_time_file_path]


def load_data(testing_data_dir,
              testing_data_file_name):

    testing_data_file_path = testing_data_dir + '/' + testing_data_file_name + '.json'
    with open(testing_data_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    num_patients_per_trial_arm_array                    = np.array(data[0])
    MPC_stat_power_array                                = np.array(data[1])
    MPC_type_1_error_array                              = np.array(data[2])
    true_drug_placebo_arm_hists_over_all_trial_sizes    = np.array(data[3])
    true_drug_treatment_arm_hists_over_all_trial_sizes  = np.array(data[4])
    false_drug_placebo_arm_hists_over_all_trial_sizes   = np.array(data[5])
    false_drug_treatment_arm_hists_over_all_trial_sizes = np.array(data[6])

    return [num_patients_per_trial_arm_array,
            MPC_stat_power_array,
            MPC_type_1_error_array,
            true_drug_placebo_arm_hists_over_all_trial_sizes,
            true_drug_treatment_arm_hists_over_all_trial_sizes,
            false_drug_placebo_arm_hists_over_all_trial_sizes,
            false_drug_treatment_arm_hists_over_all_trial_sizes]


def evaluate_model_over_testing_set_data(true_drug_placebo_arm_hists_over_all_trial_sizes,
                                         true_drug_treatment_arm_hists_over_all_trial_sizes,
                                         false_drug_placebo_arm_hists_over_all_trial_sizes,
                                         false_drug_treatment_arm_hists_over_all_trial_sizes,
                                         deepRCT_model,
                                         threshold_array):

    [num_trial_sizes, 
     num_samples_per_trial_size, 
     num_bins, _] = \
         true_drug_placebo_arm_hists_over_all_trial_sizes.shape

    deepRCT_model_stat_power_array   = np.zeros(num_trial_sizes)
    deepRCT_model_type_1_error_array = np.zeros(num_trial_sizes)

    for trial_size_index in range(num_trial_sizes):

        deepRCT_model_true_drug_prob_array = \
            np.squeeze(deepRCT_model.predict([true_drug_placebo_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :],
                                              true_drug_treatment_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :]]))
        
        deepRCT_model_false_drug_prob_array = \
            np.squeeze(deepRCT_model.predict([false_drug_placebo_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :],
                                              false_drug_treatment_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :]]))
        
        deepRCT_model_stat_power_array[trial_size_index] = \
            np.sum(deepRCT_model_true_drug_prob_array > threshold_array[trial_size_index])/num_samples_per_trial_size
        
        deepRCT_model_type_1_error_array[trial_size_index] = \
            np.sum(deepRCT_model_false_drug_prob_array > threshold_array[trial_size_index])/num_samples_per_trial_size
    
    return [deepRCT_model_stat_power_array,
            deepRCT_model_type_1_error_array]


def store_results(num_patients_per_trial_arm_array,
                  MPC_stat_power_array,
                  MPC_type_1_error_array,
                  deepRCT_model_stat_power_array,
                  deepRCT_model_type_1_error_array,
                  results_file_path,
                  testing_time_file_path,
                  start_time_in_seconds):

    data = []
    data.append(num_patients_per_trial_arm_array.tolist())
    data.append(MPC_stat_power_array.tolist())
    data.append(MPC_type_1_error_array.tolist())
    data.append(deepRCT_model_stat_power_array.tolist())
    data.append(deepRCT_model_type_1_error_array.tolist())

    with open(results_file_path, 'w+') as json_file:
        json.dump(data, json_file)
    
    stop_time_in_seconds = time.time()
    total_runtime_in_minutes_str = str( np.round((stop_time_in_seconds - start_time_in_seconds)/60, 3) ) + ' minutes of testing'
    
    with open(testing_time_file_path, 'a') as text_file:
        text_file.write('\n' + total_runtime_in_minutes_str)


def get_inputs():

    print(sys.argv)

    model_storage_folder_name = sys.argv[1]
    testing_data_file_name = sys.argv[2]
    model_type = sys.argv[3]

    dropout_rate = None
    CNNet_1D_num_filters = None
    CNNet_1D_kernel_size = None
    Leaky_ReLU_neg_slope = None

    if(len(sys.argv) > 4):
        dropout_rate = float(sys.argv[4])

    if(len(sys.argv) > 5):
        CNNet_1D_num_filters =   int(sys.argv[5])
        CNNet_1D_kernel_size =   int(sys.argv[6])
        Leaky_ReLU_neg_slope = float(sys.argv[7])

    #================================================================================================#

    model_storage_dir = os.getcwd()
    testing_data_dir = os.getcwd()

    return [model_storage_dir,
            testing_data_dir,
            model_storage_folder_name,
            testing_data_file_name,
            model_type,
            dropout_rate,
            CNNet_1D_num_filters,
            CNNet_1D_kernel_size,
            Leaky_ReLU_neg_slope]


def main():

    start_time_in_seconds = time.time()

    [model_storage_dir,
     testing_data_dir,
     model_storage_folder_name,
     testing_data_file_name,
     model_type,
     dropout_rate,
     CNNet_1D_num_filters,
     CNNet_1D_kernel_size,
     Leaky_ReLU_neg_slope] = \
         get_inputs()

    [num_patients_per_trial_arm_array,
     MPC_stat_power_array,
     MPC_type_1_error_array,
     true_drug_placebo_arm_hists_over_all_trial_sizes,
     true_drug_treatment_arm_hists_over_all_trial_sizes,
     false_drug_placebo_arm_hists_over_all_trial_sizes,
     false_drug_treatment_arm_hists_over_all_trial_sizes] = \
         load_data(testing_data_dir,
                   testing_data_file_name)

    [model_storage_file_path,
     thresholds_storage_file_path,
     results_file_path,
     testing_time_file_path] = \
         determine_hyperparameter_based_file_paths(model_storage_dir,
                                                   model_storage_folder_name,
                                                   model_type,
                                                   dropout_rate,
                                                   CNNet_1D_num_filters,
                                                   CNNet_1D_kernel_size,
                                                   Leaky_ReLU_neg_slope)
    
    deepRCT_model = models.load_model(model_storage_file_path)

    with open(thresholds_storage_file_path, 'r') as threshold_storage_json_file:
        threshold_array = np.array(json.load(threshold_storage_json_file))
    
    [deepRCT_model_stat_power_array,
     deepRCT_model_type_1_error_array] = \
         evaluate_model_over_testing_set_data(true_drug_placebo_arm_hists_over_all_trial_sizes,
                                              true_drug_treatment_arm_hists_over_all_trial_sizes,
                                              false_drug_placebo_arm_hists_over_all_trial_sizes,
                                              false_drug_treatment_arm_hists_over_all_trial_sizes,
                                              deepRCT_model,
                                              threshold_array)
    
    store_results(num_patients_per_trial_arm_array,
                  MPC_stat_power_array,
                  MPC_type_1_error_array,
                  deepRCT_model_stat_power_array,
                  deepRCT_model_type_1_error_array,
                  results_file_path,
                  testing_time_file_path,
                  start_time_in_seconds)    


if(__name__=='__main__'):

    main()
