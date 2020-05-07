import keras.models as models
import numpy as np
import json
import time
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
    
    stat_power_storage_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_stat_powers.json'
    
    type_1_error_storage_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_type_1_errors.json'

    thresholds_storage_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_thresholds.json'

    optimization_time_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_runtimes.txt'

    return [model_storage_file_path,
            stat_power_storage_file_path,
            type_1_error_storage_file_path,
            thresholds_storage_file_path,
            optimization_time_file_path]


def load_data(validation_data_dir,
              validation_data_file_name):

    validation_data_file_path = validation_data_dir + '/' + validation_data_file_name + '.json'
    with open(validation_data_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    true_drug_placebo_arm_hists_over_all_trial_sizes    = np.array(data[3])
    true_drug_treatment_arm_hists_over_all_trial_sizes  = np.array(data[4])
    false_drug_placebo_arm_hists_over_all_trial_sizes   = np.array(data[5])
    false_drug_treatment_arm_hists_over_all_trial_sizes = np.array(data[6])

    return [true_drug_placebo_arm_hists_over_all_trial_sizes,
            true_drug_treatment_arm_hists_over_all_trial_sizes,
            false_drug_placebo_arm_hists_over_all_trial_sizes,
            false_drug_treatment_arm_hists_over_all_trial_sizes]


def optimize_threshold_over_trial_size(deepRCT_model_true_drug_prob_array,
                                       deepRCT_model_false_drug_prob_array,
                                       num_samples_per_trial_size):

    threshold = 0.5
    threshold_step = 10**(-6)

    acceptable_upper_type_1_error_bound = 0.055
    acceptable_lower_type_1_error_bound = 0.045

    deepRCT_model_type_1_error = np.sum(deepRCT_model_false_drug_prob_array > threshold)/num_samples_per_trial_size

    type_1_error_too_low  = deepRCT_model_type_1_error < acceptable_lower_type_1_error_bound
    type_1_error_too_high = deepRCT_model_type_1_error > acceptable_upper_type_1_error_bound

    num_times_type_1_error_is_zero = 0
    type_1_error_is_not_oscillating_from_zero = True

    while( (type_1_error_too_low or type_1_error_too_high) and type_1_error_is_not_oscillating_from_zero ):

        if(type_1_error_too_low):

            threshold = threshold - threshold_step
        
        elif(type_1_error_too_high):

            threshold = threshold + threshold_step
        
        if( np.round(100*deepRCT_model_type_1_error, 3) == 0 ):

            num_times_type_1_error_is_zero = num_times_type_1_error_is_zero + 1

            if(num_times_type_1_error_is_zero == 20):

                type_1_error_is_not_oscillating_from_zero = False
        
        deepRCT_model_type_1_error = \
            deepRCT_model_type_1_error = np.sum(deepRCT_model_false_drug_prob_array > threshold)/num_samples_per_trial_size
        
        type_1_error_too_low  = deepRCT_model_type_1_error < acceptable_lower_type_1_error_bound
        type_1_error_too_high = deepRCT_model_type_1_error > acceptable_upper_type_1_error_bound

        print('type 1 error: ' + str(np.round(100*deepRCT_model_type_1_error, 3)) + ' %, threshold = ' + str(threshold))
     
    deepRCT_model_stat_power = np.sum(deepRCT_model_true_drug_prob_array > threshold)/num_samples_per_trial_size

    return [deepRCT_model_stat_power,
            deepRCT_model_type_1_error,
            threshold]


def optimize_thresholds_over_all_trial_sizes(deepRCT_model,
                                             true_drug_placebo_arm_hists_over_all_trial_sizes,
                                             true_drug_treatment_arm_hists_over_all_trial_sizes,
                                             false_drug_placebo_arm_hists_over_all_trial_sizes,
                                             false_drug_treatment_arm_hists_over_all_trial_sizes):

    [num_trial_sizes, 
     num_samples_per_trial_size, 
     num_bins, _] = \
         true_drug_placebo_arm_hists_over_all_trial_sizes.shape

    deepRCT_model_stat_power_array   = np.zeros(num_trial_sizes)
    deepRCT_model_type_1_error_array = np.zeros(num_trial_sizes)
    threshold_array                  = np.zeros(num_trial_sizes)
    
    for trial_size_index in range(num_trial_sizes):

        deepRCT_model_true_drug_prob_array = \
            np.squeeze(deepRCT_model.predict([true_drug_placebo_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :],
                                              true_drug_treatment_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :]]))
        
        deepRCT_model_false_drug_prob_array = \
            np.squeeze(deepRCT_model.predict([false_drug_placebo_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :],
                                              false_drug_treatment_arm_hists_over_all_trial_sizes[trial_size_index, :, :, :]]))

        [deepRCT_model_stat_power_array[trial_size_index],
         deepRCT_model_type_1_error_array[trial_size_index],
         threshold_array[trial_size_index]] = \
             optimize_threshold_over_trial_size(deepRCT_model_true_drug_prob_array,
                                                deepRCT_model_false_drug_prob_array,
                                                num_samples_per_trial_size)
    
    return [deepRCT_model_stat_power_array, 
            deepRCT_model_type_1_error_array, 
            threshold_array]


def store_thresholds_and_stats(stat_power_storage_file_path,
                               type_1_error_storage_file_path,
                               thresholds_storage_file_path,
                               optimization_time_file_path,
                               deepRCT_model_stat_power_array,
                               deepRCT_model_type_1_error_array,
                               threshold_array,
                               start_time_in_seconds):

    with open(stat_power_storage_file_path, 'w+') as model_stat_power_json_file:
        json.dump(deepRCT_model_stat_power_array.tolist(), model_stat_power_json_file)
    
    with open(type_1_error_storage_file_path, 'w+') as model_type_1_error_json_file:
        json.dump(deepRCT_model_type_1_error_array.tolist(), model_type_1_error_json_file)

    with open(thresholds_storage_file_path, 'w+') as thresholds_json_file:
        json.dump(threshold_array.tolist(), thresholds_json_file)
    
    stop_time_in_seconds = time.time()
    total_runtime_in_minutes_str = str( np.round((stop_time_in_seconds - start_time_in_seconds)/60, 3) ) + ' minutes of threshold optimization'

    with open(optimization_time_file_path, 'a') as text_file:
        text_file.write('\n' + total_runtime_in_minutes_str)


def get_inputs():

    print(sys.argv)

    model_storage_folder_name = sys.argv[1]
    validation_data_file_name = sys.argv[2]
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

    #==========================================================================================#
    
    model_storage_dir = os.getcwd()
    validation_data_dir = os.getcwd()

    return [model_storage_dir,
            validation_data_dir,
            model_storage_folder_name,
            validation_data_file_name,
            model_type,
            dropout_rate,
            CNNet_1D_num_filters,
            CNNet_1D_kernel_size,
            Leaky_ReLU_neg_slope]


def main():

    start_time_in_seconds = time.time()

    [model_storage_dir,
     validation_data_dir,
     model_storage_folder_name,
     validation_data_file_name,
     model_type,
     dropout_rate,
     CNNet_1D_num_filters,
     CNNet_1D_kernel_size,
     Leaky_ReLU_neg_slope] = \
         get_inputs()
    
    [model_storage_file_path,
     stat_power_storage_file_path,
     type_1_error_storage_file_path,
     thresholds_storage_file_path,
     optimization_time_file_path] = \
         determine_hyperparameter_based_file_paths(model_storage_dir,
                                                   model_storage_folder_name,
                                                   model_type,
                                                   dropout_rate,
                                                   CNNet_1D_num_filters,
                                                   CNNet_1D_kernel_size,
                                                   Leaky_ReLU_neg_slope)
    
    deepRCT_model = models.load_model(model_storage_file_path)

    [true_drug_placebo_arm_hists_over_all_trial_sizes,
     true_drug_treatment_arm_hists_over_all_trial_sizes,
     false_drug_placebo_arm_hists_over_all_trial_sizes,
     false_drug_treatment_arm_hists_over_all_trial_sizes] = \
         load_data(validation_data_dir,
                   validation_data_file_name)

    [deepRCT_model_stat_power_array, 
     deepRCT_model_type_1_error_array, 
     threshold_array] = \
         optimize_thresholds_over_all_trial_sizes(deepRCT_model,
                                                  true_drug_placebo_arm_hists_over_all_trial_sizes,
                                                  true_drug_treatment_arm_hists_over_all_trial_sizes,
                                                  false_drug_placebo_arm_hists_over_all_trial_sizes,
                                                  false_drug_treatment_arm_hists_over_all_trial_sizes)

    store_thresholds_and_stats(stat_power_storage_file_path,
                               type_1_error_storage_file_path,
                               thresholds_storage_file_path,
                               optimization_time_file_path,
                               deepRCT_model_stat_power_array,
                               deepRCT_model_type_1_error_array,
                               threshold_array,
                               start_time_in_seconds)


if(__name__=='__main__'):

    main()
