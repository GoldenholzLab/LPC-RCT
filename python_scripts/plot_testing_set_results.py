from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
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

    results_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_results.json'

    plot_file_path = \
        model_storage_dir + '/' + model_storage_folder_name                    + '/' + \
        model_type        + '/' + hyperparameter_based_deepRCT_model_file_name + '_plot.png'

    return [results_file_path, 
            plot_file_path]


def load_data(results_file_path):

    with open(results_file_path, 'r') as json_file:
        data = json.load(json_file)

    num_patients_per_trial_arm_array = np.array(data[0])
    MPC_stat_power_array             = np.array(data[1])
    MPC_type_1_error_array           = np.array(data[2])
    deepRCT_model_stat_power_array   = np.array(data[3])
    deepRCT_model_type_1_error_array = np.array(data[4])

    return [num_patients_per_trial_arm_array,
            MPC_stat_power_array,
            MPC_type_1_error_array,
            deepRCT_model_stat_power_array,
            deepRCT_model_type_1_error_array]


def calculate_line_crossing(deepRCT_model_stat_power_array,
                            num_patients_per_trial_arm_array):

    index = 0

    ninety_percent_power_not_reached = True

    while(ninety_percent_power_not_reached):

        ninety_percent_power_not_reached = not( (100*deepRCT_model_stat_power_array[index] > 90) and (index  != 0) )

        if(ninety_percent_power_not_reached):

            index = index  + 1
        
    y1 = 100*deepRCT_model_stat_power_array[index  - 1]
    y2 = 100*deepRCT_model_stat_power_array[index]
    x1 = 2*num_patients_per_trial_arm_array[index - 1]
    x2 = 2*num_patients_per_trial_arm_array[index]
    
    slope = (y2 - y1)/(x2 - x1)
    intercept = y2 - slope*x2
    line_crossing = (90 - intercept)/slope
    
    return line_crossing


def get_inputs():
    
    print(sys.argv)

    model_storage_folder_name = sys.argv[1]
    model_type = sys.argv[2]

    dropout_rate = None
    CNNet_1D_num_filters = None
    CNNet_1D_kernel_size = None
    Leaky_ReLU_neg_slope = None

    if(len(sys.argv) > 3):
        dropout_rate = float(sys.argv[3])

    if(len(sys.argv) > 4):
        CNNet_1D_num_filters =   int(sys.argv[4])
        CNNet_1D_kernel_size =   int(sys.argv[5])
        Leaky_ReLU_neg_slope = float(sys.argv[6])

    #=======================================================================================#

    model_storage_dir = os.getcwd()

    return [model_type,
            model_storage_dir,
            model_storage_folder_name,
            dropout_rate,
            CNNet_1D_num_filters,
            CNNet_1D_kernel_size,
            Leaky_ReLU_neg_slope]


def main():

    [model_type,
     model_storage_dir,
     model_storage_folder_name,
     dropout_rate,
     CNNet_1D_num_filters,
     CNNet_1D_kernel_size,
     Leaky_ReLU_neg_slope] = \
         get_inputs()

    [results_file_path, 
     plot_file_path] = \
         determine_hyperparameter_based_file_paths(model_storage_dir,
                                                   model_storage_folder_name,
                                                   model_type,
                                                   dropout_rate,
                                                   CNNet_1D_num_filters,
                                                   CNNet_1D_kernel_size,
                                                   Leaky_ReLU_neg_slope)
    
    [num_patients_per_trial_arm_array,
     MPC_stat_power_array,
     MPC_type_1_error_array,
     deepRCT_model_stat_power_array,
     deepRCT_model_type_1_error_array] = \
         load_data(results_file_path)
    
    '''
    deepRCT_line_crossing = \
        calculate_line_crossing(deepRCT_model_stat_power_array,
                                num_patients_per_trial_arm_array)
    
    MPC_line_crossing = \
        calculate_line_crossing(MPC_stat_power_array,
                                num_patients_per_trial_arm_array)
    
    print([deepRCT_line_crossing, MPC_line_crossing])
    '''

    plt.figure()

    plt.plot(2*num_patients_per_trial_arm_array, 100*MPC_stat_power_array,             label='MPC Stat power')
    plt.plot(2*num_patients_per_trial_arm_array, 100*MPC_type_1_error_array,           label='MPC Type 1 Error')
    plt.plot(2*num_patients_per_trial_arm_array, 100*deepRCT_model_stat_power_array,   label='DL model Stat Power')
    plt.plot(2*num_patients_per_trial_arm_array, 100*deepRCT_model_type_1_error_array, label='DL model Type 1 Error')

    plt.axhline(90,color='k',label='90% power')
    plt.axhline(5, color='k', label='5% Type 1 Error', linestyle='--')

    plt.xticks(np.arange(40, 460 + 40, 40))
    plt.yticks(np.arange(0, 100 + 10, 10))
    plt.grid()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))

    plt.xlabel('number of patients per trial')
    plt.title('NV model one, rotated percent change')

    plt.legend()
    
    plt.savefig(plot_file_path)


if(__name__=='__main__'):

    main()
