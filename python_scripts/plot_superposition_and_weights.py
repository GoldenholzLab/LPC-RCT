from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import keras.models as models
import numpy as np
import json
import sys
import os
from PIL import Image
import io


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

    return results_file_path


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


def get_data_to_plot(model_storage_dir,
                     model_storage_folder_name):

    complex_model_types = ['simple', 'intermediate', 'complicated']
    dropout_rates = [0, 0.1, 0.3]
    CNNet_1D_nums_filters = [4, 8]
    CNNet_1D_kernel_sizes = [3, 5, 9]
    Leaky_ReLU_neg_slopes = [0, 0.2, 0.4]

    num_hist_bins = 80
    upper_x_axis_bound = 1
    lower_x_axis_bound = -3

    [num_patients_per_trial_arm_array,
     MPC_stat_power_array,
     MPC_type_1_error_array,
     deepRCT_model_stat_power_array,
     deepRCT_model_type_1_error_array] = \
         load_data(determine_hyperparameter_based_file_paths(model_storage_dir,
                                                             model_storage_folder_name,
                                                             'single_perceptron'))
    
    print('MPC: ' + str(calculate_line_crossing(MPC_stat_power_array,
                                                num_patients_per_trial_arm_array)))

    log_reg_stat_power_array   = np.copy(deepRCT_model_stat_power_array)
    log_reg_type_1_error_array = np.copy(deepRCT_model_type_1_error_array)
    
    num_trial_sizes = len(num_patients_per_trial_arm_array)

    deepRCT_model_stat_power_arrays_list   = []
    deepRCT_model_type_1_error_arrays_list = []
    deepRCT_model_stat_power_arrays_list.append(deepRCT_model_stat_power_array)
    deepRCT_model_type_1_error_arrays_list.append(deepRCT_model_type_1_error_array)

    line_crossing = \
        calculate_line_crossing(deepRCT_model_stat_power_array,
                            num_patients_per_trial_arm_array)

    line_crossing_str = 'Log. Reg.: ' + str(line_crossing)

    min_line_crossing     = line_crossing
    min_line_crossing_str = line_crossing_str

    print(line_crossing_str)

    for dropout_rate in dropout_rates:

        results_file_path = \
            determine_hyperparameter_based_file_paths(model_storage_dir,
                                                      model_storage_folder_name,
                                                      '3_layer',
                                                      dropout_rate)

        if( os.path.isfile(results_file_path) ):

            [_, _, _,
             deepRCT_model_stat_power_array,
             deepRCT_model_type_1_error_array] = \
                 load_data(determine_hyperparameter_based_file_paths(model_storage_dir,
                                                                     model_storage_folder_name,
                                                                     '3_layer',
                                                                     dropout_rate))
            
            deepRCT_model_stat_power_arrays_list.append(deepRCT_model_stat_power_array)
            deepRCT_model_type_1_error_arrays_list.append(deepRCT_model_type_1_error_array)

            line_crossing = \
                calculate_line_crossing(deepRCT_model_stat_power_array,
                                        num_patients_per_trial_arm_array)
            
            line_crossing_str = '3_layer, dropout=' + str(10*dropout_rate) + '% : ' + str(line_crossing)

            if(line_crossing < min_line_crossing):

                min_line_crossing     = line_crossing
                min_line_crossing_str = line_crossing_str

            print(line_crossing_str)

    min_line_crossing = 1000

    for model_type in complex_model_types:
        for dropout_rate in dropout_rates:
            for CNNet_1D_num_filters in CNNet_1D_nums_filters:
                for CNNet_1D_kernel_size in CNNet_1D_kernel_sizes:
                    for Leaky_ReLU_neg_slope in Leaky_ReLU_neg_slopes:

                        results_file_path = \
                            determine_hyperparameter_based_file_paths(model_storage_dir,
                                                                      model_storage_folder_name,
                                                                      model_type,
                                                                      dropout_rate,
                                                                      CNNet_1D_num_filters,
                                                                      CNNet_1D_kernel_size,
                                                                      Leaky_ReLU_neg_slope)
                        
                        if( os.path.isfile(results_file_path) ):

                            [_, _, _,
                             deepRCT_model_stat_power_array,
                             deepRCT_model_type_1_error_array] = \
                                 load_data(results_file_path)

                            deepRCT_model_stat_power_arrays_list.append(deepRCT_model_stat_power_array)
                            deepRCT_model_type_1_error_arrays_list.append(deepRCT_model_type_1_error_array)

                            line_crossing = \
                                calculate_line_crossing(deepRCT_model_stat_power_array,
                                                        num_patients_per_trial_arm_array)

                            line_crossing_str = model_type + ', dropout=' + str(10*dropout_rate) + '%, num filters=' + \
                                                str(CNNet_1D_num_filters) + ', kernel size=' + str(CNNet_1D_kernel_size) + \
                                                ', negative slope=' + str(Leaky_ReLU_neg_slope) + ' : ' + str(line_crossing)

                            print(line_crossing_str)

                            if(line_crossing < min_line_crossing):

                                min_line_crossing     = line_crossing
                                min_line_crossing_str = line_crossing_str

    print(min_line_crossing_str)

    num_models = len(deepRCT_model_stat_power_arrays_list)
    deepRCT_model_stat_power_arrays   = np.zeros((num_models, num_trial_sizes))
    deepRCT_model_type_1_error_arrays = np.zeros((num_models, num_trial_sizes))
    deepRCT_model_stat_power_arrays_list.reverse()
    deepRCT_model_type_1_error_arrays_list.reverse()
    for model_index in range(num_models):
        deepRCT_model_stat_power_arrays[model_index, :]   = deepRCT_model_stat_power_arrays_list.pop()
        deepRCT_model_type_1_error_arrays[model_index, :] = deepRCT_model_type_1_error_arrays_list.pop()
    
    mean_deepRCT_model_stat_power_array   = np.mean(deepRCT_model_stat_power_arrays,   0)
    mean_deepRCT_model_type_1_error_array = np.mean(deepRCT_model_type_1_error_arrays, 0)

    std_dev_deepRCT_model_stat_power_array   = np.std(deepRCT_model_stat_power_arrays,   0)
    std_dev_deepRCT_model_type_1_error_array = np.std(deepRCT_model_type_1_error_arrays, 0)

    single_perceptron_model_file_path = \
        model_storage_dir + '/' + model_storage_folder_name + '/single_perceptron/only_one_model.h5'

    single_perceptron_model = models.load_model(single_perceptron_model_file_path)
    
    placebo_arm_hist_weights = np.squeeze(single_perceptron_model.layers[5].get_weights()[0])[0:num_hist_bins]
    drug_arm_hist_weights    = np.squeeze(single_perceptron_model.layers[5].get_weights()[0])[num_hist_bins:]

    x_axis_step = (upper_x_axis_bound - lower_x_axis_bound)/num_hist_bins
    x_axis = 100*np.arange(lower_x_axis_bound, upper_x_axis_bound, x_axis_step) + (x_axis_step/2)

    return [num_patients_per_trial_arm_array,
            MPC_stat_power_array,
            MPC_type_1_error_array,
            log_reg_stat_power_array,
            log_reg_type_1_error_array,
            mean_deepRCT_model_stat_power_array,
            mean_deepRCT_model_type_1_error_array,
            std_dev_deepRCT_model_stat_power_array,
            std_dev_deepRCT_model_type_1_error_array,
            placebo_arm_hist_weights,
            drug_arm_hist_weights,
            x_axis]


if(__name__=='__main__'):


    model_storage_dir = os.getcwd()
    NV_model_one_model_storage_folder_name = 'NV_model_one_weekly_regular_level_16_keras_models'
    NV_model_two_model_storage_folder_name = 'NV_model_two_weekly_regular_level_16_keras_models'

    [NV_model_one_num_patients_per_trial_arm_array,
     NV_model_one_MPC_stat_power_array,
     NV_model_one_MPC_type_1_error_array,
     NV_model_one_log_reg_stat_power_array,
     NV_model_one_log_reg_type_1_error_array,
     NV_model_one_mean_deepRCT_model_stat_power_array,
     NV_model_one_mean_deepRCT_model_type_1_error_array,
     NV_model_one_std_dev_deepRCT_model_stat_power_array,
     NV_model_one_std_dev_deepRCT_model_type_1_error_array,
     NV_model_one_placebo_arm_hist_weights,
     NV_model_one_drug_arm_hist_weights,
     NV_model_one_x_axis] = \
         get_data_to_plot(model_storage_dir,
                          NV_model_one_model_storage_folder_name)

    [NV_model_two_num_patients_per_trial_arm_array,
     NV_model_two_MPC_stat_power_array,
     NV_model_two_MPC_type_1_error_array,
     NV_model_two_log_reg_stat_power_array,
     NV_model_two_log_reg_type_1_error_array,
     NV_model_two_mean_deepRCT_model_stat_power_array,
     NV_model_two_mean_deepRCT_model_type_1_error_array,
     NV_model_two_std_dev_deepRCT_model_stat_power_array,
     NV_model_two_std_dev_deepRCT_model_type_1_error_array,
     NV_model_two_placebo_arm_hist_weights,
     NV_model_two_drug_arm_hist_weights,
     NV_model_two_x_axis] = \
         get_data_to_plot(model_storage_dir,
                          NV_model_two_model_storage_folder_name)
    
    NV_model_placebo_arm_weights = \
        [NV_model_one_placebo_arm_hist_weights, 
         NV_model_two_placebo_arm_hist_weights]
    
    NV_model_drug_arm_weights = \
        [NV_model_one_drug_arm_hist_weights, 
         NV_model_two_drug_arm_hist_weights]
    
    fig1 = plt.figure(figsize=(8,9))

    ax = plt.subplot(2,1,1)

    plt.plot(2*NV_model_one_num_patients_per_trial_arm_array, 100*NV_model_one_MPC_stat_power_array,   color='blue',                 label='MPC Stat. Power')
    plt.plot(2*NV_model_one_num_patients_per_trial_arm_array, 100*NV_model_one_MPC_type_1_error_array, color='blue', linestyle='--', label='MPC Type 1 Error')
    plt.plot(2*NV_model_one_num_patients_per_trial_arm_array, 100*NV_model_one_log_reg_stat_power_array,   color='green',                 label='LPC Stat. Power')
    plt.plot(2*NV_model_one_num_patients_per_trial_arm_array, 100*NV_model_one_log_reg_type_1_error_array, color='green', linestyle='--', label='LPC Type 1 Error')
    plt.errorbar(2*NV_model_one_num_patients_per_trial_arm_array, 100*NV_model_one_mean_deepRCT_model_stat_power_array,   100*NV_model_one_std_dev_deepRCT_model_stat_power_array,   color='k',                 label='candidate model Stat. Power Average')
    plt.errorbar(2*NV_model_one_num_patients_per_trial_arm_array, 100*NV_model_one_mean_deepRCT_model_type_1_error_array, 100*NV_model_one_std_dev_deepRCT_model_type_1_error_array, color='k', linestyle='--', label='candidate model Type 1 Error Average')
    plt.axhline(90,label='90% Stat. Power', color='red', linestyle='--')
    plt.axhline(5, label='5% Type 1 Error', color='red')

    plt.xticks(np.arange(40, 460 + 40, 40))
    plt.yticks(np.arange(0, 100 + 10, 10))
    plt.grid()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))

    plt.xlabel('number of patients per trial')
    plt.title('Statistical Power Curves for NV model 1')

    plt.legend()

    ax.text(-0.1, 1, 'A)', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

    ax = plt.subplot(2,1,2)

    plt.plot(2*NV_model_two_num_patients_per_trial_arm_array, 100*NV_model_two_MPC_stat_power_array,   color='blue',                 label='MPC Stat. Power')
    plt.plot(2*NV_model_two_num_patients_per_trial_arm_array, 100*NV_model_two_MPC_type_1_error_array, color='blue', linestyle='--', label='MPC Type 1 Error')
    plt.plot(2*NV_model_two_num_patients_per_trial_arm_array, 100*NV_model_two_log_reg_stat_power_array,   color='green',                 label='LPC Stat. Power')
    plt.plot(2*NV_model_two_num_patients_per_trial_arm_array, 100*NV_model_two_log_reg_type_1_error_array, color='green', linestyle='--', label='LPC Type 1 Error')
    plt.errorbar(2*NV_model_two_num_patients_per_trial_arm_array, 100*NV_model_two_mean_deepRCT_model_stat_power_array,   100*NV_model_two_std_dev_deepRCT_model_stat_power_array,   color='k',                 label='candidate model Stat. Power Average')
    plt.errorbar(2*NV_model_two_num_patients_per_trial_arm_array, 100*NV_model_two_mean_deepRCT_model_type_1_error_array, 100*NV_model_two_std_dev_deepRCT_model_type_1_error_array, color='k', linestyle='--', label='candidate model Type 1 Error Average')
    plt.axhline(90,label='90% Stat. Power', color='red', linestyle='--')
    plt.axhline(5, label='5% Type 1 Error', color='red')

    plt.xticks(np.arange(40, 460 + 40, 40))
    plt.yticks(np.arange(0, 100 + 10, 10))
    plt.grid()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))

    plt.xlabel('number of patients per trial')
    plt.title('Statistical Power Curves for NV model 2')

    plt.legend()

    ax.text(-0.1, 1, 'B)', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    fig2 = plt.figure(figsize=(12,8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2)

    for NV_model_index in range(2):

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[NV_model_index], hspace=0.3)

        ax = plt.Subplot(fig2, inner[0])
        ax.set_title('NV model ' + str(NV_model_index + 1) + ' placebo arm histogram weights')
        ax.stem(NV_model_two_x_axis, NV_model_placebo_arm_weights[NV_model_index], use_line_collection=True)
        ax.set_ylim([-8, 8])
        ax.set_yticks(np.arange(-8, 8 + 2, 2))
        ax.yaxis.grid(True)
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xlabel('percent change')
        ax.set_ylabel('placebo arm histogram weights')
        
        fig2.add_subplot(ax)

        ax = plt.Subplot(fig2, inner[1])
        ax.set_title('NV model ' + str(NV_model_index + 1) + ' drug arm histogram weights')
        ax.stem(NV_model_two_x_axis, NV_model_drug_arm_weights[NV_model_index], use_line_collection=True)
        ax.set_ylim([-8, 8])
        ax.set_yticks(np.arange(-8, 8 + 2, 2))
        ax.yaxis.grid(True)
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xlabel('percent change')
        ax.set_ylabel('drug arm histogram weights')

        fig2.add_subplot(ax)

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.95)
    
    png1 = io.BytesIO()
    fig1.savefig(png1, dpi = 600, bbox_inches = 'tight', format = 'png')
    png2 = Image.open(png1)
    png2.save('Romero-fig1.tiff')
    png1.close()

    png1 = io.BytesIO()
    fig2.savefig(png1, dpi = 600, bbox_inches = 'tight', format = 'png')
    png2 = Image.open(png1)
    png2.save('Romero-fig2.tiff')
    png1.close()

    #plt.show()

