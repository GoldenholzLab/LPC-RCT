from clinical_trial_generation import generate_true_and_false_drug_histograms_and_MPC_stat_significances
import numpy as np
import json
import time
import sys


def generate_evaluation_data(one_or_two,
                             num_baseline_months,
                             num_maintenance_months,
                             baseline_time_scale,
                             maintenance_time_scale,
                             minimum_cumulative_baseline_seizure_count,
                             placebo_percent_effect_mean_upper_bound,
                             placebo_percent_effect_mean_lower_bound,
                             drug_percent_effect_mean_upper_bound,
                             drug_percent_effect_mean_lower_bound,
                             placebo_percent_effect_std_dev,
                             drug_percent_effect_std_dev,
                             num_bins,
                             hist_range,
                             num_patients_per_trial_arm_step,
                             final_num_patients_trial_arm_step,
                             num_trials_per_stat_power_estim):

    num_patients_per_trial_arm_array = \
        np.arange(num_patients_per_trial_arm_step, 
                  final_num_patients_trial_arm_step + num_patients_per_trial_arm_step,
                  num_patients_per_trial_arm_step)

    num_trial_sizes = int(final_num_patients_trial_arm_step/num_patients_per_trial_arm_step)

    MPC_true_drug_stat_significance_matrix  = np.zeros((num_trials_per_stat_power_estim, num_trial_sizes), dtype=bool)
    MPC_false_drug_stat_significance_matrix = np.zeros((num_trials_per_stat_power_estim, num_trial_sizes), dtype=bool)

    true_drug_placebo_arm_hists_over_all_trial_sizes    = np.zeros((num_trial_sizes, num_trials_per_stat_power_estim, num_bins, 1))
    true_drug_treatment_arm_hists_over_all_trial_sizes  = np.zeros((num_trial_sizes, num_trials_per_stat_power_estim, num_bins, 1))
    false_drug_placebo_arm_hists_over_all_trial_sizes   = np.zeros((num_trial_sizes, num_trials_per_stat_power_estim, num_bins, 1))
    false_drug_treatment_arm_hists_over_all_trial_sizes = np.zeros((num_trial_sizes, num_trials_per_stat_power_estim, num_bins, 1))

    for trial_iter in range(num_trials_per_stat_power_estim):

        for trial_size_index in range(num_trial_sizes):

            num_patients_per_trial_arm = num_patients_per_trial_arm_array[trial_size_index]

            num_placebo_arm_patients = num_patients_per_trial_arm
            num_drug_arm_patients    = num_patients_per_trial_arm

            [MPC_true_drug_stat_significance_matrix[trial_iter, trial_size_index], 
             MPC_false_drug_stat_significance_matrix[trial_iter, trial_size_index],
             true_drug_placebo_arm_hists_over_all_trial_sizes[trial_size_index, trial_iter, :, 0], 
             true_drug_treatment_arm_hists_over_all_trial_sizes[trial_size_index, trial_iter, :, 0],
             false_drug_placebo_arm_hists_over_all_trial_sizes[trial_size_index, trial_iter, :, 0],
             false_drug_treatment_arm_hists_over_all_trial_sizes[trial_size_index, trial_iter, :, 0]] = \
                 generate_true_and_false_drug_histograms_and_MPC_stat_significances(one_or_two,
                                                                                    num_placebo_arm_patients,
                                                                                    num_drug_arm_patients,
                                                                                    num_baseline_months,
                                                                                    num_maintenance_months,
                                                                                    baseline_time_scale,
                                                                                    maintenance_time_scale,
                                                                                    minimum_cumulative_baseline_seizure_count,
                                                                                    placebo_percent_effect_mean_upper_bound,
                                                                                    placebo_percent_effect_mean_lower_bound,
                                                                                    placebo_percent_effect_std_dev,
                                                                                    drug_percent_effect_mean_upper_bound,
                                                                                    drug_percent_effect_mean_lower_bound,
                                                                                    drug_percent_effect_std_dev,
                                                                                    num_bins,
                                                                                    hist_range)
            
            print('trial #' + str(trial_iter + 1) + ' of ' + str(2*num_patients_per_trial_arm) + ' patients')

    MPC_stat_power_array   = np.sum(MPC_true_drug_stat_significance_matrix,  0)/num_trials_per_stat_power_estim
    MPC_type_1_error_array = np.sum(MPC_false_drug_stat_significance_matrix, 0)/num_trials_per_stat_power_estim

    return [num_patients_per_trial_arm_array,
            MPC_stat_power_array,
            MPC_type_1_error_array,
            true_drug_placebo_arm_hists_over_all_trial_sizes,
            true_drug_treatment_arm_hists_over_all_trial_sizes,
            false_drug_placebo_arm_hists_over_all_trial_sizes,
            false_drug_treatment_arm_hists_over_all_trial_sizes]


def store_evaluation_data(num_patients_per_trial_arm_array,
                          MPC_stat_power_array,
                          MPC_type_1_error_array,
                          true_drug_placebo_arm_hists_over_all_trial_sizes,
                          true_drug_treatment_arm_hists_over_all_trial_sizes,
                          false_drug_placebo_arm_hists_over_all_trial_sizes,
                          false_drug_treatment_arm_hists_over_all_trial_sizes,
                          evaluation_data_dir,
                          evaluation_data_file_name):

    evaluation_data_file_path = evaluation_data_dir + '/' + evaluation_data_file_name + '.json'

    data = []
    data.append(num_patients_per_trial_arm_array.tolist())
    data.append(MPC_stat_power_array.tolist())
    data.append(MPC_type_1_error_array.tolist())
    data.append(true_drug_placebo_arm_hists_over_all_trial_sizes.tolist())
    data.append(true_drug_treatment_arm_hists_over_all_trial_sizes.tolist())
    data.append(false_drug_placebo_arm_hists_over_all_trial_sizes.tolist())
    data.append(false_drug_treatment_arm_hists_over_all_trial_sizes.tolist())
    with open(evaluation_data_file_path, 'w+') as json_file:
        json.dump(data, json_file)


def get_inputs():

    one_or_two = sys.argv[1]

    placebo_percent_effect_mean_upper_bound = float(sys.argv[2])
    placebo_percent_effect_mean_lower_bound = float(sys.argv[3])

    drug_percent_effect_mean_upper_bound = float(sys.argv[4])
    drug_percent_effect_mean_lower_bound = float(sys.argv[5])

    placebo_percent_effect_std_dev = float(sys.argv[6])
    drug_percent_effect_std_dev    = float(sys.argv[7])

    num_patients_per_trial_step   = int(sys.argv[8])
    final_num_patients_trial_step = int(sys.argv[9])

    num_trials_per_stat_power_estim = int(sys.argv[10])

    evaluation_data_file_name = sys.argv[11]

    #======================================================================================================#

    if((num_patients_per_trial_step % 2 == 1) or (final_num_patients_trial_step % 2 == 1)):

        raise ValueError('The number of patients per trial needs to divisible by 2.')

    num_patients_per_trial_arm_step = int(num_patients_per_trial_step/2)
    final_num_patients_trial_arm_step = int(final_num_patients_trial_step/2)

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count = 4

    num_bins = 80
    hist_range = [-3, 1]

    import os
    evaluation_data_dir = os.getcwd()

    return [one_or_two,
            num_baseline_months,
            num_maintenance_months,
            baseline_time_scale,
            maintenance_time_scale,
            minimum_cumulative_baseline_seizure_count,
            placebo_percent_effect_mean_upper_bound,
            placebo_percent_effect_mean_lower_bound,
            drug_percent_effect_mean_upper_bound,
            drug_percent_effect_mean_lower_bound,
            placebo_percent_effect_std_dev,
            drug_percent_effect_std_dev,
            num_bins,
            hist_range,
            num_patients_per_trial_arm_step,
            final_num_patients_trial_arm_step,
            num_trials_per_stat_power_estim,
            evaluation_data_dir,
            evaluation_data_file_name]


def main():

    [one_or_two,
     num_baseline_months,
     num_maintenance_months,
     baseline_time_scale,
     maintenance_time_scale,
     minimum_cumulative_baseline_seizure_count,
     placebo_percent_effect_mean_upper_bound,
     placebo_percent_effect_mean_lower_bound,
     drug_percent_effect_mean_upper_bound,
     drug_percent_effect_mean_lower_bound,
     placebo_percent_effect_std_dev,
     drug_percent_effect_std_dev,
     num_bins,
     hist_range,
     num_patients_per_trial_arm_step,
     final_num_patients_trial_arm_step,
     num_trials_per_stat_power_estim,
     evaluation_data_dir,
     evaluation_data_file_name] = \
         get_inputs()

    [num_patients_per_trial_arm_array,
     MPC_stat_power_array,
     MPC_type_1_error_array,
     true_drug_placebo_arm_hists_over_all_trial_sizes,
     true_drug_treatment_arm_hists_over_all_trial_sizes,
     false_drug_placebo_arm_hists_over_all_trial_sizes,
     false_drug_treatment_arm_hists_over_all_trial_sizes] = \
         generate_evaluation_data(one_or_two,
                                  num_baseline_months,
                                  num_maintenance_months,
                                  baseline_time_scale,
                                  maintenance_time_scale,
                                  minimum_cumulative_baseline_seizure_count,
                                  placebo_percent_effect_mean_upper_bound,
                                  placebo_percent_effect_mean_lower_bound,
                                  drug_percent_effect_mean_upper_bound,
                                  drug_percent_effect_mean_lower_bound,
                                  placebo_percent_effect_std_dev,
                                  drug_percent_effect_std_dev,
                                  num_bins,
                                  hist_range,
                                  num_patients_per_trial_arm_step,
                                  final_num_patients_trial_arm_step,
                                  num_trials_per_stat_power_estim)

    store_evaluation_data(num_patients_per_trial_arm_array,
                          MPC_stat_power_array,
                          MPC_type_1_error_array,
                          true_drug_placebo_arm_hists_over_all_trial_sizes,
                          true_drug_treatment_arm_hists_over_all_trial_sizes,
                          false_drug_placebo_arm_hists_over_all_trial_sizes,
                          false_drug_treatment_arm_hists_over_all_trial_sizes,
                          evaluation_data_dir,
                          evaluation_data_file_name)
    
    print(true_drug_placebo_arm_hists_over_all_trial_sizes.shape)
    print(true_drug_treatment_arm_hists_over_all_trial_sizes.shape)
    print(false_drug_placebo_arm_hists_over_all_trial_sizes.shape)
    print(false_drug_treatment_arm_hists_over_all_trial_sizes.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(2*num_patients_per_trial_arm_array, 100*MPC_stat_power_array)
    plt.plot(2*num_patients_per_trial_arm_array, 100*MPC_type_1_error_array)
    plt.show()


if(__name__=='__main__'):

    start_time_in_seconds = time.time()

    main()

    total_runtime_in_seconds_str = str(np.round((time.time() - start_time_in_seconds)/60, 3)) + ' minutes'
    print(total_runtime_in_seconds_str)
