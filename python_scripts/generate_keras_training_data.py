from clinical_trial_generation import generate_true_and_false_drug_histograms
import numpy as np
import json
import time
import sys


def generate_training_data(one_or_two,
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
                           hist_range,
                           num_training_samples_per_classification):

    placebo_arm_hists = np.zeros((2*num_training_samples_per_classification, num_bins, 1))
    drug_arm_hists    = np.zeros((2*num_training_samples_per_classification, num_bins, 1))
    labels            = np.zeros(2*num_training_samples_per_classification)

    for bi_sample_index in range(num_training_samples_per_classification):

        [true_drug_placebo_arm_rotated_percent_change_hist, 
         true_drug_treatment_arm_rotated_percent_change_hist,
         false_drug_placebo_arm_rotated_percent_change_hist,
         false_drug_treatment_arm_rotated_percent_change_hist] = \
             generate_true_and_false_drug_histograms(one_or_two,
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
        
        placebo_arm_hists[2*bi_sample_index,     :, 0] = true_drug_placebo_arm_rotated_percent_change_hist
        drug_arm_hists[2*bi_sample_index,        :, 0] = true_drug_treatment_arm_rotated_percent_change_hist
        placebo_arm_hists[2*bi_sample_index + 1, :, 0] = false_drug_placebo_arm_rotated_percent_change_hist
        drug_arm_hists[2*bi_sample_index    + 1, :, 0] = false_drug_treatment_arm_rotated_percent_change_hist

        labels[2*bi_sample_index]     = 1
        labels[2*bi_sample_index + 1] = 0

        print('training sample #' + str(2*bi_sample_index + 1) + ' done')
        print('training sample #' + str(2*bi_sample_index + 2) + ' done')
    
    return [placebo_arm_hists, 
            drug_arm_hists, 
            labels]


def store_training_data(placebo_arm_hists, 
                        drug_arm_hists, 
                        labels,
                        training_data_dir,
                        training_data_file_name):

    training_data_file_path = training_data_dir + '/' + training_data_file_name + '.json'

    data = []
    data.append(placebo_arm_hists.tolist())
    data.append(drug_arm_hists.tolist())
    data.append(labels.tolist())

    with open(training_data_file_path, 'w+') as json_training_data_storage_file:
        json.dump(data, json_training_data_storage_file)


def get_inputs():

    one_or_two = sys.argv[1]

    num_patients_per_trial = int(sys.argv[2])

    placebo_percent_effect_mean_upper_bound = float(sys.argv[3])
    placebo_percent_effect_mean_lower_bound = float(sys.argv[4])

    drug_percent_effect_mean_upper_bound = float(sys.argv[5])
    drug_percent_effect_mean_lower_bound = float(sys.argv[6])

    placebo_percent_effect_std_dev = float(sys.argv[7])
    drug_percent_effect_std_dev    = float(sys.argv[8])

    num_training_samples_per_classification = int(sys.argv[9])

    training_data_file_name = sys.argv[10]

    #======================================================================================================#

    if(num_patients_per_trial % 2):

        raise ValueError('The number of patients per trial needs to divisible by 2.')

    num_placebo_arm_patients = int(num_patients_per_trial/2)
    num_drug_arm_patients    = int(num_patients_per_trial/2)

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count = 4

    num_bins = 80
    hist_range = [-3, 1]

    import os
    training_data_dir = os.getcwd()

    return [one_or_two,
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
            hist_range,
            num_training_samples_per_classification,
            training_data_dir,
            training_data_file_name]


def main():

    [one_or_two,
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
     hist_range,
     num_training_samples_per_classification,
     training_data_dir,
     training_data_file_name] = \
         get_inputs()

    [placebo_arm_hists, 
     drug_arm_hists, 
     labels] = \
         generate_training_data(one_or_two,
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
                                hist_range,
                                num_training_samples_per_classification)
    
    store_training_data(placebo_arm_hists, 
                        drug_arm_hists, 
                        labels,
                        training_data_dir,
                        training_data_file_name)


if(__name__=='__main__'):

    start_time_in_seconds = time.time()

    main()

    stop_time_in_seconds = time.time()
    total_runtime_in_minutes_str = str(np.round((stop_time_in_seconds - start_time_in_seconds)/60, 3)) + ' minutes'

    print(total_runtime_in_minutes_str)
