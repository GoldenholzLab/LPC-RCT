from seizure_diary_generation import generate_NV_model_patient_pop_params
from seizure_diary_generation import generate_baseline_seizure_diaries
from seizure_diary_generation import generate_maintenance_seizure_diaries
from seizure_diary_generation import apply_percent_effects_to_seizure_diaries
from endpoint_functions import calculate_percent_changes
import matplotlib.pyplot as plt
import numpy as np
import time


def generate_one_trial_MPC(one_or_two,
                           num_patients,
                           num_baseline_months,
                           num_maintenance_months,
                           baseline_time_scale,
                           maintenance_time_scale,
                           minimum_cumulative_baseline_seizure_count):

    [NV_model_monthly_means, 
     NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_patients,
                                              one_or_two)
    
    baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(NV_model_monthly_means,
                                          NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)
    
    maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(NV_model_monthly_means,
                                             NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months) 

    percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  maintenance_seizure_diaries)
    
    MPC = np.median(percent_changes)

    return MPC


def generate_MPC_arrays_for_both_NV_models(num_patients,
                                           num_baseline_months,
                                           num_maintenance_months,
                                           baseline_time_scale,
                                           maintenance_time_scale,
                                           minimum_cumulative_baseline_seizure_count,
                                           num_trials):

    NV_model_one_MPC_array = np.zeros(num_trials)
    NV_model_two_MPC_array = np.zeros(num_trials)

    for trial_index in range(num_trials):

        NV_model_one_MPC_array[trial_index] = \
            generate_one_trial_MPC('one',
                                   num_patients,
                                   num_baseline_months,
                                   num_maintenance_months,
                                   baseline_time_scale,
                                   maintenance_time_scale,
                                   minimum_cumulative_baseline_seizure_count)
            
        NV_model_two_MPC_array[trial_index] = \
            generate_one_trial_MPC('two',
                                   num_patients,
                                   num_baseline_months,
                                   num_maintenance_months,
                                   baseline_time_scale,
                                   maintenance_time_scale,
                                   minimum_cumulative_baseline_seizure_count)

        print('trial #' + str(trial_index + 1))
    
    return [NV_model_one_MPC_array,
            NV_model_two_MPC_array]


if(__name__=='__main__'):

    one_or_two = 'one'

    num_patients = 10

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count = 4

    placebo_percent_effect_mean_upper_bound = 0.4
    placebo_percent_effect_mean_lower_bound = 0
    placebo_percent_effect_std_dev = 0.05

    num_trials = 2000

    #=================================================================================#

    [NV_model_one_MPC_array,
     NV_model_two_MPC_array] = \
         generate_MPC_arrays_for_both_NV_models(num_patients,
                                                num_baseline_months,
                                                num_maintenance_months,
                                                baseline_time_scale,
                                                maintenance_time_scale,
                                                minimum_cumulative_baseline_seizure_count,
                                                num_trials)

    plt.figure()
    plt.boxplot([100*NV_model_one_MPC_array, 100*NV_model_two_MPC_array], sym='')
    plt.xticks([1,2], ['NV model one', 'NV model two'])
    plt.ylabel('placebo arm MPC')
    plt.title('Placebo Arm MPCs for both NV Models')
    plt.show()
