from seizure_diary_generation import generate_NV_model_patient_pop_params
from seizure_diary_generation import generate_baseline_seizure_diaries
from seizure_diary_generation import generate_maintenance_seizure_diaries
from seizure_diary_generation import apply_percent_effects_to_seizure_diaries
from endpoint_functions import calculate_MPC_p_value
from scipy.stats import ranksums
import numpy as np


def generate_one_trial_seizure_diaries(one_or_two,
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
                                       drug_efficacy_presence,
                                       drug_percent_effect_mean_upper_bound=None,
                                       drug_percent_effect_mean_lower_bound=None,
                                       drug_percent_effect_std_dev=None):

    [placebo_arm_NV_model_monthly_means,
     placebo_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_placebo_arm_patients,
                                              one_or_two)
    
    [treatment_arm_NV_model_monthly_means,
     treatment_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_drug_arm_patients,
                                              one_or_two)

    [placebo_arm_NV_model_monthly_means,
     placebo_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_placebo_arm_patients,
                                              one_or_two)
    
    [treatment_arm_NV_model_monthly_means,
     treatment_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_drug_arm_patients,
                                              one_or_two)
    
    placebo_arm_baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(placebo_arm_NV_model_monthly_means,
                                          placebo_arm_NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)
    
    treatment_arm_baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(treatment_arm_NV_model_monthly_means,
                                          treatment_arm_NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)

    placebo_arm_maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(placebo_arm_NV_model_monthly_means,
                                             placebo_arm_NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months)
    
    treatment_arm_maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(treatment_arm_NV_model_monthly_means,
                                             treatment_arm_NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months)

    placebo_percent_effect_mean = \
        np.random.uniform(placebo_percent_effect_mean_lower_bound,
                          placebo_percent_effect_mean_upper_bound)

    placebo_percent_effects = \
        np.random.normal(placebo_percent_effect_mean,
                         placebo_percent_effect_std_dev,
                         num_placebo_arm_patients)

    placebo_arm_maintenance_seizure_diaries = \
        apply_percent_effects_to_seizure_diaries(placebo_arm_maintenance_seizure_diaries,
                                                 placebo_percent_effects)
    
    treatment_arm_maintenance_seizure_diaries = \
        apply_percent_effects_to_seizure_diaries(treatment_arm_maintenance_seizure_diaries,
                                                 placebo_percent_effects)
    
    if(drug_efficacy_presence == True):

        drug_percent_effect_mean = \
            np.random.uniform(drug_percent_effect_mean_lower_bound,
                              drug_percent_effect_mean_upper_bound)

        drug_percent_effects = \
            np.random.normal(drug_percent_effect_mean,
                             drug_percent_effect_std_dev,
                             num_drug_arm_patients)
        
        treatment_arm_maintenance_seizure_diaries = \
            apply_percent_effects_to_seizure_diaries(treatment_arm_maintenance_seizure_diaries,
                                                     drug_percent_effects)
    
    return [placebo_arm_baseline_seizure_diaries,
            placebo_arm_maintenance_seizure_diaries,
            treatment_arm_baseline_seizure_diaries,
            treatment_arm_maintenance_seizure_diaries]


def generate_noisy_stat_power_estim(one_or_two,
                                    num_placebo_arm_patients,
                                    num_drug_arm_patients,
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
                                    drug_efficacy_presence,
                                    num_trials):

    MPC_p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):
        
        [placebo_arm_baseline_seizure_diaries,
         placebo_arm_maintenance_seizure_diaries,
         treatment_arm_baseline_seizure_diaries,
         treatment_arm_maintenance_seizure_diaries] = \
             generate_one_trial_seizure_diaries(one_or_two,
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
                                                drug_efficacy_presence,
                                                drug_percent_effect_mean_upper_bound,
                                                drug_percent_effect_mean_lower_bound,
                                                drug_percent_effect_std_dev)

        MPC_p_value_array[trial_index] = \
            calculate_MPC_p_value(baseline_time_scale,
                                  maintenance_time_scale,
                                  placebo_arm_baseline_seizure_diaries,
                                  placebo_arm_maintenance_seizure_diaries,
                                  treatment_arm_baseline_seizure_diaries,
                                  treatment_arm_maintenance_seizure_diaries)

        print('trial #' + str(trial_index + 1) + ', done')

    MPC_stat_power = np.sum(MPC_p_value_array < 0.05)/num_trials

    return MPC_stat_power


def generate_noiseless_stat_power_estim(one_or_two,
                                        num_placebo_arm_patients,
                                        num_drug_arm_patients,
                                        placebo_percent_effect_mean_lower_bound,
                                        placebo_percent_effect_mean_upper_bound,
                                        drug_percent_effect_mean_lower_bound,
                                        drug_percent_effect_mean_upper_bound,
                                        placebo_percent_effect_std_dev,
                                        drug_percent_effect_std_dev,
                                        drug_efficacy_presence):
    
    MPC_p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):

        [placebo_arm_NV_model_monthly_baseline_means, 
         _] = \
             generate_NV_model_patient_pop_params(num_placebo_arm_patients,
                                                  one_or_two)
    
        [treatment_arm_NV_model_monthly_baseline_means, 
         _] = \
             generate_NV_model_patient_pop_params(num_drug_arm_patients,
                                                  one_or_two)
                                                  
        placebo_percent_effects = \
            np.random.normal(np.random.uniform(placebo_percent_effect_mean_lower_bound,
                                               placebo_percent_effect_mean_upper_bound),
                             placebo_percent_effect_std_dev,
                             num_placebo_arm_patients)
    
        drug_percent_effects = \
                np.random.normal(np.random.uniform(drug_percent_effect_mean_lower_bound,
                                                   drug_percent_effect_mean_upper_bound),
                                 drug_percent_effect_std_dev,
                                 num_drug_arm_patients)

        placebo_arm_NV_model_monthly_maintenance_means = \
            (1 - placebo_percent_effects)*placebo_arm_NV_model_monthly_baseline_means

        if(drug_efficacy_presence):
            treatment_arm_NV_model_monthly_maintenance_means = \
                (1 - placebo_percent_effects)*(1 - drug_percent_effects)*treatment_arm_NV_model_monthly_baseline_means
        else:
            treatment_arm_NV_model_monthly_maintenance_means = \
                (1 - placebo_percent_effects)*treatment_arm_NV_model_monthly_baseline_means
    
        placebo_arm_percent_changes   = 1 -   (placebo_arm_NV_model_monthly_maintenance_means/placebo_arm_NV_model_monthly_baseline_means)
        treatment_arm_percent_changes = 1 - (treatment_arm_NV_model_monthly_maintenance_means/treatment_arm_NV_model_monthly_baseline_means)

        [_, MPC_p_value_array[trial_index]] = ranksums(placebo_arm_percent_changes, treatment_arm_percent_changes)

        print('trial #' + str(trial_index + 1) +  ', done')

    MPC_stat_power = np.sum(MPC_p_value_array < 0.05)/num_trials

    return MPC_stat_power


if(__name__=='__main__'):

    one_or_two = 'two'

    num_placebo_arm_patients = 50
    num_drug_arm_patients    = 50

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count= 4

    placebo_percent_effect_mean_upper_bound = 0.4
    placebo_percent_effect_mean_lower_bound = 0

    drug_percent_effect_mean_upper_bound = 0.25
    drug_percent_effect_mean_lower_bound = 0.2

    placebo_percent_effect_std_dev = 0.05
    drug_percent_effect_std_dev    = 0.05

    drug_efficacy_presence = False

    num_trials = 2000

    #==========================================================+#

    noiseless_MPC_stat_power = \
        generate_noiseless_stat_power_estim(one_or_two,
                                        num_placebo_arm_patients,
                                        num_drug_arm_patients,
                                        placebo_percent_effect_mean_lower_bound,
                                        placebo_percent_effect_mean_upper_bound,
                                        drug_percent_effect_mean_lower_bound,
                                        drug_percent_effect_mean_upper_bound,
                                        placebo_percent_effect_std_dev,
                                        drug_percent_effect_std_dev,
                                        drug_efficacy_presence)
    
    noisy_MPC_stat_power = \
        generate_noisy_stat_power_estim(one_or_two,
                                        num_placebo_arm_patients,
                                        num_drug_arm_patients,
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
                                        drug_efficacy_presence,
                                        num_trials)

    print([np.round(100*noisy_MPC_stat_power, 3), np.round(100*noiseless_MPC_stat_power, 3)])

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.figure()
    plt.bar([1, 2], [100*noisy_MPC_stat_power, 100*noiseless_MPC_stat_power], tick_label=['noisy', 'noiseless'])
    plt.title('NV model ' + one_or_two + ' Type 1 Error, ' + str(num_placebo_arm_patients + num_drug_arm_patients) + ' patients/trial')
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
    plt.savefig('NV_model_' + one_or_two + '_noisy_vs_noiseless_type_1_error.png')
    plt.show()

