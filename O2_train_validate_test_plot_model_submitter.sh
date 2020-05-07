
training_samples_file_name='200000_NV_model_one_weekly_regular_level_16_training_samples'
validation_data_file_name='NV_model_one_weekly_regular_level_16_validation_data'
testing_data_file_name='NV_model_one_weekly_regular_level_16_testing_data'
model_storage_folder_name='NV_model_one_weekly_regular_level_16_keras_models'
num_training_samples_per_classification=75000

inputs[0]=$training_samples_file_name
inputs[1]=$validation_data_file_name
inputs[2]=$testing_data_file_name
inputs[3]=$model_storage_folder_name
inputs[4]=$num_training_samples_per_classification

sbatch O2_train_validate_test_plot_single_perceptron_model_wrapper.sh ${inputs[@]}

for dropout_rate in 0 0.1 0.3
do
    inputs[5]=$dropout_rate

    sbatch O2_train_validate_test_plot_3_layer_model_wrapper.sh ${inputs[@]}

done

for dropout_rate in 0 0.1 0.3
do
    inputs[5]=$dropout_rate

    for CNNet_1D_num_filters in 4 8
    do
        inputs[6]=$CNNet_1D_num_filters

        for CNNet_1D_kernel_size in 3 5 9
        do
            inputs[7]=$CNNet_1D_kernel_size

            for Leaky_ReLU_neg_slope in 0 0.2 0.4
            do
                inputs[8]=$Leaky_ReLU_neg_slope

                sbatch O2_train_validate_test_plot_simple_model_wrapper.sh ${inputs[@]}
                sbatch O2_train_validate_test_plot_intermediate_model_wrapper.sh ${inputs[@]}
                sbatch O2_train_validate_test_plot_complicated_model_wrapper.sh ${inputs[@]}

            done
        done
    done
done
