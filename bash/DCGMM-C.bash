# 2GMM pooling 
python3 -m DCGMM.experiment.Experiment_GMM \
--exp_id 1008 \
--model_type Stacked_GMM \
--dataset_file MNIST \
--dataset_name mnist \
--data_type 32 \
--T1 1 4 \
--measuring_points 10 \
--vis_points 0 \
--epochs 50 \
--log_level DEBUG \
--batch_size 100 \
--sampling_batch_size 100 \
--sampling_points 0 \
--nr_sampling_batches      1       \
--save_All True \
--sampling_layer 5 \
--comment xxxxxxxxxxxxxxxxx \
--L1 Folding_Layer \
--L1_patch_width  8 \
--L1_patch_height 8 \
--L1_stride_x 1 \
--L1_stride_y 1 \
--L1_sharpening_iterations 0 \
--L1_sharpening_rate 0.0 \
--comment xxxxxxxxxxxxxxxxx 21x21\
--L2 GMM_Layer \
--L2_K 25 \
--L2_mode diag \
--L2_somSigmaInf 0.01 \
--L2_eps0 0.011 \
--L2_epsInf 0.01 \
--L2_convMode True \
--L2_energy mc \
--L2_regularizer_delta 0.05  \
--L2_sigmaUpperBound 20 \
--L2_covariance_mode variance \
--L2_muInit 0.1 \
--L2_lambda_pi 0.2  \
--L2_lambda_mu 1.  \
--L2_lambda_sigma 0.1 \
--L2_wait 0 \
--L2_sampling_I -1 \
--L2_sampling_divisor 20 \
--L2_sampling_S 3 \
--comment xxxxxxxxxxxxxxxxx \
--L3 MaxPooling_Layer \
--L3_kernel_size_x 2 \
--L3_kernel_size_y 2 \
--L3_stride_x 2 \
--L3_stride_y 2 \
--comment OOOOOOOOOOOOOOOOOOOOOOOOOOOO 11x11 \
--L4 Folding_Layer \
--L4_patch_width 11 \
--L4_patch_height 11 \
--L4_stride_x 1 \
--L4_stride_y 1 \
--L4_sharpening_iterations 0 \
--L4_sharpening_rate 0.0 \
--comment xxxxxxxxxxxxxxxxx \
--L5 GMM_Layer \
--L5_K 36 \
--L5_mode diag \
--L5_somSigmaInf 0.01 \
--L5_eps0 0.011 \
--L5_epsInf 0.01 \
--L5_convMode True \
--L5_energy mc \
--L5_regularizer_delta 0.05  \
--L5_sigmaUpperBound 20 \
--L5_covariance_mode variance \
--L5_muInit 0.1 \
--L5_lambda_pi 0.2  \
--L5_lambda_mu 1.  \
--L5_lambda_sigma 0.1 \
--L5_wait 0.0 \
--L5_sampling_I -1 \
--L5_sampling_divisor 20 \
--L5_sampling_S 3 \
--comment xxxxxxxxxxxxxxxxx \
--L6 Linear_Classifier_Layer \
--L6_num_classes 10 \
--L6_regEps 0.01 \
--L6_energy ce \
--L6_input 3
