# 4GMM no pooling
python3 -m DCGMM.experiment.Experiment_GMM \
--exp_id 1004 \
--model_name Stacked_GMM  \
--dataset_file MNIST \
--dataset_name mnist \
--data_type 32 \
--T1 1 4 \
--measuring_points 10 \
--vis_points 0 \
--epochs 30 \
--log_level DEBUG \
--batch_size 100 \
--sampling_batch_size 100 \
--sampling_points 0 \
--nr_sampling_batches      1       \
--save_All True \
--comment 0000000000000000000000000000 \
--L1 Folding_Layer \
--L1_patch_width  3 \
--L1_patch_height 3 \
--L1_stride_x 1 \
--L1_stride_y 1 \
--L1_sharpening_iterations 0 \
--L1_sharpening_rate 0.0 \
--comment OOOOOOOOOOOO_Input_26x26 \
--L2 GMM_Layer \
--L2_K 25 \
--L2_mode diag \
--L2_somSigmaInf 0.01 \
--L2_eps0 0.011 \
--L2_epsInf 0.01 \
--L2_convMode True \
--L2_energy mc \
--L2_delta 0.05  \
--L2_sigmaUpperBound 20 \
--L2_covariance_mode variance \
--L2_muInit 0.1 \
--L2_lambda_pi 0.  \
--L2_lambda_mu 1.  \
--L2_lambda_sigma 0.1 \
--L2_wait 0 \
--L2_sampling_I -1 \
--L2_sampling_S 3 \
--L2_sampling_divisor 20 \
--comment OOOOOOOOOOOOOOOOOOOOOOOOOOOO \
--L3 Folding_Layer \
--L3_patch_width  4 \
--L3_patch_height 4 \
--L3_stride_x 2 \
--L3_stride_y 2 \
--L3_sharpening_iterations 0 \
--L3_sharpening_rate 0.0 \
--comment OOOOOOOOOO_Input_12x12 \
--L4 GMM_Layer \
--L4_K 25 \
--L4_mode diag \
--L4_somSigmaInf 0.01 \
--L4_eps0 0.011 \
--L4_epsInf 0.01 \
--L4_convMode True \
--L4_energy mc \
--L4_delta 0.05  \
--L4_sigmaUpperBound 20 \
--L4_covariance_mode variance \
--L4_muInit 0.1 \
--L4_lambda_pi 0.  \
--L4_lambda_mu 1.  \
--L4_lambda_sigma 0.1 \
--L4_wait 0.0 \
--L4_sampling_I -1 \
--L4_sampling_S 3 \
--L4_sampling_divisor 20 \
--comment OOOOOOOOOOOOOOOOOOOOOOOOOOOO 12x12 \
--L5 Folding_Layer \
--L5_patch_width  4 \
--L5_patch_height 4 \
--L5_stride_x 2 \
--L5_stride_y 2 \
--L5_sharpening_iterations 0 \
--L5_sharpening_rate 0.0 \
--comment OOOOOOOOOOO00000 5x5 \
--L6 GMM_Layer \
--L6_K 25 \
--L6_mode diag \
--L6_somSigmaInf 0.01 \
--L6_eps0 0.011 \
--L6_epsInf 0.01 \
--L6_convMode True \
--L6_energy mc \
--L6_delta 0.05  \
--L6_sigmaUpperBound 20 \
--L6_covariance_mode variance \
--L6_muInit 0.1 \
--L6_lambda_pi 0.  \
--L6_lambda_mu 1.  \
--L6_lambda_sigma 0.1 \
--L6_wait 0.0 \
--L6_sampling_I -1 \
--L6_sampling_S 3 \
--L6_sampling_divisor 20 \
--comment OOOOOOOOOOOOOOOOOOOOOOOOOOOO \
--L7 Folding_Layer \
--L7_patch_width  5 \
--L7_patch_height 5 \
--L7_stride_x 2 \
--L7_stride_y 2 \
--L7_sharpening_iterations 0 \
--L7_sharpening_rate 0.0 \
--comment OOOOOOOOOOO_Input_1x1 \
--L8 GMM_Layer \
--L8_K 25 \
--L8_mode diag \
--L8_somSigmaInf 0.01 \
--L8_eps0 0.011 \
--L8_epsInf 0.01 \
--L8_convMode True \
--L8_energy mc \
--L8_delta 0.05  \
--L8_sigmaUpperBound 20 \
--L8_covariance_mode variance \
--L8_muInit 0.1 \
--L8_lambda_pi 0.  \
--L8_lambda_mu 1.  \
--L8_lambda_sigma 0.1 \
--L8_wait 0.0 \
--L8_sampling_I -1 \
--L8_sampling_S 3 \
--L8_sampling_divisor 20 \
--comment OOOOOOOOOOOOOOOOOOOOOOOOOOOO \
--L9 Linear_Classifier_Layer \
--L9_energy mse \
--L9_num_classes 10 \
--L9_regEps 0.01


