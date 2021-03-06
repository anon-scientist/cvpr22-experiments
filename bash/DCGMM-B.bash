# 2GMM no pooling 
python3 -m DCGMM.experiment.Experiment_GMM \
--exp_id 1001 \
--model_type Stacked_GMM_Outliers \
--dataset_file MNIST \
--dataset_name mnist \
--data_type 32 \
--T1 1 \
--measuring_points 10 \
--vis_points 0 \
--epochs 25 \
--log_level DEBUG \
--batch_size 100 \
--sampling_batch_size 100 \
--sampling_points 0 \
--nr_sampling_batches      1       \
--save_All True \
--sampling_layer 4 \
--comment xxxxxxxxxxxxxxxxx \
--L1 Folding_Layer \
--L1_patch_width  8 \
--L1_patch_height 8 \
--L1_stride_x 2 \
--L1_stride_y 2 \
--L1_sharpening_iterations 0 \
--L1_sharpening_rate 0.0 \
--L1_target_layer 4 \
--comment xxxxxxxxxxxxxxxxx \
--L2 GMM_Layer \
--L2_K 49 \
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
--L3 Folding_Layer \
--L3_patch_width 11 \
--L3_patch_height 11 \
--L3_stride_x 1 \
--L3_stride_y 1 \
--L3_sharpening_iterations 0 \
--L3_sharpening_rate 0.0 \
--comment xxxxxxxxxxxxxxxxx \
--L4 GMM_Layer \
--L4_K 36 \
--L4_mode diag \
--L4_somSigmaInf 0.01 \
--L4_eps0 0.011 \
--L4_epsInf 0.01 \
--L4_convMode True \
--L4_energy mc \
--L4_regularizer_delta 0.05  \
--L4_sigmaUpperBound 20 \
--L4_covariance_mode variance \
--L4_muInit 0.1 \
--L4_lambda_pi 0.2  \
--L4_lambda_mu 1.  \
--L4_lambda_sigma 0.1 \
--L4_wait 0.0 \
--L4_sampling_I -1 \
--L4_sampling_divisor 20 \
--L4_sampling_S 3 \
--comment xxxxxxxxxxxxxxxxx \
--L5 Linear_Classifier_Layer \
--L5_num_classes 10 \
--L5_regEps 0.01 \
--L5_energy ce \
#--L5_input 2

