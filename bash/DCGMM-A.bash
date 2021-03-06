# single GMM layer
python3 -m DCGMM.experiment.Experiment_GMM \
--exp_id 1000 \
--model_type Stacked_GMM \
--dataset_file MNIST \
--dataset_name mnist \
--data_type 32 \
--T1 1 4 \
--measuring_points 10 \
--vis_points 0 \
--epochs 40 \
--log_level DEBUG \
--batch_size 100 \
--sampling_batch_size 100 \
--sampling_points 10 \
--nr_sampling_batches      1       \
--save_All True \
--sampling_layer 2 \
--comment xxxxxxxxxxxxxxx \
--L1 Folding_Layer \
--L1_patch_width  28 \
--L1_patch_height 28 \
--L1_stride_x 2 \
--L1_stride_y 2 \
--L1_sharpening_iterations 0 \
--L1_sharpening_rate 0.0 \
--comment 000000 Input 1/1/784 \
--L2 GMM_Layer \
--L2_K 25 \
--L2_mode diag \
--L2_l 4 \
--L2_somSigmaInf 0.01 \
--L2_eps0 0.011 \
--L2_epsInf 0.01 \
--L2_convMode True \
--L2_energy mc \
--L2_regularizer_delta 0.05  \
--L2_sigmaUpperBound 20 \
--L2_covariance_mode variance \
--L2_muInit 0.1 \
--L2_lambda_pi 0.5  \
--L2_lambda_mu 1.  \
--L2_lambda_sigma 0.5 \
--L2_lambda_D 0.0 \
--L2_wait 0 \
--L2_use_pis False \
--L2_sigma_start -100 \
--comment 000000 Input 1/1/784 \
--L3 Linear_Classifier_Layer \
--L3_num_classes 10 \
--L3_energy ce \
--L3_regEps 0.01
