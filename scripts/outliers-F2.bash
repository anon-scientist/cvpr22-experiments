# DCGMM-F outliers no pooling
python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id 62514 --epochs 20 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--dataset_file MNIST --batch_size 100 \
--sampling_layer 8 --L4_wait 0.1 --L6_wait 0.2 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--no_avg_metrics scores blah \
--L2_K 36 --L4_K 49 --L6_K 49 --L8_K 49
#./tmp.bash

python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 62514 --epochs 0.05 \
--load_task 1 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 2 \
--L1_target_layer 4 --L1_reconstruction_weight 0.1 --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--dataset_file MNIST --batch_size 100 \
--L2_K 36 --L4_K 49 --L6_K 49 --L8_K 49 \
--sampling_layer 8 --L4_wait 0.1 --L6_wait 0.15 --L8_wait 0.2
#./tmp.bash

python3 outliersFromJson.py ./62514_stacked_gmm_outliers.json
mv toc.png tocF2.png


