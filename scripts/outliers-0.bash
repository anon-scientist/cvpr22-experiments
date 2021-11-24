# DCGMM-0 outliers, two stacked global layers
python3 -m DCGMM.utils.executeExp bash/DCGMM-0.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id 7429 --epochs 15 \
--model_type Stacked_GMM_Outliers \
--dataset_file MNIST --batch_size 100 \
--L2_K 49 --L4_K 49  --L4_wait 0.1 \
--sampling_layer 4 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 4 \
--no_avg_metrics scores blah 
#./tmp.bash

python3 -m utils.executeExp bash/DCGMM-0.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 7429 --epochs 10 \
--load_task 1 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 4 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L3_sampling_S 2 \
--L1_target_layer 3 --L1_reconstruction_weight 0.1 --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--dataset_file MNIST --batch_size 100 \
--L2_K 49 --L4_K 49 --L2_wait 1.0 --L4_wait 1.0 \
--sampling_layer 4 
chmod 777 tmp.bash
./tmp.bash

python3 -m utils.outliersFromJson ./7429_stacked_gmm_outliers.json
mv toc.png toc0.png

