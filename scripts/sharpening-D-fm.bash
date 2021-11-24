# DCGMM-D outliers: 3 GMMs pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-D.bash --T1 0 1 2 3 4 5 6 7 8 9  --exp_id 12411 --epochs 20 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--dataset_name fashion_mnist --batch_size 100 \
--sampling_layer 8 --L5_wait 0.1 --L8_wait 0.2 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--no_avg_metrics scores blah \
--L2_K 25 --L5_K 25 --L8_K 25 --L2_convMode False --L5_convMode False --L8_convMode False
# ---
chmod 777 ./tmp.bash
./tmp.bash

# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-D.bash --T1 0 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 12411 --epochs 0.05 \
--load_task 1 \
--dataset_name fashion_mnist --batch_size 100 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--perform_sampling True \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L5_sampling_S 2 --L8_sampling_I -2 \
--L1_target_layer 2 --L1_reconstruction_weight 0. --L1_sharpening_rate 6 --L1_sharpening_iterations 00 \
--L4_target_layer 5 --L4_reconstruction_weight 0. --L4_sharpening_rate 6 --L4_sharpening_iterations 00 \
--L7_target_layer 8 --L7_reconstruction_weight 0. --L7_sharpening_rate 0.01 --L7_sharpening_iterations 0 \
--L2_K 25 --L5_K 25 --L8_K 25 --L2_convMode False --L5_convMode False \
--sampling_layer 8 --L5_wait 0.1 --L8_wait 0.2
# ---
chmod 777 ./tmp.bash
./tmp.bash
exp_id=12411
python3 -m DCGMM.utils.vis --what mus --prefix ./results/${exp_id}_sampling_ --channels 1  --out ./results/${exp_id}_sharp.png 


