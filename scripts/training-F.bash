# DCGMM-F outliers 4 GMMs no pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 0 1 2 3 4 5 6 7 8 9  --exp_id dynamics --epochs 3  \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--sampling_layer 8 --L4_wait 0.1 --L6_wait 0.2 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--no_avg_metrics scores blah \
--L2_K 25 --L4_K 25 --L6_K 25 --L8_K 49 \
--L2_convMode True --L4_convMode True --L6_convMode True --L8_convMode True \
--dataset_name fashion_mnist --batch_size 100 \
--measuring_points 50 \
--L2_wait 0 --L4_wait 0.1 --L6_wait 0.2 --L8_wait 0.3
# ---
chmod 777 ./tmp.bash
./tmp.bash

