# DCGMM-B outliers
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id 112421 --epochs 30 \
--load_task 0 \
--measuring_points 10 \
--model_type Stacked_GMM_Outliers \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 --L4_wait 0.2 --L2_convMode False --L2_regularizer_delta 0.05 --L2_lambda_pi 0 \
--no_avg_metrics scores blah \
--sampling_layer 4 --outlier_detection_layer 4
#./tmp.bash

python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 112421 --epochs 0.1 \
--load_task 1 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--dataset_file MNIST --batch_size 100 \
--L2_convMode False --L2_K 25 --L4_K 25 \
--no_avg_metrics scores blah \
--sampling_layer 4 --outlier_detection_layer 4
./tmp.bash

python3 outliersFromJson.py ./112421_stacked_gmm_outliers.json
mv toc.png tocB.png

