#!/bin/bash

cd ~/Repos/exjobb/

EPOCHS=2
PLOT_FREQ=5000
PRINT_FREQ=10
BATCH_SIZE=12
DATASET=flexible/
N_DRIVE_FRAMES=100 # The number of frames that the car should drive in CARLA

# These values are to be set for every network (pure regression does not use CLUSTER_PATH)
#FOLDER_NAME=test_deluxe/regression_idea/pc_reg_1_with_20_pc/
#NETWORK_FILE=pc_reg_1
#NETWORK_CLASS=PCNet
CLUSTER_PATH='/home/annaochjacob/Repos/exjobb/preprocessing/path_clustering/'

echo "*** Evaluating network on test set in order to generate predictions ***"
################################################################################
# PURE REGRESSION IDEA
################################################################################
#-------------------------------------------------------------------------------
#echo "*** regression_idea/cnn_bias_all***"
#FOLDER_NAME=test_deluxe/regression_idea/cnn_bias_all/
#NETWORK_FILE=cnn_bias_flexible
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/gott_och_blandat_1***"
#FOLDER_NAME=test_deluxe/regression_idea/gott_och_blandat_1/
#NETWORK_FILE=gott_och_blandat_1
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_channels_1***"
#FOLDER_NAME=test_deluxe/regression_idea/more_channels_1/
#NETWORK_FILE=more_channels_1
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_channels_2***"
#FOLDER_NAME=test_deluxe/regression_idea/more_channels_2/
#NETWORK_FILE=more_channels_2
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_dilation_1_attempt_2***"
#FOLDER_NAME=test_deluxe/regression_idea/more_dilation_1_attempt_2/
#NETWORK_FILE=more_dilation_1
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_layers_1***"
#FOLDER_NAME=test_deluxe/regression_idea/more_layers_1/
#NETWORK_FILE=more_layers_1
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------

#python3 save_visualization.py --prediction -m ${FOLDER_NAME} -r test/ -d ${DATASET}test/

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_layers_2***"
#FOLDER_NAME=test_deluxe/regression_idea/more_layers_2/
#NETWORK_FILE=more_layers_2
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_layers_3***"
#FOLDER_NAME=test_deluxe/regression_idea/more_layers_3/
#NETWORK_FILE=more_layers_3
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_layers_4***"
#FOLDER_NAME=test_deluxe/regression_idea/more_layers_4/
#NETWORK_FILE=more_layers_4
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b 6 \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_maxpool_1***"
#FOLDER_NAME=test_deluxe/regression_idea/more_maxpool_1/
#NETWORK_FILE=more_maxpool_1
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b 24 \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#-------------------------------------------------------------------------------
#echo "*** regression_idea/more_size_stride_1***"
#FOLDER_NAME=test_deluxe/regression_idea/more_size_stride_1/
#NETWORK_FILE=more_size_stride_1
#NETWORK_CLASS=CNNBiasAll

#python3 network/main.py \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate


################################################################################
# CLUSTER IDEA
################################################################################
#-------------------------------------------------------------------------------
#echo "*** cluster_idea/cluster_1_10_cl_7_pc***"
#FOLDER_NAME=test_deluxe/cluster_idea/cluster_1_10_cl_7_pc/
#NETWORK_FILE=cluster_1
#NETWORK_CLASS=ClusterNet

#python3 network/main_clusters.py \
#-ncl 10 \
#-npc 7 \
#-cpath ${CLUSTER_PATH}10_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
#-------------------------------------------------------------------------------
#echo "*** cluster_idea/cluster_1_20_cl_7_pc_continued***"
#FOLDER_NAME=test_deluxe/cluster_idea/cluster_1_20_cl_7_pc_continued/
#NETWORK_FILE=cluster_1
#NETWORK_CLASS=ClusterNet

#python3 network/main_clusters.py \
#-ncl 20 \
#-npc 7 \
#-cpath ${CLUSTER_PATH}20_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

#python3 save_visualization.py --prediction -m ${FOLDER_NAME} -r test/ -d ${DATASET}test/

#-------------------------------------------------------------------------------
#echo "*** cluster_idea/cluster_1_100_cl_7_pc***"
#FOLDER_NAME=test_deluxe/cluster_idea/cluster_1_100_cl_7_pc/
#NETWORK_FILE=cluster_1
#NETWORK_CLASS=ClusterNet

#python3 network/main_clusters.py \
#-ncl 100 \
#-npc 7 \
#-cpath ${CLUSTER_PATH}100_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
################################################################################
#echo "*** cluster_idea/cluster_1_100_cl_7_pc_continued***"
#FOLDER_NAME=test_deluxe/cluster_idea/cluster_1_100_cl_7_pc_continued/
#NETWORK_FILE=cluster_1
#NETWORK_CLASS=ClusterNet

#python3 network/main_clusters.py \
#-ncl 100 \
#-npc 7 \
#-cpath ${CLUSTER_PATH}100_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate

################################################################################
# PCA IDEA
################################################################################
#echo "*** pca_idea/pc_reg_1_fixed_5_pc***"
#FOLDER_NAME=test_deluxe/pca_idea/pc_reg_1_fixed_5_pc/
#NETWORK_FILE=pc_reg_1
#NETWORK_CLASS=PCNet

#python3 network/main_pc_reg.py \
#-npc 5 \
#-cpath ${CLUSTER_PATH}20_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
################################################################################
#echo "*** pca_idea/pc_reg_1_fixed_7_pc***"
#FOLDER_NAME=test_deluxe/pca_idea/pc_reg_1_fixed_7_pc/
#NETWORK_FILE=pc_reg_1
#NETWORK_CLASS=PCNet

#python3 network/main_pc_reg.py \
#-npc 7 \
#-cpath ${CLUSTER_PATH}20_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
################################################################################
#echo "*** pca_idea/pc_reg_1_fixed_10_pc***"
#FOLDER_NAME=test_deluxe/pca_idea/pc_reg_1_fixed_10_pc/
#NETWORK_FILE=pc_reg_1
#NETWORK_CLASS=PCNet

#python3 network/main_pc_reg.py \
#-npc 10 \
#-cpath ${CLUSTER_PATH}20_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
################################################################################
#echo "*** pca_idea/pc_reg_1_fixed_5_pc_continued***"
#FOLDER_NAME=test_deluxe/pca_idea/pc_reg_1_fixed_5_pc_continued/
#NETWORK_FILE=pc_reg_1
#NETWORK_CLASS=PCNet

#python3 network/main_pc_reg.py \
#-npc 5 \
#-cpath ${CLUSTER_PATH}20_clusters/ \
#-e $EPOCHS \
#-b $BATCH_SIZE \
#-pl $PLOT_FREQ \
#-p $PRINT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
################################################################################
echo "*** pca_idea/pc_reg_1_fixed_20_pc***"
FOLDER_NAME=test_loss_before_continue/pc_reg_1_fixed_20_pc/
NETWORK_FILE=pc_reg_1
NETWORK_CLASS=PCNet

python3 network/main_pc_reg.py \
-npc 20 \
-cpath ${CLUSTER_PATH}20_clusters/ \
-e $EPOCHS \
-b $BATCH_SIZE \
-pl $PLOT_FREQ \
-p $PRINT_FREQ \
-a ${NETWORK_FILE}.${NETWORK_CLASS} \
-d $DATASET \
-s ${FOLDER_NAME} \
-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
-off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
--resume ${FOLDER_NAME}all_time_best.pt \
--evaluate
################################################################################
# SEMSEG IDEA
################################################################################
echo "*** semseg/semseg_6_s_6_r***"
FOLDER_NAME=test_deluxe/semseg/semseg_6_s_6_r/
NETWORK_FILE=semseg
NETWORK_CLASS=Network

python3 network/main_semseg.py \
-e 10 \
-b 12 \
-pl $PLOT_FREQ \
-a ${NETWORK_FILE}.${NETWORK_CLASS} \
-p $PRINT_FREQ \
-d $DATASET \
-s ${FOLDER_NAME} \
-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
-off '5 10 15 20 25 30' \
-res 150 150 \
-r 3.0 \
--resume ${FOLDER_NAME}all_time_best.pt \
--evaluate
################################################################################
echo "*** semseg/semseg_6_s_4_r***"
FOLDER_NAME=test_deluxe/semseg/semseg_6_s_4_r/
NETWORK_FILE=semseg
NETWORK_CLASS=Network

python3 network/main_semseg.py \
-e 10 \
-b 12 \
-pl $PLOT_FREQ \
-a ${NETWORK_FILE}.${NETWORK_CLASS} \
-p $PRINT_FREQ \
-d $DATASET \
-s ${FOLDER_NAME} \
-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
-off '5 10 15 20 25 30' \
-res 150 150 \
-r 4.0 \
--resume ${FOLDER_NAME}all_time_best.pt \
--evaluate
################################################################################
#echo "*** semseg/semseg_6_s_10_r_continued***"
#FOLDER_NAME=test_deluxe/semseg/semseg_6_s_10_r_continued/
#NETWORK_FILE=semseg
#NETWORK_CLASS=Network

#python3 network/main_semseg.py \
#-e 10 \
#-b 12 \
#-pl $PLOT_FREQ \
#-a ${NETWORK_FILE}.${NETWORK_CLASS} \
#-p $PRINT_FREQ \
#-d $DATASET \
#-s ${FOLDER_NAME} \
#-ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
#-off '5 10 15 20 25 30' \
#-res 150 150 \
#-r 4.0 \
#--resume ${FOLDER_NAME}all_time_best.pt \
#--evaluate
