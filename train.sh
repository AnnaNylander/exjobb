#!/bin/bash
cd ~/Repos/exjobb/network/

EPISODES=1
PLOT_FREQ=500
PRINT_FREQ=10
BATCH_SIZE=16
CLASS=cnn_bias.CNNBiasAll
SAVE_NAME=CNNBiasAll

python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/Adam_lr1e-5_wd0/ -o "Adam( model.parameters(), lr=1e-5, weight_decay=0, amsgrad=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/Adam_lr1e-6_wd01/ -o "Adam( model.parameters(), lr=1e-6, weight_decay=0.1, amsgrad=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/Adam_lr1e-6_wd0/ -o "Adam( model.parameters(), lr=1e-6, weight_decay=0, amsgrad=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/Adam_lr1e-5_wd01/ -o "Adam( model.parameters(), lr=1e-5, weight_decay=0.1, amsgrad=True)"

python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/SGD_lr1e-6_wd0_m09/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0, momentum=0.9, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/SGD_lr1e-6_wd0_m05/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0, momentum=0.5, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/SGD_lr1e-6_wd01_m09/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0.1, momentum=0.9, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a $CLASS -d fruit_salad/ -s ${SAVE_NAME}/SGD_lr1e-5_wd01_m05/ -o "SGD( model.parameters(), lr=1e-5, weight_decay=0, momentum=0.5, nesterov=True)"
