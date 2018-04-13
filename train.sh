#!/bin/bash
cd ~/Repos/exjobb/network/

EPISODES=1
PLOT_FREQ=500
PRINT_FREQ=10
BATCH_SIZE=32

python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/SGD_lr10e-6_wd0_m09/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0, momentum=0.9, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/SGD_lr10e-6_wd01_m09/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0.1, momentum=0.9, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/SGD_lr10e-6_wd0_m2/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0, momentum=2, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/SGD_lr10e-6_wd01_m2/ -o "SGD( model.parameters(), lr=1e-6, weight_decay=0.1, momentum=2, nesterov=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/Adam_lr10e-6_wd0/ -o "Adam( model.parameters(), lr=1e-6, weight_decay=0, amsgrad=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/Adam_lr10e-6_wd01/ -o "Adam( model.parameters(), lr=1e-6, weight_decay=0.1, amsgrad=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/Adam_lr10e-5_wd0/ -o "Adam( model.parameters(), lr=1e-5, weight_decay=0, amsgrad=True)"
python3 main.py -e 1 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/Adam_lr10e-5_wd01/ -o "Adam( model.parameters(), lr=1e-5, weight_decay=0.1, amsgrad=True)"
