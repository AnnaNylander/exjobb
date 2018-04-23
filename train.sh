#!/bin/bash
cd ~/Repos/exjobb/network/

EPOCHS=1
PLOT_FREQ=500
PRINT_FREQ=10
BATCH_SIZE=16
FOLDER_NAME=Adam_lr1e-5_wd1e-5/

python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_deluxe.CNNOnly -d fruit_salad/ -s CNNOnly/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b 8 -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_deluxe.CNNLSTM -d fruit_salad/ -s CNNLSTM/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b 4 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.LSTMNet -pf 29 -fs 1 -d fruit_salad/ -s LSTMNet/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b 4 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.LSTMNetBi -pf 29 -fs 1 -d fruit_salad/ -s LSTMNetBi/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b 4 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.GRUNet -pf 29 -fs 1 -d fruit_salad/ -s GRUNet/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork2 -d fruit_salad/ -s SmallerNetwork2/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasFirst -d fruit_salad/ -s CNNBiasFirst/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasAll -d fruit_salad/ -s CNNBiasAll/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasLast -d fruit_salad/ -s CNNBiasLast/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
