#!/bin/bash
cd ~/Repos/exjobb/network/

EPOCHS=2
PLOT_FREQ=500
PRINT_FREQ=10
BATCH_SIZE=16
FOLDER_NAME=trained_2_epochs/
DATASET=eukaryote/

# Recurrent network
python3 main.py -e $EPOCHS -b 1 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.LSTMNet -pf 29 -fs 1 -d $DATASET -s ${FOLDER_NAME}LSTMNet/ -o "SGD( model.parameters(), lr=3*1e-4, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b 1 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.LSTMNetBi -pf 29 -fs 1 -d $DATASET -s ${FOLDER_NAME}LSTMNetBi/ -o "SGD( model.parameters(), lr=3*1e-4, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b 1 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.GRUNet -pf 29 -fs 1 -d $DATASET -s ${FOLDER_NAME}GRUNet/ -o "SGD( model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)" --scheduler

python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_deluxe.CNNOnly -d $DATASET -s ${FOLDER_NAME}CNNOnly/ -bl -o "SGD( model.parameters(), lr=3*1e-4, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b 8 -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_deluxe.CNNLSTM -d $DATASET -s ${FOLDER_NAME}CNNLSTM/ -bl -o "SGD( model.parameters(), lr=3*1e-4, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d $DATASET -s ${FOLDER_NAME}SmallerNetwork1/ -bl -o "SGD( model.parameters(), lr=1e-5.5, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork2 -d $DATASET -s ${FOLDER_NAME}SmallerNetwork2/ -bl -o "SGD( model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasFirst -d $DATASET -s ${FOLDER_NAME}CNNBiasFirst/ -bl -o "SGD( model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasAll -d $DATASET -s ${FOLDER_NAME}CNNBiasAll/ -bl -o "SGD( model.parameters(), lr=1e-5.75, momentum=0.9, weight_decay=1e-5)" --scheduler
python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasLast -d $DATASET -s ${FOLDER_NAME}CNNBiasLast/ -bl -o "SGD( model.parameters(), lr=1e-5.75, momentum=0.9, weight_decay=1e-5)" --scheduler


# OLD TRAINING SCRIPT:

#EPOCHS=1
#PLOT_FREQ=500
#PRINT_FREQ=10
#BATCH_SIZE=16
#FOLDER_NAME=Adam_lr1e-5_wd1e-5/

#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_deluxe.CNNOnly -d fruit_salad/ -s CNNOnly/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b 8 -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_deluxe.CNNLSTM -d fruit_salad/ -s CNNLSTM/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b 4 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.LSTMNet -pf 29 -fs 1 -d fruit_salad/ -s LSTMNet/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b 4 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.LSTMNetBi -pf 29 -fs 1 -d fruit_salad/ -s LSTMNetBi/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b 4 -pl $PLOT_FREQ -p $PRINT_FREQ -a rnn.GRUNet -pf 29 -fs 1 -d fruit_salad/ -s GRUNet/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork1 -d fruit_salad/ -s SmallerNetwork1/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a network.SmallerNetwork2 -d fruit_salad/ -s SmallerNetwork2/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasFirst -d fruit_salad/ -s CNNBiasFirst/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasAll -d fruit_salad/ -s CNNBiasAll/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -a cnn_bias.CNNBiasLast -d fruit_salad/ -s CNNBiasLast/${FOLDER_NAME} -o "Adam( model.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)"
