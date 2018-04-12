#!/bin/bash

cd ~/Repos/exjobb/preprocessing/
TRAIN=0.7
TEST=0.2

python3 split_dataset.py --train $TRAIN --test $TEST -s mango_split/ -d mango/
python3 split_dataset.py --train $TRAIN --test $TEST -s banana_split/ -d banana/
python3 split_dataset.py --train $TRAIN --test $TEST -s watermelon_split/ -d watermelon/
python3 split_dataset.py --train $TRAIN --test $TEST -s kiwi_split/ -d kiwi/
python3 split_dataset.py --train $TRAIN --test $TEST -s strawberry_split/ -d strawberry/
python3 split_dataset.py --train $TRAIN --test $TEST -s carambola_split/ -d carambola/
python3 split_dataset.py --train $TRAIN --test $TEST -s litchi_split/ -d litchi/
python3 split_dataset.py --train $TRAIN --test $TEST -s rambutan_split/ -d rambutan/
python3 split_dataset.py --train $TRAIN --test $TEST -s pitahaya_split/ -d pitahaya/
python3 split_dataset.py --train $TRAIN --test $TEST -s durian_split/ -d durian/
python3 split_dataset.py --train $TRAIN --test $TEST -s apple_split/ -d apple/
python3 split_dataset.py --train $TRAIN --test $TEST -s orange_split/ -d orange/
python3 split_dataset.py --train $TRAIN --test $TEST -s pear_split/ -d pear/
python3 split_dataset.py --train $TRAIN --test $TEST -s raspberry_split/ -d raspberry/
python3 split_dataset.py --train $TRAIN --test $TEST -s papaya_split/ -d papaya/
python3 split_dataset.py --train $TRAIN --test $TEST -s pineapple_split/ -d pineapple/
python3 split_dataset.py --train $TRAIN --test $TEST -s peach_split/ -d peach/
python3 split_dataset.py --train $TRAIN --test $TEST -s jackfruit_split/ -d jackfruit/
python3 split_dataset.py --train $TRAIN --test $TEST -s lemon_split/ -d lemon/
python3 split_dataset.py --train $TRAIN --test $TEST -s pomegranate_split/ -d pomegranate/


echo Completed!

#python3 main.py -a cnn_bias.CNNBiasFirst -b 2 -e 100 -d mango/ -s CNNBiasFirst/
