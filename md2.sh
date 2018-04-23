#!/bin/bash

cd ~/Repos/exjobb/preprocessing/

python3 make_dataset.py --save-path apple/ --data-path apple/
python3 make_dataset.py --save-path orange/ --data-path orange/
python3 make_dataset.py --save-path pear/ --data-path pear/
python3 make_dataset.py --save-path raspberry/ --data-path raspberry/
python3 make_dataset.py --save-path papaya/ --data-path papaya/
python3 make_dataset.py --save-path pineapple/ --data-path pineapple/
python3 make_dataset.py --save-path peach/ --data-path peach/
python3 make_dataset.py --save-path jackfruit/ --data-path jackfruit/
python3 make_dataset.py --save-path lemon/ --data-path lemon/
python3 make_dataset.py --save-path pomegranate/ --data-path pomegranate/
