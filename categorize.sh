#!/bin/bash
cd ~/Repos/exjobb/preprocessing/

NAME=watermelon
python3 categorize_dataset.py -a ${NAME}/${NAME}.csv -s fruits_categorized/${NAME}/ -d fruits/${NAME}/
