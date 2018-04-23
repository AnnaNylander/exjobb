#!/bin/bash

cd ~/Repos/exjobb/preprocessing/

python3 make_dataset.py --save-path mango/ --data-path mango/
python3 make_dataset.py --save-path banana/ --data-path banana/
python3 make_dataset.py --save-path watermelon/ --data-path watermelon/
python3 make_dataset.py --save-path kiwi/ --data-path kiwi/
python3 make_dataset.py --save-path strawberry/ --data-path strawberry/
python3 make_dataset.py --save-path carambola/ --data-path carambola/
python3 make_dataset.py --save-path litchi/ --data-path litchi/
python3 make_dataset.py --save-path rambutan/ --data-path rambutan/
python3 make_dataset.py --save-path pitahaya/ --data-path pitahaya/
python3 make_dataset.py --save-path durian/ --data-path durian/
