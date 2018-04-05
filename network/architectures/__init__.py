#import architectures
import os
import re

path = os.path.dirname(os.path.realpath(__file__))
print(path)
files = [m.group(0) for m in [re.search('.*(?=\.py)',i) for i in os.listdir(path)] if m is not None]

__all__= files
