import numpy as np
import os


def compare_arrays(arr1, arr2, name1='first', name2='second', verbose=False):
    mutuals = np.intersect1d(arr1, arr2, assume_unique=True)    
    in_one_not_two = np.setdiff1d(arr1, arr2)
    in_two_not_one = np.setdiff1d(arr2, arr1)
    if verbose:
        print('  # Mutuals: {}'.format(len(mutuals)))
        print('  # Only in {}: {}'.format(name1, len(in_one_not_two)))
        print('  # Only in {}: {}'.format(name2, len(in_two_not_one)))
    return mutuals, in_one_not_two, in_two_not_one



def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))