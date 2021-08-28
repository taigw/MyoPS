from batchgenerators.utilities.file_and_folder_operations import load_json, load_pickle
import numpy as np
import os
import cv2
from scipy import ndimage
# x = 1
# y = x
# print(y is x, y == x)  # true, true
# y = 2
# print(y is x, y == x)  # false, false
# x = x + 1
# print(y is x, y == x)  # true, true

# a = [1, 2, 3]
# b = a
# print(b is a, b == a)  # true, true
# b = [1, 2, 3]
# print(b is a, b == a)  # false, true
# c = a
# c.append(4)
# print(c is a, c == a)  # true, true

# a = (1, 2, 3)
# b = a
# print(b is a, b == a)  # true, true
# b = (1, 2, 3)
# print(b is a, b == a)  # true, true

from multiprocessing import Process


def print_func(continent):
    print('The name of continent is : ', continent)

if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    proc = Process(target=print_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = Process(target=print_func, args=(name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

