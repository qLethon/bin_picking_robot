import os

def count_files(dir_path):
    return len([f.path for f in os.scandir(dir_path) if f.is_file()])

def probability_to_green_image_array(P):
    import numpy as np
    P *= 255
    green = np.zeros((P.shape[0], P.shape[1], 3), dtype=np.uint8)
    for h in range(P.shape[0]):
        for w in range(P.shape[1]):
            green[h][w][1] = P[h][w]

    return green