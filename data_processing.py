# Data shape: 22283 * 5896

import numpy as np

def read_data(filepath="Gene_Chip_Data/microarray.original.txt"):
    data = np.zeros((5896,22283))
    cnt = 0
    with open(filepath, 'r') as f:
        f.readline()
        line = f.readline()
        while line != '':
            line = line.strip()
            # print line
            row = line.split('\t')[1:]
            # print len(row)
            for i in range(5896):
                data[i][cnt] = float(row[i])
            line = f.readline()
            cnt += 1
            print("Successfully read in data {}/22283 !".format(cnt))
    return data

if __name__ == "__main__":
    # with open("Gene_Chip_Data/microarray.original.txt", 'r') as f:
    #     len = len(f.readlines())
    #     print len
    data = read_data()
    np.save("Gene_Chip_Data/data.npy", data)
