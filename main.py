from seq2seq import Seq2seq
from random import randint
import csv
import time
import datetime
import numpy as np
import copy

def get_real_arr(arr):
    """
    返回删除所有空值后的arr
    """
    arr_copy = copy.deepcopy(arr)
    arr_copy = list(filter(None, arr_copy))
    while '' in arr_copy:
        arr_copy.remove('')
    return arr_copy


def main():
    seq2seq = Seq2seq()

    last_seq = None
    cost = 0


    route_path = 'C:/Users/10503/Desktop/Cloud/PRP/data_R/Osaka/fake_path.csv'
    user_path = 'C:/Users/10503/Desktop/Cloud/PRP/data_R/Osaka/fake_user.csv'

    visPath = np.loadtxt(open(route_path,"r"),dtype=np.str, delimiter=",",skiprows=0)
    # list = list.astype(np.int)
    user_iv = np.loadtxt(open(user_path,"r"),dtype=np.str, delimiter=",",skiprows=0)
    # user_iv = user_iv.astype(np.float)

    with open(route_path, 'r') as f:
        reader = f.readlines()
        lens = len(reader)

    for k in range(1000):
        for i in range(lens):
            # print(i)
            # print(list[i])
            X = user_iv[i]
            X = get_real_arr(X)
            X = list(map(eval, X))
            for j in range(len(X)):
                X[j] = j*100 + X[j]*10
            X = list(map(int, X))

            Y = visPath[i]
            Y = get_real_arr(Y)
            Y = list(map(eval, Y))


            # print('train: ', X, '->', Y)
            cost += seq2seq.train(X, Y, Y[0], Y[len(Y)-1])

            if i % 100 == 0:
                print(k*3000 + i, '\t', cost / 1000)
                cost = 0

                X = user_iv[i]
                X = get_real_arr(X)
                X = list(map(eval, X))
                for j in range(len(X)):
                    X[j] = j*100 + X[j]*10
                X = list(map(int, X))

                Y = seq2seq.predict(X, Y[0], Y[len(Y)-1])

                print(X, '->', Y)

                Z = visPath[i]
                Z = get_real_arr(Z)
                Z = list(map(eval, Z))
                print('Expect:', X, '->', Z)

                seq2seq.lr /= 2


if __name__ == "__main__":
    main()
