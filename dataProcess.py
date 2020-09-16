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
    route_path = 'C:/Users/10503/Desktop/Cloud/PRP/data_R/Osaka/fake_path.csv'
    user_path = 'C:/Users/10503/Desktop/Cloud/PRP/data_M/userVisits/userVisits-Osak.csv'
    user_res = 'C:/Users/10503/Desktop/Cloud/PRP/data_M/userVisits/userIV-Osak.csv'
    poi_theme = 'C:/Users/10503/Desktop/Cloud/PRP/data_M/poiList/POI-Osak.csv'

    f = open(user_path, 'r')
    g = open(user_res, 'w')
    pp = open(poi_theme, 'r')

    # reader = csv.reader(f)
    writer = csv.writer(g)
    # print('start')
    reader = [each for each in csv.DictReader(f, delimiter=';')]
    # print(type(reader))
    # total = len(reader)
    # print(total)
    # poi_dur = numpy.zeros(shape=(1,total_poi))
    user_name = []
    poi_total_photo = np.zeros(shape=(1,29))
    for row in reader:
        # print(row["userID"])
        if row['userID'] not in user_name:
            user_name.append(row['userID'])
            # print(row['userID'])

    # print(len(user_name))
    photos = np.zeros(shape=(len(user_name),30))

    for row in reader:
        if row['userID'] not in user_name:
            user_name.append(row['userID'])
            # print(row['userID'])

    for row in reader:
        i = user_name.index(row['userID'])
        poiID = int(row['poiID']) - 1
        photos[i][poiID] += 1
        poi_total_photo[0][poiID] += 1

    poi_vis = np.zeros(shape=(1,29))
    for k in range(0,29):
        for j in range(0, len(user_name)):
            if photos[j][k] != 0:
                poi_vis[0][k] += 1
        if poi_vis[0][k] != 0:
            poi_total_photo[0][k] /= poi_vis[0][k]

    for j in range(0, len(user_name)):
        for k in range(0,29):
            if poi_total_photo[0][k] != 0:
                photos[j][k] /= poi_total_photo[0][k]

    poi_reader = [each for each in csv.DictReader(pp, delimiter=';')]

    line = 0
    poi = np.zeros(shape=(1,29))
    theme = []
    for row in poi_reader:
        if row['theme'] not in theme:
            theme.append(row['theme'])
    for row in poi_reader:
        i = int(row['poiID']) - 1
        poi[0][i] = theme.index(row['theme'])

    userIV = np.zeros(shape=(len(user_name),len(theme)))
    userIV_num = np.zeros(shape=(len(user_name),len(theme)))

    for j in range(0, len(user_name)):
        for k in range(0,29):
            if photos[j][k] != 0:
                t = int(poi[0][k])
                userIV[j][t] += photos[j][k]
                userIV_num[j][t] += 1
        for t in range(0,len(theme)):
            if userIV_num[j][t] != 0:
                userIV[j][t] /= userIV_num[j][t]

    print(userIV)

    # print(poi)
    # print(photos)
    writer.writerow(['userID',theme])
    for j in range(0, len(user_name)):
        writer.writerow([user_name[j],userIV[j]])

    f.close()
    g.close()
    pp.close()

if __name__ == "__main__":
    main()
