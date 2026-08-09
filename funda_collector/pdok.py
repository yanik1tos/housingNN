#!/usr/bin/env python3

from math import radians, sin, cos, sqrt, atan2, asin
import requests
import re

import pandas as pd
import csv

import traceback


INPUT  = "data/amsterdam_houses_all.csv"
OUTPUT = "data/amsterdam_houses_all_distance.csv"


def getLonLat(address):
    url = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"

    params = {
        "q": address
    }

    r = requests.get(url, params=params)



    a = re.findall(r"POINT\(([-\d.]+) ([-\d.]+)\)", r.text)

    num = 0
    longitude = 0
    latitude = 0

    for i in a:
        lon, lat = map(float, i)
        if (4 < lon < 7 and 52 < lat < 54):
            longitude += lon
            latitude += lat

            num += 1


    longitude /= num
    latitude /= num


    return longitude, latitude


def distanceCenter(lat1, lon1):
    lat2 = 52.3676
    lon2 = 4.9041

    R = 6371

    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2

    return 2 * R * asin(sqrt(a))


def distanceAddress(address):
    lon, lat = getLonLat(address)

    return distanceCenter(lat, lon)



def addDistanceCSV():
    data = pd.read_csv(INPUT)


    remember = {}


    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url",
                "price",
                "address",
                "area",
                "energy",
                "rooms",
                "bedrooms",
                "bathrooms",
                "year",
                "distance",
            ]
        )

        writer.writeheader()

        for index, row in data.iterrows():
            try:
                address = row["address"]
                if (address in remember):
                    dist = remember[address]
                else:
                    dist = distanceAddress(row["address"])
                    remember[address] = dist


                if (dist > 10):
                    print("Distance too much:", dist)
                    continue


                d = row.to_dict()
                d["distance"] = dist

                writer.writerow(d)
                f.flush()

            except Exception as e:
                print(e)
                print(traceback.print_exc())


        f.close()



if "__main__" in __name__:
    print(getLonLat("1073 SB Amsterdam"))
    # addDistanceCSV()
    print(distanceAddress("1073 SB Amsterdam"))