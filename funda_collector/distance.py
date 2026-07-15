#!/usr/bin/env python3

from math import radians, sin, cos, sqrt, atan2, asin
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


import pandas as pd
import csv

import time
import traceback


INPUT  = "data/amsterdam_houses_all.csv"
OUTPUT = "data/amsterdam_houses_all_distance.csv"


geolocator = Nominatim(user_agent="housing_neural_network", timeout=10)
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.1,
    max_retries=3,
    error_wait_seconds=5
)


def distanceLats(lat1, lon1):
    lat2 = 52.3676
    lon2 = 4.9041

    R = 6371

    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)

    # a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2

    # return 2*R*atan2(sqrt(a), sqrt(1-a))

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2

    return 2 * R * asin(sqrt(a))



def distanceAddress(address):
    location = geocode(address)

    lat = location.latitude
    lon = location.longitude

    # print(lat, lon)

    return distanceLats(lat, lon)




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
    # print(distanceAddress("Damrak 87, 1012 LP Amsterdam"))
    addDistanceCSV()