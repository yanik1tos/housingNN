#!/usr/bin/env python3

import csv
import requests
from bs4 import BeautifulSoup
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from distance import distanceAddress


INPUT = "data/links_all.txt"
OUTPUT = "data/amsterdam_houses.csv"



energy_labels_dict = {
    "A++++": 10,
    "A+++": 30,
    "A++": 50,
    "A+": 70,
    "A": 90,
    "B": 120,
    "C": 160,
    "D": 210,
    "E": 275,
    "F": 350,
    "G": 450,
}



def get_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def parse_house(url):
    try:
        html = get_html(url)
        soup = BeautifulSoup(html, "html.parser")

        # print(soup)
        # print(url)

        spans = soup.find_all("span")

        price = None
        address = None
        area = None
        energy = None
        rooms = None
        bedrooms = None
        bathrooms = 1
        year = None

        prev = None

        for i in range(1, len(spans)):
            span = spans[i]

            text = span.text

            # print(text)

            if (price == None) and ("€" in text) and ("month" not in text) and (20 < i and i < 35):
                price = int(text.split()[1].replace(',', '').strip())
            elif (address == None) and (len(text) == 17) and (len(text.split()) == 3) and \
                    (text.split()[0].isdigit()) and (20 < i and i < 35):
                address = text
            elif (area == None) and ("m²" in text):
                if (len(text.split()) == 2):
                    area = int(text.split()[0].strip())
                elif (len(text.split()) == 3):
                    area = int(text.split()[1].strip())
                else:
                    print("-----------------", text)

            elif (energy == None) and ("energy label" in text):
                energy = energy_labels_dict[prev]
            elif ("room" in text) and (len(text) < 30):
                if (rooms == None) and (" room" in text):
                    index = text.find(" room")
                    rooms = int(text[index - 1])

                if (bedrooms == None) and (" bedroom" in text):
                    index = text.find(" bedroom")
                    bedrooms = int(text[index - 1])

                if (" bathroom" in text):
                    index = text.find(" bathroom")
                    bathrooms = int(text[index - 1])

            elif (year == None) and (text.isdigit() and text[0] in ['1', '2']) and (len(text) == 4):
                year = int(text)
            elif (year == None) and ("before" in text.lower() or "after" in text.lower()) and (text.split()[1].isdigit()):
                year = int(text.split()[1])

            elif (year == None) and (len(text) == 9) and (text[4] == '-') and \
                (text.split('-')[0].isdigit() and text.split('-')[1].isdigit()):

                year = (int(text[:4]) + int(text[5:])) // 2

            prev = text

        # distance = 1


        if (rooms is not None) and (bedrooms is None):
            bedrooms = rooms - 1


        l = [price, address, area, energy, rooms, bedrooms, bathrooms, year]


        # print("Living area:", area)
        # print("Price:", price)
        # print("Energy Label:", energy)
        # print("Rooms:", rooms)
        # print("Bedrooms:", bedrooms)
        # print("Bathrooms:", bathrooms)
        # print("Construction year:", year)

        # print(l)

        if (None in l):
            if (energy is not None):
                print("### Couldn't find info about his website", url)
                print(l)
                print()

            return None
    except Exception as e:
        print(e)

        return None

    return {
        "url": url,
        "price": price,
        "address": address,
        "area": area,
        "energy": energy,
        "rooms": rooms,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year": year,
    }


urls = []

with open(INPUT, 'r') as f:
    for line in f.readlines():
        urls.append(line.strip())

    f.close()


print("Parsing...")

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
        ]
    )

    writer.writeheader()


    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [
            executor.submit(parse_house, url)
            for url in urls
        ]

        delay = 0

        for future in as_completed(futures):
            house = future.result()

            if house is not None:
                # house["distance"] = distanceAddress(house["address"])

                writer.writerow(house)
                f.flush()

                # print("Saved:", house["url"])

                # delay += 1

                # if (delay >= 100):
                #     time.sleep(1)
                #     delay = 0

    # for url in urls:
    #     try:
    #         house = parse_house(url)
    #         writer.writerow(house)
    #         # print("Saved:", url)

    #     except Exception as e:
    #         print("Failed:", url, e)
    #         traceback.print_exc()

    #     # print()

    f.close()


print("Done")
