#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


def from_page(url, links, n):
    html = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    ).text

    soup = BeautifulSoup(html, "html.parser")


    for a in soup.find_all("a", href=True):
        href = a["href"]

        # print(href)

        l = "https://www.funda.nl" + href

        if pref in href and l not in links:
            links.add(l)

        if len(links) >= n:
            break

        if (len(links) % 200 == 0):
            print(f"{len(links)} done")

    return links



url = "https://www.funda.nl/en/zoeken/koop?selected_area=[%22amsterdam%22]&search_result="
pref = "/en/detail/koop/amsterdam/appartement-"



def get_links(n):
    print("Getting links...")

    i = 1
    links = set()
    same = 0

    while len(links) < n:
        ln = len(links)
        links = from_page(url + str(i), links, n)

        if (len(links) == ln):
            same += 1

            if (same == 100):
                break
        else:
            same = 0

        i += 1


    print(f"Done with {len(links)} links")

    return list(links)



if "__main__" in __name__:
    n = 10000

    links = get_links(n)


    with open("data/links_all_12.txt", "w") as f:
        for link in links:
            f.write(link + "\n")

        f.close()
