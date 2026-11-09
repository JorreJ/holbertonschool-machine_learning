#!/usr/bin/env python3
"""Module to query SWAPI for starships based on passenger capacity."""

import requests


def availableShips(passengerCount):
    """Retrieve starships capable of carrying a given number of passengers.

    Args:
        passengerCount (int): Minimum number of passengers the starship
            must be able to carry.

    Returns:
        list: Names of all starships that can accommodate at least
        passengerCount passengers.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ship_list = []
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            passengers = ship['passengers'].replace(",", "")
            if passengers.isdigit():
                if int(passengers) >= passengerCount:
                    ship_list.append(ship['name'])
        url = data['next']
    return ship_list
