#!/usr/bin/env python3
"""Query SWAPI for planets that are homeworlds of sentient species."""

import requests


def sentientPlanets():
    """Retrieve all planets that serve as homeworlds for sentient species.

    Returns:
        list: Names of all planets that are the homeworld of at least one
        sentient species.
    """
    url = "https://swapi.dev/api/species/"
    homeworld = []
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for species in data['results']:
            if (species['classification'] == 'sentient' or
                    species['designation'] == 'sentient'):
                homeworld.append(species['homeworld'])
        url = data['next']

    url = "https://swapi.dev/api/planets/"
    planets = {}
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for planet in data['results']:
            planets[planet['url']] = planet['name']
        url = data['next']
    planet_list = []

    for planet in planets:
        if planet in homeworld:
            planet_list.append(planets[planet])

    return planet_list
