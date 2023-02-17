import requests
import json
import os
from unidecode import unidecode

def get_pokemon_details(pokemon_name):
    pokemon_info = None

    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name}"
    response = requests.get(url)
    if response.status_code == 200:
        pokemon_info = response.json()
        
    return pokemon_info


def get_all_pokemon():
    url = "https://pokeapi.co/api/v2/pokemon?limit=1000"
    pokemon_list = []
    while url:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            pokemon_list.extend(data['results'])
            url = data.get("next", None)
        else:
            return None
    return pokemon_list

def load_and_save_pokemon(path: str):
    poke_list = get_all_pokemon()
    
    poke_info = []  
    for pokemon in poke_list:
        print(pokemon["name"])
        poke_data = get_pokemon_details(pokemon["name"])
        poke_info.append(poke_data)
        
    with open(path, 'w') as file:
        json.dump(poke_info, file)

    return poke_info

def filter_pokemon_info(info):

    to_keep = ["abilities", "forms", "height", "id", 
               "species", "sprites", "stats", "types", "weight"]

    return [{key: pokemon[key] for key in to_keep} for pokemon in info]


def retreive_species_data(url):
    to_keep = ["color", "egg_groups", "evolves_from_species",  "forms_switchable", "flavor_text_entries",
               "gender_rate", "genera", "generation", "habitat", "has_gender_differences",
               "is_baby", "is_legendary", "is_mythical", "shape"]

    species_data = None
    
    response = requests.get(url)
    if response.status_code == 200:
        raw_data = response.json()
        species_data = {key: raw_data[key] for key in to_keep}

        # filter out non-english flavor text_entries
        english_genera = [unidecode(entry["genus"]) for entry in species_data["genera"] if entry["language"]["name"] == "en"]
        english_flavor_text = list(set([unidecode(entry["flavor_text"].replace("\n", " ").replace("\u000c", " "))
                               for entry in species_data["flavor_text_entries"] if entry["language"]["name"] == "en"]))

        species_data["genera"] = english_genera.pop() if english_genera else "null"
        species_data["flavor_text_entries"] = english_flavor_text

    return species_data


def update_species_data(info_list):

    for pokemon in info_list:
        print(pokemon["species"]["name"])
        species_url = pokemon["species"]["url"]
        pokemon["species"]["data"] = retreive_species_data(species_url)

    return info_list
    

def get_image(url: str, path: str):

    response = requests.get(url)

    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)


if __name__ == "__main__":

    poke_path = "pokemon_info.json"

    if not os.path.exists(poke_path):
        poke_info = load_and_save_pokemon(poke_path)
    else:
        with open(poke_path) as file:
            poke_info = json.load(file)        

    # poke_info = filter_pokemon_info(poke_info)

    poke_info = update_species_data(poke_info)

    with open(poke_path, 'w') as file:
        json.dump(poke_info, file)
    
    # url = "https://pokeapi.co/api/v2/pokemon-species/138/"
    # data = retreive_species_data(url)

    # with open("test.json", 'w') as f:
    #     json.dump(data, f)