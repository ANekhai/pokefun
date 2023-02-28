import pandas as pd
import json
import os
from pathlib import Path

from api_interactor import get_image


def pokemon_json_to_rows(poke_info):

    row_list = []

    for poke in poke_info:
        # create first pass dictionaries using the easiest extractable data
        row_dict = {
            "id": poke["id"],
            "name": poke["species"]["name"],
            "height": poke["height"],
            "weight": poke["weight"]
        }

        # Abilities
        for ability in poke["abilities"]:
            row_dict[f"ability_{ability['slot']}"] = ability["ability"]["name"]

        # Species Flattening
        species_data = poke["species"]["data"]
        flat_entries = ["forms_switchable", "gender_rate", "genera",
                        "has_gender_differences", "is_baby", "is_legendary", "is_mythical"]
        named_entries = ["color", "evolves_from_species", "generation", "habitat", "shape"]

        for key in flat_entries: row_dict[key] = species_data[key]
        for key in named_entries: row_dict[key] = species_data[key]["name"] if species_data[key] else None
        
        # Egg Group Enumeration
        for i, egg_group in enumerate(species_data["egg_groups"]): row_dict[f"egg_group_{i}"] = egg_group["name"]

        # Flavor Text Enumeration
        for i, text in enumerate(species_data["flavor_text_entries"]): row_dict[f"flavor_text_{i}"] = text

        # # Sprites
        # row_dict["front_sprite"] = poke["sprites"]["front_default"]
        # row_dict["female_sprite"] = poke["sprites"]["front_female"]
        # row_dict["shiny_sprite"] = poke["sprites"]["front_shiny"]
        # row_dict["shiny_female_sprite"] = poke["sprites"]["front_shiny_female"]

        # row_dict["artwork"] = poke["sprites"]["other"]["official-artwork"]["front_default"]
        # row_dict["artwork_shiny"] = poke["sprites"]["other"]["official-artwork"]["front_shiny"]

        # Stats
        for stat in poke["stats"]:
            row_dict[stat["stat"]["name"]] = stat["base_stat"]

        # Types
        for p_type in poke["types"]:
            row_dict[f"type_{p_type['slot']}"] = p_type["type"]["name"]

        row_list.append(row_dict)

    return row_list


def info_to_csv(poke_info, path: str):
    df = pd.DataFrame.from_dict(poke_info)
    df.to_csv(path)
    return df


        
    return df

if __name__ == "__main__":

    json_path = "pokemon_info.json"
    csv_path = "pokemon.csv"
    data_path = Path("data")

    with open(json_path) as f:
        poke_info = json.load(f)

    poke_row_list = pokemon_json_to_rows(poke_info)
    
    print(poke_row_list[1])

    # poke_df = info_to_csv(poke_row_list, csv_path)


