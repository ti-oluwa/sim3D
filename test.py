import time
import requests
import json
import os
import pandas as pd


def download_data(directory: str) -> None:
    token = "wZvwkVnOdCVttoVAnVSUnCVqypQUePCU"
    headers = {"token": token}
    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

    ranges = [
        ("2010-01-01", "2010-12-31"),
        ("2011-01-01", "2011-12-31"),
        ("2012-01-01", "2012-12-31"),
        ("2013-01-01", "2013-12-31"),
        ("2014-01-01", "2014-12-31"),
        ("2015-01-01", "2015-12-31"),
        ("2016-01-01", "2016-12-31"),
        ("2017-01-01", "2017-12-31"),
        ("2018-01-01", "2018-12-31"),
        ("2019-01-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
    ]
    for start_date, end_date in ranges:
        params = {
            "datasetid": "GHCND",
            "datatypeid": "PRCP",
            "stationid": "GHCND:USW00094728",
            "startdate": start_date,
            "enddate": end_date,
            "units": "metric",
            "limit": 1000,
        }

        response = requests.get(url, headers=headers, params=params)

        save_path = os.path.join(directory, f"{start_date}_{end_date}.json")
        # Ensure the directory exists
        if response.status_code == 200:
            with open(save_path, "w") as file:
                data = response.json()["results"]
                json.dump(data, file, indent=4)
            
            print(f"Downloaded data from {start_date} to {end_date}")
            print(f"Data saved to {save_path}")
            time.sleep(1)  # To avoid hitting the API rate limit
        else:
            print(f"Error: {response.status_code} - {response.text}")


def convert_json_to_csv(json_file: str, csv_file: str):
    with open(json_file, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Converted {json_file} to {csv_file}")


def convert_all_json_in_directory_to_csv(directory: str):
    import os

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_file = os.path.join(directory, filename)
            csv_file = os.path.join(directory, "csv", filename.replace(".json", ".csv"))
            convert_json_to_csv(json_file, csv_file)


def main():
    directory = "data"
    os.makedirs(os.path.join(directory, "csv"), exist_ok=True)
    download_data(directory)
    convert_all_json_in_directory_to_csv(directory)


if __name__ == "__main__":
    main()
