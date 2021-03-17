import requests


def fetch_data(data_source, output_file):
    response = requests.get(data_source)
    if response.status_code == 200:
        file = open(output_file, "w+", encoding="utf-8")
        file.write(response.text)
        file.close()
        return True
    return False


def clean_lines(data_source, output_data):
    with open(data_source, "r", encoding="utf-8") as f:
        data = f.readlines()
    with open(output_data, "w+", encoding="utf-8") as outfile:
        for line in data:
            if len(line.strip()) > 0:
                outfile.write(f"{line.strip()}\n")


TXT_DATA_SOURCE = "https://www.gutenberg.org/files/16/16-0.txt"
HTML_DATA_SOURCE = "https://www.gutenberg.org/files/16/16-h/16-h.htm"
DATA_PATH = "data/peter_pan.txt"
CLEAN_DATA_PATH = "data/peter_pan_clean.txt"

fetch_data(HTML_DATA_SOURCE, DATA_PATH)
clean_lines(DATA_PATH, CLEAN_DATA_PATH)
