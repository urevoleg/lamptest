import re
import time
import json
import secrets
from collections import defaultdict

from queue import Queue
from threading import Thread

import  requests

import pandas as pd
import numpy as np

import os

from bs4 import BeautifulSoup

from tqdm.auto import tqdm

from selenium.webdriver import Chrome
from selenium import webdriver

from pathlib import Path
DIR_HOME = str(Path.home())

import warnings
warnings.filterwarnings('ignore')


def download_and_save_img(url, name):
    response = requests.get(url)
    with open(f"src/{name}.png", 'wb') as f:
        f.write(response.content)


def get_attrs(elem, feature, hash_id, queue):
    if feature == 'desc':
        if elem.find("a").get("href", None):
            img_url = "https://lamptest.ru" + elem.find("a").get("href")
            queue.put((img_url, hash_id))
    return elem.text.strip()


def create_driver():
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(os.path.join(DIR_HOME, "projects", "chromedriver"), chrome_options=chrome_options)
    return driver


params_list = ['brand', 'model', 'desc', 'base', 'shape', 'price',
 'p', 'lm_prc', 'lm', 'eff', 'eq', 'color', 'cri', 'angle', 'flicker', 'rating', 'war']


class DownloadWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            url, name = self.queue.get()
            try:
                download_and_save_img(url, name)
            finally:
                self.queue.task_done()


def parse():

    queue = Queue()
    # Create 8 worker threads
    for x in range(8):
        worker = DownloadWorker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()

    driver = create_driver()

    output = defaultdict(list)

    list_of_page_dataframe = []
    for page in tqdm(range(0, 5000, 50), desc='Page loop'):
        hash_id = secrets.token_hex(nbytes=16)
        # print(f'page: {page}')
        temp_df = pd.DataFrame(columns=params_list)
        url = f'http://lamptest.ru/search/#currency=rub&type=LED&start={page}'
        driver.get(url)
        time.sleep(1)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html')
        columns = defaultdict(list)

        for cl in params_list:
            res = soup.find_all("td", class_=cl)

            for idx, p in enumerate(res):
                d = get_attrs(p, feature=cl, hash_id='-'.join([hash_id, str(idx)]), queue=queue)
                columns[cl] += [d]
            output[cl] += columns[cl]

        # hash
        res = soup.find_all("td", class_="model")
        for idx, p in enumerate(res):
            columns["hash_id"] += ['-'.join([hash_id, str(idx)])]
        output["hash_id"] += columns["hash_id"]

    queue.join()

    with open("src/lamtest-parse-data.json", 'w') as f:
        f.write(json.dumps(output))


def preprocessing():
    bulbs = pd.read_json('src/lamtest-parse-data.json')

    # cast types
    for col in ["price", "p", "lm_prc", "lm", "eff", "eq", "color", "cri", "angle", "flicker", "rating", "war"]:
        bulbs.loc[:, col] = pd.to_numeric(bulbs.loc[:, col], errors='coerce')

    from functools import partial

    def extract_param_from_desc(x, param='K'):
        str_pattern = f"(\d+){param}"
        pattern = re.compile(str_pattern)
        if re.findall(pattern, x):
            return re.findall(pattern, x)[0]
        return np.NaN

    filter_color_temp_pattern = bulbs['desc'].str.match(r'\d{4}K')
    bulbs.loc[filter_color_temp_pattern, "color_from_desc"] = bulbs.loc[filter_color_temp_pattern, "desc"].map(
        partial(extract_param_from_desc, param='K')).astype(float)

    filter_lm_pattern = bulbs['desc'].str.match(r'.*\d+lm')
    bulbs.loc[filter_lm_pattern, "lm_from_desc"] = bulbs.loc[filter_lm_pattern, "desc"].map(
        partial(extract_param_from_desc, param='lm')).astype(float)

    filter_w_pattern = bulbs['desc'].str.match(r'.*\d+W')
    bulbs.loc[filter_lm_pattern, "w_from_desc"] = bulbs.loc[filter_lm_pattern, "desc"].map(
        partial(extract_param_from_desc, param='W')).astype(float)

    bulbs.to_csv("src/lamptest-bulbs.csv", index=None)



if __name__ == '__main__':
    #parse()
    preprocessing()
