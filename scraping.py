import time
import requests
import pandas as pd
import numpy as np
import random

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


pd.set_option("display.max_columns", 999)

# extract all parcel id from dataset downloaded from bc government data website
pid_df = pd.read_csv('./property-addresses.csv', sep=';')
pid_lst = pid_df[['SITE_ID']]

# for building with invalid pid, use address to search in the website

pid_df = pid_df.dropna(subset=['CIVIC_NUMBER'])
pid_df['road'] = pid_df['STD_STREET'].apply(lambda x: x.replace('W ', '') + ' WEST' if 'W ' in x else x)
pid_df['road'] = pid_df['road'].apply(lambda x: x.replace('E ', '') + ' EAST' if 'E ' in x else x)
pid_df = pid_df.dropna(subset=['road'])
pid_df['lon'] = pid_df['Geom'].apply(lambda x: x[x.find('[') + 1 : x.find(']')].split(',')[0])
pid_df['lat'] = pid_df['Geom'].apply(lambda x: x[x.find('[') + 1 : x.find(']')].split(',')[1])

pid_df['CIVIC_NUMBER'] = pid_df['CIVIC_NUMBER'].astype('int').astype('str')
pid_df['full_add'] = pid_df['CIVIC_NUMBER'].str.cat(pid_df['road'], sep=' ')

pid_df['full_add'] = pid_df['full_add'] + ' vancouver'

pid_df.to_csv('./buildings.csv', index_label=0)

# scrap data
pid_df = pd.read_csv('./buildings.csv', index_col=0)
pid_df = pid_df.dropna(subset=['SITE_ID'])

driver = webdriver.Chrome('./chromedriver.exe')

i = 0
for index, row in pid_df.iterrows():
    # to not get blocked, quit webdriver after search for 10 results
    if index % 10 == 0:
        driver.quit()
        driver = webdriver.Chrome('./chromedriver.exe')
    pid = row['SITE_ID']
    full_address = row['full_add']
    driver.get("https://www.bcassessment.ca/")
    if pid.isdigit():
        select = Select(driver.find_element_by_id('ddlSearchType'))
        select.select_by_visible_text('PID')
        driver.find_element_by_id('txtPID').send_keys(pid)
        driver.find_element_by_id('btnSearch').send_keys(Keys.ENTER)
    else:
        select = Select(driver.find_element_by_id('ddlSearchType'))
        select.select_by_visible_text('Civic address')

        driver.find_element_by_id('rsbSearch').send_keys(full_address)
        element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".ui-menu-item-wrapper"))
        )
        element.click()

    time.sleep(random.uniform(2, 8))

    try:
        pid_df.loc[index, 'actualy_address'] = soup.find(id='mainaddresstitle').contents[0]
    except:
        pid_df.loc[index, 'actualy_address'] = np.nan
    try:
        pid_df.loc[index, 'total_value'] = soup.find(id='lblTotalAssessedValue').contents[0]
    except:
        pid_df.loc[index, 'total_value'] = np.nan
    try:
        pid_df.loc[index, 'prev_value'] = soup.find(id='lblPreviousAssessedValue').contents[0]
    except:
        pid_df.loc[index, 'prev_value'] = np.nan
    try:
        pid_df.loc[index, 'built_year'] = soup.find(id='lblYearBuilt').contents[0]
    except:
        pid_df.loc[index, 'built_year'] = np.nan
    try:
        pid_df.loc[index, 'bedroom'] = soup.find(id='lblBedrooms').contents[0]
    except:
        pid_df.loc[index, 'bedroom'] = np.nan
    try:
        pid_df.loc[index, 'bathroom'] = soup.find(id='lblBathRooms').contents[0]
    except:
        pid_df.loc[index, 'bathroom'] = np.nan
    try:
        pid_df.loc[index, 'garage'] = soup.find(id='lblGarages').contents[0]
    except:
        pid_df.loc[index, 'garage'] = np.nan
    try:
        pid_df.loc[index, 'carport'] = soup.find(id='lblCarPorts').contents[0]
    except:
        pid_df.loc[index, 'carport'] = np.nan
    try:
        pid_df.loc[index, 'area'] = soup.find(id='lblLandSize').contents[0]
    except:
        pid_df.loc[index, 'area'] = np.nan
    print(pid_df.loc[index])

pid_df.to_csv('./0-15000.csv', index=False)


