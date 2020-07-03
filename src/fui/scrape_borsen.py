from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException
from lxml import html, etree
import time
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.fui.utils import params
import re
from googleapiclient.discovery import build

def google_query(query, api_key, cse_id, **kwargs):
    query_service = build("customsearch",
                          "v1",
                          developerKey=api_key
                          )
    query_results = query_service.cse().list(q=query,    # Query
                                             cx=cse_id,  # CSE ID
                                             **kwargs
                                             ).execute()
    return query_results['items']


def infomedia_login(driver):
    with open(params().paths['credentials'] + 'infomedia.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        head = next(csv_reader)
        cred = next(csv_reader)
    infomedia = dict(zip(head, cred))
    username = driver.find_element_by_id("UserName")
    password = driver.find_element_by_id("Password")
    username.send_keys(infomedia['username'])
    password.send_keys(infomedia['password'])
    driver.find_element_by_id("loginBtn").click()

def get_infomedia_headlines(fromdate, todate, driver=None):
    p = Path(params().paths['scraped']+params().filenames['headlines']+'_'+fromdate.strftime("%d-%m-%y")+'_'+todate.strftime("%d-%m-%y"))
    if p.exists():
        df = pd.read_csv(str(p), encoding="latin-1", sep=";", header=None)
        headlines = df[0].to_list()
        for i, l in enumerate(headlines):
            headlines[i] = l.replace('\xad', '')
        return headlines
    else:
        url = 'https://apps.infomedia.dk/mediearkiv/'
        if driver is None:
            driver = init_chromedriver()
        driver.get(url)
        infomedia_login(driver)

        WebDriverWait(driver, 20).until(
            ec.visibility_of_element_located((By.XPATH, '//*[@id="__filterSpecDropDown_DateRange"]')))
        driver.find_element_by_xpath('//*[@class="ace_text-input"]').send_keys('e*')
        driver.find_element_by_xpath('//*[@id="__filterSpecDropDown_DateRange"]').click()
        time.sleep(3)
        driver.find_element_by_xpath('//input[@data-bind="value: FormattedFromDate"]').click()
        driver.find_element_by_xpath('//input[@data-bind="value: FormattedFromDate"]').send_keys(Keys.CONTROL + "a")
        driver.find_element_by_xpath('//input[@data-bind="value: FormattedFromDate"]').send_keys(
            f'{fromdate.month}/{fromdate.day}/{fromdate.year}')
        time.sleep(3)
        driver.find_element_by_xpath('//input[@data-bind="value: FormattedToDate"]').click()
        driver.find_element_by_xpath('//input[@data-bind="value: FormattedToDate"]').send_keys(Keys.CONTROL + "a")
        driver.find_element_by_xpath('//input[@data-bind="value: FormattedToDate"]').send_keys(
            f'{todate.month}/{todate.day}/{todate.year}')
        time.sleep(3)
        driver.find_element_by_id("iqlsearchbtn").click()
        time.sleep(20)

        # show 100 results per page
        driver.find_elements_by_xpath('//*[@class="dropdown-toggle no-layout"]')[1].click()
        time.sleep(3)
        dropdown = driver.find_element_by_xpath('//*[@class="sub-dropdown-toggle"]')
        driver.execute_script("arguments[0].click();", dropdown)
        time.sleep(3)
        driver.find_element_by_xpath('//*[@class="sub-dropdown-menu"]/li[5]').click()
        time.sleep(20)

        headers = []
        metas = []
        while True:
            articles = driver.find_elements_by_class_name('article-item')
            for a in articles:
                header = a.find_element_by_class_name('header').text.replace('Extract', '')
                meta = a.find_element_by_class_name('meta-data').text
                if meta.find("Page 1") == -1:
                    headers.append(header)
                    metas.append(meta)
                    print(header)
            time.sleep(5)
            try:
                driver.find_element_by_class_name("ifm-ms-next-article").click()
                time.sleep(20)
            except:
                print("Stopping...")
                break

        for l in headers:
            l = l.replace(u'\xad', '')
            l = l.replace(u'\u00ad', '')
            l = l.replace(u'\N{SOFT HYPHEN}', '')
            print(l)
        formstre = datetime.now().strftime("%d-%m-%y")
        with open(params().paths['scraped']+params().filenames['headlines']+'_'+fromdate.strftime("%d-%m-%y")+'_'+todate.strftime("%d-%m-%y"),
                  'w', newline='') as headlines:
            wr = csv.writer(headlines, delimiter=';')
            wr.writerows(zip(headers, metas))

        return headers

def scrape_borsen(driver, url):
    driver.get(url)
    tree = html.fromstring(driver.page_source)
    headline = tree.xpath('//*[@class="headline hyphenate"]//text()')
    body = tree.xpath('//*[@itemprop="articleBody"]//text()')
    date = tree.xpath('//*[@class="timestamp"][1]//text()')
    return headline, body, date

def borsen_login(driver):
    with open(params().paths['credentials'] + 'borsen.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        head = next(csv_reader)
        cred = next(csv_reader)
    borsen = dict(zip(head, cred))
    driver.find_element_by_xpath('//*[@class="coi-banner__accept"]').click()
    time.sleep(1)
    #driver.find_element_by_class_name("link login-button").click()
    time.sleep(1)
    driver.find_element_by_xpath('//*[@class="link login-button"]').click()
    time.sleep(1)
    driver.find_element_by_class_name("user").send_keys(borsen['username'])
    time.sleep(1)
    driver.find_element_by_class_name("password").send_keys(borsen['password'])
    time.sleep(1)
    driver.find_element_by_class_name("login-btn").click()
    time.sleep(5)

def init_chromedriver():
    options = webdriver.ChromeOptions()
    options.add_argument('--profile-directory=Profile 1')
    options.add_argument('--user-data-dir=C:\\Users\\Erik\\AppData\\Local\\ChromeProfiles\\User Data')
    driver = webdriver.Chrome(options=options, executable_path='C:\\Users\\Erik\\Downloads\\chromedriver_win32\\chromedriver.exe')
    return driver

def get_article(url, driver):
    try:
        driver.get(url)
    except TimeoutException:
        return '404', '404', '404'
    tree = html.fromstring(driver.page_source)
    error = tree.xpath('//*[@class="text-center"][1]/p/text()')
    if len(error) > 0:
        print("error")
        error = ' '.join(error).strip()
        _e = error.find('Vi beklager, men vi kan ikke finde siden du leder efter.')
        if _e != -1:
            return '404', '404', '404'
    headline = tree.xpath('//*[@class="headline hyphenate"]//text()')
    if len(headline) == 0:
        headline = tree.xpath('//*[@class="headline hyphenate "]//text()')
    headline = ' '.join(headline).strip().replace('\n', '')
    body = tree.xpath('//*[@itemprop="articleBody"]//text()')
    body = ' '.join(body).strip().replace('\n','')
    try:
        date = tree.xpath('//*[@class="timestamp"][1]//text()')[0]
        date = date.strip().replace('\n', '')
        date = date.replace('KL.', '')
        date = date.replace('maj', 'may')
        date = date.replace('okt', 'oct')
        print(date)
        try:
            date = pd.to_datetime(date, format="%d. %b %Y %H:%M")
        except TypeError:
            return headline, body, date
    except IndexError:
        date = tree.xpath('//*[@class="icon"]/*[@class="description"]//text()')[0]
        try:
            date = pd.to_datetime(date, format="%d / %m / %y")
            print(date)
        except TypeError:
            return headline, body, date
    return tree, headline, body, date

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_articles(driver=None, headlines=None, chunksize=2, start=0):
    with open(params().paths['credentials'] + 'google.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        head = next(csv_reader)
        cred = next(csv_reader)
    api = dict(zip(head, cred))

    if headlines is None:
        headlines = get_infomedia_headlines()
    if driver is None:
        driver = init_chromedriver()
    for i, c in enumerate(chunks(headlines, chunksize)):
        if i < start:
            continue
        print(f'Chuck {i}, size is {chunksize}')
        results = []
        for j, h in enumerate(c):
            h = h.replace('\xad', '')
            h = h.replace('\n', '')
            h = re.sub(r"^(\d\s*)", "", h)
            print(j, h)
            #url = bing_headline(h)
            q = 'site:borsen.dk ' + h
            try:
                res = google_query(q, api['api_key'], api['cse_id'], num=1, dateRestrict='m2')
            except KeyError:
                continue
            url = res[0]['link']
            print(url)
            if len(re.findall("/", url)) < 5:
                continue
            elif (url.find('.pdf') != -1) | (url.find('live/') != -1) | (url.find('aktie/') != -1) | (url.find('indeks/') != -1) | (url.find('valuta/') != -1) | (url.find('karriere/') != -1):
                continue
            headline, body, date = get_article(url, driver)
            results.append((h, url, headline, body, date))
            time.sleep(5)
        df = pd.DataFrame(results, columns=['headline_q', 'url', 'headline_web', 'bodytext', 'date'])
        filename = 'scraped_'+'_c'+str(i)+'_'+str(chunksize)+'.csv'
        print(params().paths['scraped'] + filename)
        df.to_csv(params().paths['scraped']+filename, sep=';', encoding='utf-8')
        print(f"Saved chunk to {filename}")