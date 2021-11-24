from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import time, os

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

# query = "data science"
# youtube_search = "https://www.youtube.com/results?search_query="
# youtube_query = youtube_search + query.replace(' ', '+')

driver.get(youtube_query)


print(driver.page_source[:1000])

