from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import requests
import time, os

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

# query = "data science"
# youtube_search = "https://www.youtube.com/results?search_query="
# youtube_query = youtube_search + query.replace(' ', '+')

# driver.get("https://www.techwithtim.net")
# # driver.implicitly_wait(10)
# # driver.time_sleep(20) # Does not Work
# ele1 = driver.find_element_by_name("s")
# ele1.clear()
# ele1.send_keys("TEST")  # wirte to input
# ele1.send_keys(Keys.RETURN)  # press enter
# try:
#     ele2 = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.ID, "main"))
#     )
#     articles = ele2.find_elements_by_tag_name("article")
#     for article in articles:
#         summary = article.find_element_by_class_name("entry-summary")
#         print(summary.text)
# except:
#     driver.quit()



#===========================
#===========================

# driver.get("https://www.techwithtim.net")
# ele1 = driver.find_element_by_link_text("Python Programming")
# ele1.click()
# try:
#     ele2 = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.LINK_TEXT, "Beginner Python Tutorials"))
#     )
#     ele2.click()
#     ele3 = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.ID, "sow-button-19310003"))
#     )
#     ele3.click()
#
#     driver.back()
#     driver.back()
#     driver.back()
#     driver.forward()
#
# except:
#     driver.quit()


#==================
#==================

driver.get("https://www.imdb.com")
search = driver.find_element_by_id("suggestion-search")
search.send_keys("iron man2")
search.send_keys(Keys.RETURN)
#

try:
    ele2 = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.LINK_TEXT, "Iron Man 2"))
    )
    ele2.click()
    # ele3 = driver.find_element_by_class_name("")
    ele3 = driver.find_element(By.CLASS_NAME, "TitleHeader__TitleText-sc-1wu6n3d-0" )
    print(ele3.text)
except:
    driver.quit()


# ele1 = driver.find_element_by_link_text("Python Programming")
# ele1.click()
# try:
#     ele2 = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.LINK_TEXT, "Beginner Python Tutorials"))
#     )
#     ele2.click()
#     ele3 = WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.ID, "sow-button-19310003"))
#     )
#     ele3.click()
#
#     driver.back()
#     driver.back()
#     driver.back()
#     driver.forward()
#
# except:
#     driver.quit()







