from bs4 import BeautifulSoup
import requests
import csv



req1 = requests.get("https://www.imdb.com/title/tt1228705/?ref_=nv_sr_srsg_6").text
soup2 = BeautifulSoup(req1, "html5lib")
# print(soup2.prettify())
print(soup2.find("div"))
# result1 = soup2.find_all("div" , class_="page-term--views--list-item" )
# for box in result1:
#     try:
#         img = box.find("img")["src"]
#     except:
#         img = None
#     try:
#         title = box.find("a", class_="term-page--news-article--item--title--link").span.text
#     except:
#         title = None
#     try:
#         text = box.find("p", class_="term-page--news-article--item--dek").span.text
#     except:
#         text = None
#     try:
#         time_r = box.find("time").text
#     except:
#         time_r = None
#     print(img)
#     print(title)
#     print(text)
#     print(time_r)
#     print("========================")



