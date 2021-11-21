from bs4 import BeautifulSoup
import requests

with open("ht1.html") as ht1:
    soup = BeautifulSoup(ht1, "html5lib")

# print(soup.prettify())
# print(soup.div)
# print(soup.find("div")) #this will print first div
# print(soup.find("div" , class_="div2" )) # this will print first div that has class div2
# print(soup.prettify())
# r1 = soup.find("ul")
# print(len(r1))
# all_divs = soup.find_all("div") #Find_all will not stop only with first div
# print(len(all_divs))


req1 = requests.get("https://news.mit.edu/topic/machine-learning").text
soup2 = BeautifulSoup(req1, "html5lib")
result1 = soup2.find_all("div" , class_="page-term--views--list-item" )
for box in result1:
    print(box.find("a", class_="term-page--news-article--item--title--link").span.text)
    print(box.find("p", class_="term-page--news-article--item--dek").span.text)
    print(box.find("time").text)
    print("========================")



