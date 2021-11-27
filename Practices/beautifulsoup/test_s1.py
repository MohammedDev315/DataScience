from bs4 import BeautifulSoup
import requests
import csv

with open("tt1.html") as ht1:
    soup = BeautifulSoup(ht1, "html5lib")

# print(soup.find('a').text)
# print(soup.find_all('p')[-1].find('a').text)
# print(soup.find(text='New York Yankees'))

# print({e['itemprop'] : e.text.strip() for e in soup.find_all('strong')})
print({e['itemprop'] : e.text.strip() for e in soup.find_all(attrs={'itemprop':True})})
# print({e.text : e.next for e in soup.find_all('strong')})
# print({e.text : e.next for e in soup.find_all(attrs={'itemprop':True})})



# print(soup.find('strong')[-1].find('a').text)
# print(soup.find("div")) #this will print first div
# print(soup.find("div" , class_="div2" )) # this will print first div that has class div2
# print(soup.prettify())
# r1 = soup.find("ul")
# print(len(r1))
# all_divs = soup.find_all("div") #Find_all will not stop only with first div
# print(len(all_divs))