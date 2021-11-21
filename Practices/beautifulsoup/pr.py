from bs4 import BeautifulSoup
import requests
import csv

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


csv_file = open("result1.csv" , "w")
write_to_csv = csv.writer(csv_file)
write_to_csv.writerow(['image_link' , 'title' , 'text' , 'post_time' ])

req1 = requests.get("https://news.mit.edu/topic/machine-learning").text
soup2 = BeautifulSoup(req1, "html5lib")
result1 = soup2.find_all("div" , class_="page-term--views--list-item" )
for box in result1:
    try:
        img = box.find("img")["src"]
    except:
        img = None
    try:
        title = box.find("a", class_="term-page--news-article--item--title--link").span.text
    except:
        title = None
    try:
        text = box.find("p", class_="term-page--news-article--item--dek").span.text
    except:
        text = None
    try:
        time_r = box.find("time").text
    except:
        time_r = None
    print(img)
    print(title)
    print(text)
    print(time_r)
    print("========================")
    write_to_csv.writerow([img , title , text , time_r])

csv_file.close()



