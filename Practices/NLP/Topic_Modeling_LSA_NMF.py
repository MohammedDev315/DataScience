#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity


example = ['Football baseball basketball',
            'baseball giants cubs redsox',
            'football broncos cowboys',
            'baseball redsox tigers',
            'pop stars hendrix prince',
            'hendrix prince jagger rock',
            'joplin pearl jam tupac rock'
          ]
#%%
vectorizer = CountVectorizer(stop_words='english')
doc_word = vectorizer.fit_transform(example)
print(doc_word.shape)
pd.DataFrame(doc_word.toarray(), index=example, columns=vectorizer.get_feature_names()).head(10)
#%%
lsa = TruncatedSVD(2)
doc_topic = lsa.fit_transform(doc_word)
print(doc_topic)
print(lsa.explained_variance_ratio_)
#%%
topic_word = pd.DataFrame(lsa.components_.round(3) , index = ["component_1","component_2"], columns = vectorizer.get_feature_names() )
print(topic_word)
print(len(lsa.components_))
for ix, topic in enumerate(lsa.components_):
    print(ix)
    print(topic)
#%%
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
#%%
display_topics(lsa, vectorizer.get_feature_names(), 5)
#%%
Vt = pd.DataFrame(doc_topic.round(5),
             index = example,
             columns = ["component_1","component_2" ])
#%%
print(cosine_similarity((doc_topic[0], doc_topic[1])).round())
print(cosine_similarity((doc_topic[0], doc_topic[6])).round())
#%%
example = ['Hamilton brought a boost. The Lion King provided ballast. And Broadway, once again, broke a record: The theater season that just ended attracted more people, and more money, than any before. Broadway seems to be defying the cultural odds: An ancient art form in the digital age, it is strengthening thanks to an everincreasing influx of tourists and a resurgent enthusiasm for musical theater. The season that ended on Sunday included 13,317,980 visitors to Broadway shows  a record number, up 1.6 percent over the previous season, according to figures released on Monday by the Broadway League. Theaters grossed 1.373 billion, also a record, up 0.6 percent over the previous season, although the grosses are not adjusted for inflation. Once again, Simba ruled supreme: The Lion King, still mighty more than 18 years after it opened, grossed 102.7 million on Broadway last season, far outpacing any other show. The musical, which has multiple productions running simultaneously around the globe, has grossed more than 6.2 billion worldwide, and has been seen by 85 million people over its history, according to Disney; by contrast, 478,605 people have seen the Broadway production of Hamilton thus far.Hamilton (featuring a onetime Simba, Christopher Jackson, in the role of George Washington) offered an enormous jolt of energy to the Broadway season. This hiphop musical about Americas founding fathers has dominated the cultural conversation, raked in awards and been celebrated at the White House. Many Broadway leaders believe the show has helped the industry as a whole, bringing attention from corners of the culture that have long preferred to mock jazz hands and dream ballets.',
          'When Candace Payne, aka the Chewbacca Mask Mom, sat in her car last Thursday filming her new Hasbro toy, an electronic Chewbacca Mask from Kohls, she inadvertently made history  not just for Facebook Live as its most popular video, but for the entire haul and unboxing video genre. Paynes video starts out like every other video in the genre  she talks about her shopping trip, and is incredibly excited to show the viewer her new purchase   but after that, the similarities stop. Shes not in a bedroom, but in her car, and Payne isnt describing multiple purchases, just one. The platform, execution and reception of her vlog has impacted the genre, in quite a few ways. First, Paynes video actually went viral among ordinary people, something that doesnt really happen to other haul and unboxing videos  not to this extent. While it is true boxing and haul videos by top YouTube vloggers will get a few million views (only!) thanks to the large communities said vlogger has built over the years, no one has ever seen an instant worldwide smash hit like Paynes video. Grandparents and aunts that dont even know what a haul video is were watching, liking, and sharing Paynes video.',
          'LOS ANGELES (AP)  An original animators desk from Walt Disney Studios and a vintage Mickey Mouse doll signed by Walt Disney are among the items up for bid next month in an online auction of rare Disney memorabilia. The website of Van Eaton Galleries lists more than 700 items for sale. Among the items listed are original production cels for Disney classics like The Jungle Book, Sleeping Beauty, Bambi and Snow White and the Seven Dwarfs. Collectors can also bid on costumes from the original Mickey Mouse Club, including one worn by Annette Funicello. An exhibition titled, Collecting Disney, opens Wednesday at the gallery in Sherman Oaks, California, ahead of the online auction that begins June 18.',
          'After putting together one of their best playoff performances in a must win Game 3 on Saturday, the Toronto Raptors picked up where they left off in Mondays Game 4, with AllStar guards Kyle Lowry and DeMar DeRozan finally teaming up for a complete performance. Lowry (35 points) and DeRozan (32 points) shot a combined 28 for 43 for 67 points and became the first teammates in a conference finals series to score 30  plus points on 60% or better shooting since Charles Barkley and Dan Majerle for Phoenix Suns in 1993, further proving that when the starting backcourt is on, the Raptors are extremely difficult to beat. Those numbers are of stark contrast to the majority of the Raptors first two playoff series, where both Lowry and DeRozan struggled mightily to deliver significant offensive production.',
          'The Cleveland Cavaliers enjoyed one of their most devastating 12 minutes of offensive basketball in the second half Monday night and, considering their playoff run, thats saying something. But it came after a long stretch of some of their most puzzling play in weeks, and that cost them a valuable playoff game. The Toronto Raptors evened the Eastern Conference finals at 22 with a 10599 victory after holding on in the face of a vicious Cavs late rally. Yet, as well as the Raptors played  stars Kyle Lowry and DeMar DeRozan were just terrific with a combined 67 points, the most theyve ever scored as teammates  it really came amid some headscratching, gameplan adjustments by coach Tyronn Lue. After spending the past few weeks finding a rhythm that has produced mostly spectacular results, Lue completely changed his rotations in the first half in what seemed like an overreaction from the Game 3 loss.',
          'Leave it to Rich Hill to end the As four game losing streak. The last time Oakland had won before Monday, Hill was on the mound. And at Safeco Field, he was magnificent, working calmly and efficiently whether the bases were empty or full. Hill pitched eight scoreless innings to help the As top the division leading Mariners 5 0. The As have won all four games theyve played at Seattle this season. Oakland has 20 wins and Hill has seven of them, the most for an As pitcher before the end of May since Mark Mulder had eight in 2003, a year Mulder made the All Star team. Every game he goes out there we feel were going to win, no matter what were going through, Oakland manager Bob Melvin said. He brings a lot of intensity to the mound, a lot of fight. Hill hasnt allowed more than three earned runs in any of his 10 starts and his ERA is down to 2.18. He also became the first As starter to pitch into the eighth inning since Sonny Gray pitched eight innings last Aug. 22, a span of 83 games; Melvin said his plan was to use only Hill and closer Ryan Madson, and Hill even wanted to go back out for the ninth after throwing 107 pitches. Hills streak of starts in which he gave up no more than four hits while working at least five innings ended at six; the Mariners recorded eight hits off him, few of them struck well. Hills streak was the best in franchise history dating to at least 1913. Seattle loaded the bases with no outs in the second inning without hitting the ball hard, with Nelson Cruzs infield single, an opposite field flare by Dae Ho Lee and a bloop to center by Kyle Seager. At that point, Hill said, second baseman Chris Coghlan came over to him and said, Control what you can control.']
ex_label = [e[:30]+"..." for e in example]
#%%
vectorizer = CountVectorizer(stop_words = 'english')
doc_word = vectorizer.fit_transform(example)
pd.DataFrame(doc_word.toarray(), index=ex_label, columns=vectorizer.get_feature_names()).head(10)
#%%
nmf_model = NMF(2)
doc_topic = nmf_model.fit_transform(doc_word)
topic_word = pd.DataFrame(nmf_model.components_.round(3),
             index = ["component_1","component_2"],
             columns = vectorizer.get_feature_names())

display_topics(nmf_model, vectorizer.get_feature_names(), 10)
#%%
H = pd.DataFrame(doc_topic.round(5),
             index = ex_label,
             columns = ["component_1","component_2" ])
print(H.head())
