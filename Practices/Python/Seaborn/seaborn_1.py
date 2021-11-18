import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import math 
import random
import matplotlib.pyplot as plt

nba_df = pd.read_csv("NBA_GameLog_2010_2017.csv")
nba_df['Date'] = pd.to_datetime(nba_df['Date'])


tor_2016_2017 = nba_df.loc[ (nba_df.loc[:, "Season"] == 2017) & (nba_df.loc[:, "Team"] == "TOR") ]
freq_table = pd.DataFrame(tor_2016_2017.loc[:, "W.L"].value_counts())
freq_table.columns = ["Frequency"]
print(freq_table)
freq_table.Frequency.sum()
freq_table["RelativeFrequency"] = freq_table["Frequency"] / freq_table.Frequency.sum()
print(freq_table)


plt.figure(figsize = [6,7])
# plt.pie(freq_table.RelativeFrequency, labels=["W", "L"])
# plt.bar(freq_table.index,freq_table.RelativeFrequency)
# plt.hist(tor_2016_2017.loc[:, "Tm.Pts"])
# sns.distplot(tor_2016_2017.loc[:, "Tm.FG_Perc"],kde=False) #using seaborn
tor_2016_2017_home = tor_2016_2017.loc[(tor_2016_2017.loc[:, "Home"] == 1), ]
# plt.scatter(tor_2016_2017_home.index, tor_2016_2017_home.loc[:, "Home.Attendance"])



team_points_prob = tor_2016_2017.loc[:, "Tm.Pts"].value_counts() / tor_2016_2017.loc[:, "Tm.Pts"].value_counts().sum()
team_points = tor_2016_2017.loc[:, "Tm.Pts"].value_counts().index
plt.bar(team_points, team_points_prob)


print(tor_2016_2017_home.loc[:, "Home.Attendance"].describe())
aa = tor_2016_2017.loc[(tor_2016_2017.loc[:,'Home'] == 1)&(tor_2016_2017.loc[:, "Home.Attendance"] >  19800)]
print(tor_2016_2017.loc[:, "Home.Attendance"].describe())
print(aa.loc[: , "Home.Attendance" ].describe())
print(aa.loc[: , "Home.Attendance" ].head(10))
print(len(aa.loc[: , "Home.Attendance" ]))
# sns.jointplot(tor_2016_2017.loc[:, "G"], tor_2016_2017.loc[:, "Tm.TOV"]) ## combine scatter plot and histogram of the turnovers
# sns.jointplot(tor_2016_2017_home.index, tor_2016_2017_home.loc[:, "Home.Attendance"]) ## combine scatter plot and histogram of the turnovers
print(tor_2016_2017.loc[:, ["Tm.Pts", "Opp.Pts", 'Tm.FGM', 'Tm.FGA', 'Tm.FG_Perc', 'Tm.3PM', 'Tm.3PA',
       'Tm.3P_Perc', 'Tm.FTM', 'Tm.FTA', 'Tm.FT_Perc', 'Tm.ORB', 'Tm.TRB',
       'Tm.AST', 'Tm.STL', 'Tm.BLK', 'Tm.TOV', 'Tm.PF', 'Home.Attendance']].mean(axis=0))
print("====== Median =======")
print(np.median(tor_2016_2017.loc[:, ["Tm.Pts", "Opp.Pts", 'Tm.FGM', 'Tm.FGA', 'Tm.FG_Perc', 'Tm.3PM', 'Tm.3PA',
       'Tm.3P_Perc', 'Tm.FTM', 'Tm.FTA', 'Tm.FT_Perc', 'Tm.ORB', 'Tm.TRB',
       'Tm.AST', 'Tm.STL', 'Tm.BLK', 'Tm.TOV', 'Tm.PF', 'Home.Attendance']], axis=0))

plt.show(block=True)
plt.interactive(False)


