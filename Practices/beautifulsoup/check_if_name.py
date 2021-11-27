import pandas as pd
df1 = pd.read_csv('film_stars2.csv')
df2 = pd.read_csv('result3_after_dropping.csv')

# because data has ',' we have to remove to convert rank to integer
for x in range(len(df1)):
    aa = df1.iloc[x , 0]
    if x >= 999:
        df1.iloc[x , 0] = int(f"{aa.split(',')[0]}{aa.split(',')[1]}")
    else:
        df1.iloc[x , 0] = int(aa)

df1['rank'] = df1['rank'].astype(int)

#Join each name with its rank.
names = []
for index_num in range(len(df1)):
    name_with_rank = f'{df1.iloc[index_num,1].lower()} = {df1.iloc[index_num,0]}'
    names.append(name_with_rank)

#convert all filem_stars with their filems to a list and append it to big list
all_filems_and_name = []
# for x in range(len(df2)):
for x in range(100):
    tem_filem_and_names = []
    tem_filem_and_names.append(df2.iloc[x , 0])
    for name in df2.iloc[x , 16].split(',')[:-1]:
        tem_filem_and_names.append(name)
    all_filems_and_name.append(tem_filem_and_names)

count_less_than_3 = 0
#convert previous list to rank using rankkin list
for filem_name_one_list in all_filems_and_name:
    filem_name_result = ['',0,0,0,0,0,0,0,0,0,0,0]
    # these to lines will detremin which rows have huge amount of missing vlaue
    first_five_none = 0
    total_none = 0
    filem_name_result[0] = filem_name_one_list[0] # get the name of file and put it first ele
    #Check each name with each rank and put it on fillem_name_result based on its order
    for num in range(1,10):
        searched_name = filem_name_one_list[num].lower().strip()
        matching = [s.lower() for s in names if searched_name in s.lower()]
        if len(matching) > 0:
            rank = 10000 - ( int(matching[0].split('=')[1].strip()) )
            filem_name_result[num] = rank
        else:
            filem_name_result[num] = None
            total_none = total_none + 1
            if num <= 5:
                first_five_none =first_five_none + 1

    filem_name_result[10] = first_five_none
    filem_name_result[11] = total_none
    if first_five_none <= 2 :
        count_less_than_3 = count_less_than_3 + 1
        print(filem_name_result)

print(count_less_than_3)

