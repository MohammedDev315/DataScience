import pandas as pd

df = pd.read_csv("result4_after_adding_cast_score.csv")
director_df = pd.read_csv("directors.csv")


def clean_rank_of_directors(data_in):
    try:
        return int(data_in.replace("," , ""))
    except:
        return 0

director_df['cleaned_rank'] = director_df.loc[: , 'rank'].apply(clean_rank_of_directors)


def convert_time_to_minuts(time_1):
    minuts = int(time_1.split(" ")[1].split("m")[0])
    horse_to_minuts = int(time_1.split(" ")[0].split("h")[0])*60
    total_time_in_minuts = horse_to_minuts + minuts
    return total_time_in_minuts


def convert_box_office_to_full_number(data_in):
    if "M" in data_in:
        data_in = data_in.replace("M",'')
        data_in = data_in.replace("$" , '')
        millions = int(data_in.split('.')[0])*1000000
        thousands = int(data_in.split('.')[1])*100000
        data_in = millions + thousands
    elif "K" in data_in:
        data_in = data_in.replace("K",'')
        data_in = data_in.replace("$" , '')
        millions = int(data_in.split('.')[0])*100000
        thousands = int(data_in.split('.')[1])*10000
        data_in = millions + thousands
    else:
        data_in = int(data_in.replace("$" , ''))
    return data_in

def get_year_only(data_in):
    return data_in.split(' ')[2]


def score_of_director(data_in):
    total = 0
    for director_name in data_in.split(","):
        maks = (director_df.loc[:, 'name'] == director_name)
        try:
            total = total + int(director_df[maks]["cleaned_rank"])
            return 12000 - total
        except:
            return 0

#
df["film_income"] = df.loc[: , "box_office"].apply(convert_box_office_to_full_number)
df["runtime_minuts"] = df.loc[: , 'runtime'].apply(convert_time_to_minuts)
df["release_year"] = df.loc[: , "release_date_theaters"].apply(get_year_only)
df["director_rank"] = df.loc[: , "director"].apply(score_of_director)

df_final = df.loc[: , [
'filem_name', 'genre', 'director_rank' , 'director',
       'sum_of_first_five', 'film_income', 'runtime_minuts',
       'release_year' ]]


df_final.to_csv("result5_after_ranking_runtime_income.csv", encoding='utf-8', index=False)



