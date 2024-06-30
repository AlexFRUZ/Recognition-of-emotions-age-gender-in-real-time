import pandas as pd
import os

data = pd.read_csv(r"C:\Users\user1\OneDrive\Робочий стіл\cyberbullying_tweets.csv")

output_folder = "Cyber_Bulling"
os.makedirs(output_folder, exist_ok=True)

for index, row in data.iterrows():
    cyberbullying_type = row['cyberbullying_type']
    tweet = row['tweet_text']
    
    type_folder = os.path.join(output_folder, cyberbullying_type)
    os.makedirs(type_folder, exist_ok=True)
    
    file_name = f"tweet_{index}.txt"
    file_path = os.path.join(type_folder, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(tweet)
