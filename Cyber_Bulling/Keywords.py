import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('stopwords')

def extract_keywords(texts, top_n=5):
    all_text = ' '.join(texts)
    
    tokens = word_tokenize(all_text.lower())
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    freq_dist = FreqDist(tokens)
    
    keywords = [word for word, _ in freq_dist.most_common(top_n)]
    
    return keywords

def load_texts_from_directory(directory):
    texts = {}
    for category in os.listdir(directory):
        category_dir = os.path.join(directory, category)
        if os.path.isdir(category_dir):
            category_texts = []
            for file_name in os.listdir(category_dir):
                with open(os.path.join(category_dir, file_name), 'r', encoding='utf-8') as file:
                    category_texts.append(file.read())
            texts[category] = category_texts
    return texts

def load_texts_with_top_keywords_from_directory(directory, top_n_keywords=100):
    texts = load_texts_from_directory(directory)
    top_keywords = {}
    for category, category_texts in texts.items():
        top_keywords[category] = extract_keywords(category_texts, top_n=top_n_keywords)
    return texts, top_keywords

TRAIN_DIR = r"D:\Train\Cyber_Bulling\Train" 
TEST_DIR = r"D:\Train\Cyber_Bulling\Test"

train_texts, train_top_keywords = load_texts_with_top_keywords_from_directory(TRAIN_DIR)
test_texts, test_top_keywords = load_texts_with_top_keywords_from_directory(TEST_DIR)

print("Most common keywords in training data:")
for category, keywords in train_top_keywords.items():
    print(f"Category: {category}, Keywords: {keywords}")

print("\nMost common keywords in testing data:")
for category, keywords in test_top_keywords.items():
    print(f"Category: {category}, Keywords: {keywords}")
