# script that was used for data preprocessing
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if not isinstance(text, str):
        return ""
    
    words = text.split()
    # convert to lowercase and filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def stemming(text):
    if not isinstance(text, str):
        return ""
    
    ps = PorterStemmer()
    words = text.split()
    # apply stemming then rejoin
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)

if __name__ == "__main__":
    # download dataset from huggingface and save raw data to disk
    df = pd.read_csv("hf://datasets/allenai/wildjailbreak/train/train.tsv", sep="\t")
    df.to_csv("raw_wildjailbreak_data.csv", sep='\t')

    # remove stopwords
    df["vanilla"] = df["vanilla"].apply(remove_stopwords)
    df["adversarial"] = df["adversarial"].apply(remove_stopwords)
    df["completion"] = df["completion"].apply(remove_stopwords)

    # stemming/lemmatization
    df["vanilla"] = df["vanilla"].apply(stemming)
    df["adversarial"] = df["adversarial"].apply(stemming)
    df["completion"] = df["completion"].apply(stemming)

    # save preprocessed data to disk
    df.to_csv("preprocessed_data.csv", sep='\t')
