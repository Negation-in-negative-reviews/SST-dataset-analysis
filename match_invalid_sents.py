import pickle
from match_invalid_sents_utill import get_clean_key, get_all_invalid_sents, get_reverse_dict
import numpy as np

def read_dictionary(dictionary_file, index_sentiment_map):
    phrase_sentiment_map = {}
    with open(dictionary_file, "r", encoding='utf-8') as f:
        for row in f.readlines():
            phrase = row.split("|")[0].strip()
            index = row.split("|")[1].strip()
            phrase = get_clean_key(phrase)
            if phrase in phrase_sentiment_map:
                randno = np.random.randint(low=0, high=1, size=1)
                if randno%2 == 0:
                    phrase_sentiment_map[phrase] = index_sentiment_map[index]
            else:
                phrase_sentiment_map[phrase] = index_sentiment_map[index]        
    return phrase_sentiment_map

def read_sentiment_labels(sentiment_label_file):
    index_sentiment_map = {}
    with open(sentiment_label_file, "r", encoding='utf-8') as f:
        for row in f.readlines():
            index = row.split("|")[0].strip()
            sent_score = row.split("|")[1].strip()
            index_sentiment_map[index] = sent_score

    return index_sentiment_map


if __name__=="__main__":
    index_sentiment_map = read_sentiment_labels("stanfordSentimentTreebank/sentiment_labels.txt")
    phrase_sentiment_map = read_dictionary("stanfordSentimentTreebank/dictionary.txt", index_sentiment_map)

    invalid_sents = get_all_invalid_sents("invalid_sents.txt")
    invalid_sents = {s: get_clean_key(s) for s in invalid_sents}
    invalid_reviews = get_all_invalid_sents("invalid_reviews.txt")
    invalid_reviews = {s: get_clean_key(s) for s in invalid_reviews}
    reverse_reviews = get_reverse_dict(invalid_reviews)
    reverse_sents = get_reverse_dict(invalid_sents)

    pos_reviews = {
        "pos_sents": 0,
        "neg_sents": 0,
        "total_count": 0,
        "total_sents": 0
    }

    neg_reviews = {
        "pos_sents": 0,
        "neg_sents": 0,
        "total_count": 0,
        "total_sents": 0
    }

    # unmatched_sents_count = 0
    # unmatched_reviews_count = 0
    unmatched_reviews = []
    unmatched_sents = []
    for rev in reverse_reviews:
        if rev in phrase_sentiment_map:
            sents_present = []
            pos_sents_count = 0
            neg_sents_count = 0

            for sent in reverse_sents:
                if rev.find(sent) != -1:
                    sents_present.append(sent)
                    if sent not in phrase_sentiment_map:
                        unmatched_sents.append(sent)
                    else:
                        if float(phrase_sentiment_map[sent]) >= 0.6:
                            pos_sents_count += 1
                        elif float(phrase_sentiment_map[sent]) <= 0.4:
                            neg_sents_count += 1            

            rev_score = phrase_sentiment_map[rev]
            if float(rev_score) >= 0.6:
                pos_reviews["pos_sents"] += pos_sents_count
                pos_reviews["neg_sents"] += neg_sents_count
                pos_reviews["total_count"] += 1
                pos_reviews["total_sents"] += len(sents_present)
            elif float(rev_score) <= 0.4:
                neg_reviews["pos_sents"] += pos_sents_count
                neg_reviews["neg_sents"] += neg_sents_count
                neg_reviews["total_count"] += 1
                neg_reviews["total_sents"] += len(sents_present)
        else:
            unmatched_reviews.append(reverse_reviews[rev])
    print(f"No of unmatched sentences: {len(unmatched_sents)}")
    print(f"No of unmatched reviews: {len(unmatched_reviews)}")
    print(f"Pos review: {pos_reviews}")
    print(f"neg review: {neg_reviews}")

    with open("unmatched_reviews_after_processing.txt", "w") as f:
        for rev in unmatched_reviews:
            f.write(rev.strip().strip("\n") + "\n")

    with open("unmatched_sents_after_processing.txt", "w") as f:
        for sent in unmatched_sents:
            f.write(sent.strip().strip("\n") + "\n")