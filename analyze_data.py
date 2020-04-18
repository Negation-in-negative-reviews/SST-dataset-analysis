import pickle


def read_dictionary(dictionary_file, index_sentiment_map):
    phrase_sentiment_map = {}
    with open(dictionary_file, "r", encoding='utf-8') as f:
        for row in f.readlines():
            phrase = row.split("|")[0].strip()
            index = row.split("|")[1].strip()
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
    rt_snippets_file = "stanfordSentimentTreebank/original_rt_snippets.txt"
    datasetsentences_file = "stanfordSentimentTreebank/datasetSentences.txt"
    pickle_save_file = "pickle_saves/sst_counts.pickle"

    index_sentiment_map = read_sentiment_labels("stanfordSentimentTreebank/sentiment_labels.txt")
    phrase_sentiment_map = read_dictionary("stanfordSentimentTreebank/dictionary.txt", index_sentiment_map)

    total_review_count = 0
    total_sents_count = 0
    
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

    reviews = []
    sents = []
    invalid_sents = []
    invalid_reviews = []


    from pycorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://localhost:9000')
    
    with open(rt_snippets_file, "r") as f:
        for row in f.readlines():
            row = row.strip("\n")
            output = nlp.annotate(row, properties={
                'annotators': 'ssplit',
                'outputFormat': 'json'
                })
            review = ""
            pos_sents_count = 0
            neg_sents_count = 0
            sents_count = 0
            for sent_tokens in output['sentences']:
                sent = ""
                for token in sent_tokens["tokens"]:
                    if token["originalText"] =="(":
                        sent += "("+" "
                    elif token["originalText"] == ")":
                        sent += ")"+" "
                    else:
                        sent += token["word"]+" "
                sent = sent.strip()
                sents_count += 1
                total_sents_count += 1
                if sent in phrase_sentiment_map:
                    if float(phrase_sentiment_map[sent]) >= 0.6:
                        pos_sents_count += 1
                    elif float(phrase_sentiment_map[sent]) <= 0.4:
                        neg_sents_count += 1
                    # else:
                    #     print(sent)
                else:
                    invalid_sents.append(sent)
                review += sent + " "

            total_review_count += 1
            if total_review_count%1000 == 0:
                print("Total review count: ", total_review_count)
            review = review.strip()
            if review in phrase_sentiment_map:
                if float(phrase_sentiment_map[review]) >= 0.6:
                    pos_reviews["total_count"] += 1
                    pos_reviews["pos_sents"] += pos_sents_count
                    pos_reviews["neg_sents"] += neg_sents_count
                    pos_reviews["total_sents"] += sents_count
                elif float(phrase_sentiment_map[review]) <= 0.4:
                    neg_reviews["total_count"] += 1
                    neg_reviews["pos_sents"] += pos_sents_count
                    neg_reviews["neg_sents"] += neg_sents_count         
                    neg_reviews["total_sents"] += sents_count
                # else:
                #     print("review: ", review)
            else:
                invalid_reviews.append(review)
            

    print(f"total_review_count: {total_review_count}")
    print(f"total_sent_count: {total_sents_count}")
    print("neg_reviews: ", neg_reviews)
    print("pos_reviews: ", pos_reviews)
    pickle.dump({
        "total_review_count": total_review_count,
        "total_sent_count": total_sents_count,
        "neg_reviews": neg_reviews,
        "pos_reviews": pos_reviews
    }, open(pickle_save_file, "wb"))

    with open("invalid_reviews.txt", "w") as f:
        for review in invalid_reviews:
            f.write(review+"\n")

    with open("invalid_sents.txt", "w") as f:
        for sent in invalid_sents:
            f.write(sent+"\n")
            