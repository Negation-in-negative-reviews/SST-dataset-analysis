# import nltk
# nltk.download('punkt')
import stanfordnlp
# import unidecode
import pickle

nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')

def read_dictionary(dictionary_file, index_sentiment_map):
    phrase_sentiment_map = {}
    with open(dictionary_file, "r", encoding='utf-8') as f:
        for row in f.readlines():
            phrase = row.split("|")[0].strip()
            index = row.split("|")[1].strip()
            phrase = phrase.replace(" ", "").strip().lower()      
            if phrase in phrase_sentiment_map.keys():
                print(f"duplicate: {phrase}, actual text: {row}")
                phrase_sentiment_map[phrase] = max(index_sentiment_map[index], phrase_sentiment_map[phrase])
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
    snippets_file = "stanfordSentimentTreebank/original_rt_snippets.txt"
    datasetsentences_file = "stanfordSentimentTreebank/datasetSentences.txt"
    pickle_save_file = "pickle_saves/sst_counts.pickle"

    index_sentiment_map = read_sentiment_labels("stanfordSentimentTreebank/sentiment_labels.txt")
    phrase_sentiment_map = read_dictionary("stanfordSentimentTreebank/dictionary.txt", index_sentiment_map)

    total_review_count = 0
    total_sent_count = 0
    
    pos_review_count = {
        "pos_sents": 0,
        "neg_sents": 0,
        "total_count": 0,
        "total_sents": 0
    }

    neg_review_count = {
        "pos_sents": 0,
        "neg_sents": 0,
        "total_count": 0,
        "total_sents": 0
    }

    reviews = []
    sents = []
    invalid_sents = []
    invalid_reviews = []
    # with open(datasetsentences_file, "r", encoding='utf-8') as f:
    #     f.readline()
    #     for row in f.readlines():
    #         idx = row.split("\t")[0]
    #         sent = row.split("\t")[1].replace(" ", "").strip("\n")
    #         # if sent_tokenized not in phrase_sentiment_map:
    #         #     invalid_sents.append(sent_tokenized)
    #         #     continue
    #         # total_sent_count += 1
    #         if float(phrase_sentiment_map[sent]) >= 0.6:
    #             possent_count += 1
    #         elif float(phrase_sentiment_map[sent]) <= 0.4:
    #             negsent_count += 1
            
    nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
    with open(snippets_file, "r") as f:
        for row in f.readlines():            
            doc = nlp(row)
            pos_sents_count = 0
            neg_sents_count = 0
            total_sent_count = 0
            review = ""
            for i, sentence in enumerate(doc.sentences):
                print(f"====== Sentence {i+1} tokens =======")
                sent = "".join([token.text for token in sentence.tokens]).strip()
                sent = sent.lower()
                review += sent
                total_sent_count += 1
                
                if float(phrase_sentiment_map[sent]) >= 0.6:
                    pos_sents_count += 1
                elif float(phrase_sentiment_map[sent]) <= 0.4:
                    neg_sents_count += 1
            if float(phrase_sentiment_map[review]) >= 0.6:
                pos_review_count["total_count"] += 1
                pos_review_count["pos_sents"] += pos_sents_count
                pos_review_count["neg_sents"] += neg_sents_count
                pos_review_count["total_sents"] += total_sent_count
            elif float(phrase_sentiment_map[review]) <= 0.4:
                neg_review_count["total_count"] += 1
                neg_review_count["pos_sents"] += pos_sents_count
                neg_review_count["neg_sents"] += neg_sents_count         
                neg_review_count["total_sents"] += total_sent_count
            # flag = True
            # for i, sentence in enumerate(tokens.sentences):
            #     sent_tokenized = ""

            #     for token in sentence.tokens:
            #         review_tokenized += token.text+" "
            #         sent_tokenized += token.text +" "

            #     sent_tokenized = sent_tokenized.strip()

            #     if sent_tokenized not in phrase_sentiment_map:
            #         invalid_sents.append(sent_tokenized)
            #         flag = False
            #         continue
            #     total_sent_count += 1
            #     if float(phrase_sentiment_map[sent_tokenized]) >= 0.6:
            #         pos_sents_count += 1
            #     elif float(phrase_sentiment_map[sent_tokenized]) <= 0.4:
            #         neg_sents_count += 1
                
                
            # review_tokenized = review_tokenized.strip()
            # if review_tokenized not in phrase_sentiment_map:
            #     invalid_reviews.append(review_tokenized)
            #     continue
            # total_review_count += 1
            # if flag:
            #     if float(phrase_sentiment_map[review_tokenized]) >= 0.6:
            #         pos_review_count["total_count"] += 1
            #         pos_review_count["pos_sents"] += pos_sents_count
            #         pos_review_count["neg_sents"] += neg_sents_count
            #         pos_review_count["total_sents"] += total_sent_count
            #         # posreview_count += 1
            #     elif float(phrase_sentiment_map[review_tokenized]) <= 0.4:
            #         # negreview_count += 1      
            #         neg_review_count["total_count"] += 1
            #         neg_review_count["pos_sents"] += pos_sents_count
            #         neg_review_count["neg_sents"] += neg_sents_count         
            #         neg_review_count["total_sents"] += total_sent_count

    print(f"total_review_count: {total_review_count}")
    print(f"total_sent_count: {total_sent_count}")
    pickle.dump({
        "total_review_count": total_review_count,
        "total_sent_count": total_sent_count,
        "neg_review_count": neg_review_count,
        "pos_review_count": pos_review_count
        # "posreview_count": posreview_count,
        # "negreview_count": negreview_count,
        # "possent_count": possent_count, 
        # "negsent_count": negsent_count
    }, open(pickle_save_file, "wb"))

    with open("reviews_couldnt_find.txt", "w") as f:
        for review in invalid_reviews:
            f.write(review+"\n")

    with open("sents_couldnt_find.txt", "w") as f:
        for sent in invalid_sents:
            f.write(sent+"\n")
            