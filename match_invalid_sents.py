import re


def get_all_invalid_sents(filename):
    data = []
    with open(filename) as fin:
        for line in fin:
            data.append(line.strip())
    return data

def get_reverse_dict(d):
    return {v: k for k, v in d.items()}


def get_clean_key(text):
    text = text.lower()
    words = re.findall("\w+", text)
    return " ".join(words)

if __name__ == "__main__":
    invalid_sents = get_all_invalid_sents("invalid_sents.txt")
    invalid_sents = {s: get_clean_key(s) for s in invalid_sents}
    invalid_reviews = get_all_invalid_sents("invalid_reviews.txt")
    invalid_reviews = {s: get_clean_key(s) for s in invalid_reviews}
    reverse_reviews = get_reverse_dict(invalid_reviews)
    not_matched_full_sentence = []
    for s in invalid_sents:
        k = invalid_sents[s]
        if k not in reverse_reviews:
            not_matched_full_sentence.append(s)
    print("unmatched number of sentences: ", len(not_matched_full_sentence))
    remaining = []
    for s in not_matched_full_sentence:
        k = invalid_sents[s]
        matched_review = None
        for review_key in reverse_reviews:
            if review_key.find(k) != -1:
                matched_review = review_key
                break
        if matched_review:
            print(k, matched_review)
        else:
            remaining.append(s)
    print("remaining unmatched number of sentences: ", len(remaining))
    for s in remaining:
        print(s)



