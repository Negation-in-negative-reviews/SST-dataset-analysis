# from stanfordcorenlp import StanfordCoreNLP

# nlp = StanfordCoreNLP("stanford-corenlp-full-2018-10-05")

# sentence = "The gorgeously elaborate continuation of \
#     ``The Lord of the Rings'' trilogy is so huge that a \
#         column of words cannot adequately describe co-writer/director \
#             Peter Jackson's expanded vision of J.R.R. Tolkien's Middle-earth."
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))
# print('Dependency Parsing:', nlp.dependency_parse(sentence))


from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
text =  "(L)ame and unnecessary."

# text = "This is a good script, good dialogue, funny even for adults. The characters are interesting and often very creatively constructed from figure to backstory. The film will play equally well on both the standard and giant screens."
output = nlp.annotate(text, properties={
  'annotators': 'tokenize,ssplit,pos,depparse,parse',
  'outputFormat': 'json'
#   'tokenizerOptions': 'normalizeParentheses=true'
  })
print(output["sentences"][0].keys())

for sent_tokens in output['sentences']:
    sent = ""
    for token in sent_tokens["tokens"]:
        sent += token["word"]+" "
    print(sent.strip())
# print([0].keys())