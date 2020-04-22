# sst-dataset-analysis


How to run
----
1. Download the stanford corenlp from [here](https://stanfordnlp.github.io/CoreNLP/). 
2. Start the corenlp server by running the following command inside the downloaded directory.
  ```
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
  ```
3. Install `pycorenlp` using the below command. Link for its repo [here](https://github.com/smilli/py-corenlp)
```
pip install pycorenlp
```
4. Extract `stanfordSentimentTreebank.zip` to the project directory.
5. Create a `pickle_saves` directory in the project directory.
6. Run the `analyze_data.py` file.


Notes
----
* No of unmatched sents: 1
* No of unmatched reviews: 67
This is by using `\w+` regex to process the remaining unmatched reviews and sentences
