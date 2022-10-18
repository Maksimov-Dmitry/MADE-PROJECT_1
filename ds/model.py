from gensim.utils import simple_preprocess
import nltk
from nltk.util import everygrams


class Model:
    def __init__(self, model, dictionary):
        self.model = model
        self.dictionary = dictionary
        self.stop_word = set(nltk.corpus.stopwords.words('english'))
        self.ps = nltk.stem.PorterStemmer()

    def _get_BoW_corpus(self, X):
        BoW_corpus = []
        for doc in self._docs_to_words(X):
            filtered_doc = [self._stemme(token) for token in doc if self._check(token)]
            BoW_corpus.append(self.dictionary.doc2bow(self._ngrams(filtered_doc), allow_update=False))
        return BoW_corpus

    @staticmethod
    def _docs_to_words(docs):
        for doc in docs:
            yield(simple_preprocess(str(doc), deacc=True))

    def _stemme(self, token):
        return self.ps.stem(token)
    
    def _check(self, token):
        return token not in self.stop_word

    def _ngrams(self, tokens):
        return (' '.join(a) for a in everygrams(tokens, max_len=2))

    def predict(self, X):
        BoW = self._get_BoW_corpus(X)
        result = []
        for prediction in self.model.get_document_topics(BoW):
            result.append(max(prediction, key=lambda x: x[1]))
        return result