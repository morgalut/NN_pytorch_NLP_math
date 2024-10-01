import spacy

class ArithmeticNLP:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentence(self, sentence):
        doc = self.nlp(sentence)
        for token in doc:
            print(f"{token.text}: {token.dep_} (head: {token.head.text})")

    def preprocess_text(self, text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
