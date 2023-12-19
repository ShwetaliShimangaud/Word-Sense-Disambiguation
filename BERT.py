import torch
from transformers import BertTokenizer, BertForTokenClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import accuracy_score

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

def BERT_IMP(context,word):

    # Tokenize, lemmatize, and remove stop words
    tokens = word_tokenize(context)
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]

    # Get BERT input embeddings
    input_ids = tokenizer.encode(" ".join(processed_tokens), return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert prediction to sense
    try:
        predicted_sense_index = predictions[0, tokens.index(word)+1].item()  # +1 to skip [CLS] token
        predicted_sense = wordnet.synsets(word)[predicted_sense_index].name()

        return predicted_sense
    except:
         return wordnet.synsets(word)[0].name()



def dev_test_accuracy(dev_instances, test_instances, dev_key, test_key):

    dev_corpus=[]
    dev_index=[]
    dev_context=[]
    dev_lemma=[]
    dev_sense_key=[]
    dev_sense=[]
    for data in dev_instances:
        #   Tokenizing Lemma and context
        current_lemma=dev_instances[data].lemma.decode('utf-8')
        current_context=[]
        for tmp in dev_instances[data].context:
            current_context.append(tmp.decode('utf-8'))
        current_context_string=' '.join(current_context)
        for poss_op in dev_key[data]:
                dev_index.append(dev_instances[data].id)
                dev_context.append(current_context_string)
                dev_lemma.append(current_lemma)
                dev_sense_key.append(poss_op)
                dev_sense.append(wn.synset_from_sense_key(poss_op).name())
                dev_corpus.append([current_context_string,current_lemma,wn.synset_from_sense_key(poss_op).name()])

    test_corpus=[]
    test_index=[]
    test_context=[]
    test_lemma=[]
    test_sense_key=[]
    test_sense=[]
    for data in test_instances:
        #   Tokenizing Lemma and context
        current_lemma=test_instances[data].lemma.decode('utf-8')
        current_context=[]
        for tmp in test_instances[data].context:
            current_context.append(tmp.decode('utf-8'))
        current_context_string=' '.join(current_context)
        for poss_op in test_key[data]:
                test_index.append(test_instances[data].id)
                test_context.append(current_context_string)
                test_lemma.append(current_lemma)
                test_sense_key.append(poss_op)
                test_sense.append(wn.synset_from_sense_key(poss_op).name())
                test_corpus.append([current_context_string,current_lemma,wn.synset_from_sense_key(poss_op).name()])
    
    prediction_dev=[BERT_IMP(data[0],data[1]) for data in dev_corpus]
    prediction_test=[BERT_IMP(data[0],data[1]) for data in test_corpus]

    print("Accuracy on the Dev Set with pre-trained model is :", accuracy_score(prediction_dev, dev_sense)*100, "%")
    print("Accuracy on the Test Set with pre-trained model is :", accuracy_score(prediction_test, test_sense)*100, "%")


