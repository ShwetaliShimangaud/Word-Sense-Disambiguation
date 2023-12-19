#   Task 0 : Importing required libraries and loading base code

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import accuracy_score

import loader
dev_instances, test_instances, dev_key, test_key=loader.main()
# print(len(dev_instances)) # number of dev instances
# print(len(test_instances)) # number of test instances

#   Task 1 : The most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet
print("Task 1 : The most frequent sense baseline")
def get_most_frequent_sense(word):
    senses = wn.synsets(word)
    if not senses:
        return None

    # Count occurrences of each sense in the WordNet corpus
    sense_counts = Counter(sense for synset in senses for sense in synset.lemmas())
    # Sort senses by frequency in descending order
    sorted_senses = sorted(sense_counts, key=lambda x: sense_counts[x], reverse=True)

    # Return the most frequent sense (sense #1)
    return sorted_senses[0]

match_count=0
for data in dev_instances:
    current_lemma=dev_instances[data].lemma.decode('utf-8')
    most_frequent_sense = get_most_frequent_sense(current_lemma)

    if most_frequent_sense:
        # print(f"The most frequent sense for '{current_lemma}' is: {most_frequent_sense.key()}")
        if(most_frequent_sense.key() in dev_key[data]):
            match_count+=1
    else:
        print(f"No sense found for '{current_lemma}'.")
    
accuracy=(match_count/len(dev_instances))*100
print(f"Accuracy of the most frequency sense baseline model on Dev Instances is : {accuracy} %")

match_count=0
for data in test_instances:
    current_lemma=test_instances[data].lemma.decode('utf-8')
    most_frequent_sense = get_most_frequent_sense(current_lemma)

    if most_frequent_sense:
        # print(f"The most frequent sense for '{current_lemma}' is: {most_frequent_sense.key()}")
        if(most_frequent_sense.key() in test_key[data]):
            match_count+=1
    else:
        print(f"No sense found for '{current_lemma}'.")
    
accuracy=(match_count/len(test_instances))*100
print(f"Accuracy of the most frequency sense baseline model on Test Instances is : {accuracy} %")


#   Task 2 : Implementing Lesk's Algorithm using NLTK
#   Experiment 1 : Performing analysis on base data
from nltk.wsd import lesk 
print("Task 2 : Lesk Algorithm Implementation")
def lesk_implementation(context,lemma):
    synset=lesk(context,lemma, pos='n')
    return synset.lemmas()[0].key()

def lesk_implementation_sw(context,lemma):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in context]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lemmatized_tokens if token.isalnum() and token not in stop_words]
    synset=lesk(filtered_tokens,lemma,pos='n')
    #print(synset.definition())
    return synset.lemmas()[0].key()

match_count=0
for data in dev_instances:
    #   Tokenizing Lemma and context
    current_lemma=dev_instances[data].lemma.decode('utf-8')
    current_context=[]
    for tmp in dev_instances[data].context:
        current_context.append(tmp.decode('utf-8'))
    # print(data)
    best_sense=lesk_implementation(current_context,current_lemma)
    if(best_sense in dev_key[data]):
            match_count+=1
    
accuracy=(match_count/len(dev_instances))*100
print(f"Accuracy of the Lesk algorithm model on Dev Instances is : {accuracy} %")

#   Experiment 2 : Removing Stop words
match_count=0
for data in dev_instances:
    #   Tokenizing Lemma and context
    current_lemma=dev_instances[data].lemma.decode('utf-8')
    current_context=[]
    for tmp in dev_instances[data].context:
        current_context.append(tmp.decode('utf-8'))
    # print(data)
    best_sense=lesk_implementation_sw(current_context,current_lemma)
    if(best_sense in dev_key[data]):
            match_count+=1
    
accuracy=(match_count/len(dev_instances))*100
print(f"Accuracy of the Lesk algorithm model on Dev Instances is : {accuracy} %")
