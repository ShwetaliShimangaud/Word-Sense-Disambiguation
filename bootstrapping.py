import re

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

import loader

dev_instances, test_instances, dev_key, test_key = loader.main()

# Unique lemma in dev instances
unique_lemma = []

for data in dev_instances:
    current_lemma = dev_instances[data].lemma.decode('utf-8')
    if not unique_lemma.__contains__(current_lemma):
        unique_lemma.append(current_lemma)

# unique lemmas in test which are present in dev as well.
unique_lemma_test = []

for data in test_instances:
    current_lemma = test_instances[data].lemma.decode('utf-8')
    if unique_lemma.__contains__(current_lemma):
        # print("lemma exists")
        if not unique_lemma_test.__contains__(current_lemma):
            unique_lemma_test.append(current_lemma)

# Randomly select 5 elements from the list
words_to_disambiguate = ['pressure', 'group', 'burden', 'game', 'period']

# Pressure
#     Sense              Label     sense key
#     Influence            0        1.07.00
#     Measurement unit     1        1.19.00
#     physical force       2        1.04.00

pressure_mapping = {
    'pressure%1:07:00::': 0,
    'pressure%1:19:00::': 1,
    'pressure%1:04:00::': 2

}

# Group
#     Sense              Label
#     Group as unit        0
#     Chemical group       1
#     mathematical group   2
#     verb                 3

group_mapping = {
    'group%1:03:00::': 0,
    'group%1:27:00::': 1,
    'group%1:09:00::': 2,
    'group%1:31:00::': 3
}


# Burden
#     Sense              Label
#     difficult concern    0
#     weight to be borne   1
#     literary work        2

burden_mapping = {
    'burden%1:09:01::': 0,
    'burden%1:06:01::': 1,
    'burden%1:10:00::': 2,
}

# Game
#     Sense              Label
#     contest with rules   0
#     single play of sport 1
#     a secret scheme      2

game_mapping = {
    'game%1:04:00::': 0,
    'game%1:04:03::': 1,
    'game%1:09:00::': 2,
}

# Period
#     Sense              Label
#     an amount of time    0
#     interval taken       1
#     geological time      2

period_mapping = {
    'period%1:28:00::': 0,
    'period%1:28:02::': 1,
    'period%1:28:03::': 2,
}

# convert to lowercase, strip and remove punctuations
def lowercase_punctuation(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"[^\w\s]|_", "", text)
    return text


# Remove stop words
def remove_stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# lemmatization
def lemmatize(string):
    lemmatizer = WordNetLemmatizer()
    a = lemmatizer.lemmatize(string)
    return a


def preprocess_with_lemmatization(string):
    return lemmatize(remove_stopword(lowercase_punctuation(string)))


def preprocess(X):
    temp = []
    for w in X:
        processed = preprocess_with_lemmatization(w)
        if not len(processed) == 0:
            temp.append(processed)

    return temp


class word_sense_disambiguation:
    def __init__(self, iterations):
        self.iterations = iterations

    def fit(self, X, y):
        X_train, X_seed, y_train, y_seed = train_test_split(
            X, y, test_size=0.15, random_state=99)

        # Apply Naive Bayes, as dataset is small
        nb_pipeline = Pipeline([('vectorizer', TfidfVectorizer(use_idf=True)), ('clf', MultinomialNB())])

        count = 0
        while count < self.iterations:
            nb_pipeline.fit(X_seed, y_seed)
            y_pred = nb_pipeline.predict_proba(X_train)
            # print(y_pred)

            max_prob = np.amax(y_pred, axis=1)
            # print(max_prob)

            max_indices = np.argsort(max_prob)[-60:]

            X_seed = np.asarray(X_train)[max_indices]
            y = y_pred[max_indices]
            y_seed = np.argmax(y, axis=1)
            count = count + 1

        # print(max_prob)
        return nb_pipeline

    def predict(self, clf, X):
        return clf.predict(X)

    def evaluate_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


## WORD = PRESSURE
# WSD for word pressure
curr_dir = os.getcwd()
influence = open(f'{curr_dir}\\Pressure\\influence.txt', 'r')
influence_data = np.unique(np.asarray(influence.readlines()))
influence_data = preprocess(influence_data)

measurement_unit = open(f'{curr_dir}\\Pressure\\measurement_unit.txt', 'r')
measurement_unit_data = np.unique(np.asarray(measurement_unit.readlines()))
measurement_unit_data = preprocess(measurement_unit_data)

physical_force = open(f'{curr_dir}\\Pressure\\physical_force_exertion.txt', 'r')
physical_force_data = np.unique(np.asarray(physical_force.readlines()))
physical_force_data = preprocess(physical_force_data)

y = np.concatenate((np.concatenate((np.full(len(influence_data), 0), np.full(len(measurement_unit_data), 1))),
                    np.full(len(physical_force_data), 2)))

influence_data.extend(measurement_unit_data)
influence_data.extend(physical_force_data)

X_preprocessed = influence_data

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.15, random_state=99)

accuracy_for_pressure = []
for i in range(3):
    clf = word_sense_disambiguation(i + 1)
    model = clf.fit(X_train, y_train)
    y_pred = clf.predict(model, X_test)
    acc_on_chagpt_test = clf.evaluate_accuracy(y_test, y_pred)
    accuracy_for_pressure.append(acc_on_chagpt_test * 100)
    print('With Itrations:', (i + 1), 'Accuracy for word Pressure:', acc_on_chagpt_test)


def extract_data(lemma, values, keys, mapping):
    extracted_dataset = []
    for data in values:
        current_lemma = values[data].lemma.decode('utf-8')
        if current_lemma == lemma:
            current_context = []
            for tmp in values[data].context:
                current_context.append(tmp.decode('utf-8'))
            # print(data)
            best_sense = keys[data][0]
            pred = mapping.get(best_sense)
            if not pred == None:
                context = ' '.join(current_context)
                extracted_dataset.extend([context, pred])

    return extracted_dataset


pressure_dev = extract_data("pressure", dev_instances, dev_key, pressure_mapping)
pressure_test = extract_data("pressure", test_instances, test_key, pressure_mapping)

dev_x = [value for index, value in enumerate(pressure_dev) if index % 2 == 0]
dev_y = [value for index, value in enumerate(pressure_dev) if index % 2 != 0]

test_x = [value for index, value in enumerate(pressure_test) if index % 2 == 0]
test_y = [value for index, value in enumerate(pressure_test) if index % 2 != 0]

clf = word_sense_disambiguation(1)
model = clf.fit(X_train, y_train)
y_pred_dev = clf.predict(model, dev_x)
accuracy_on_dev_instances = clf.evaluate_accuracy(dev_y, y_pred_dev)

print("Accuracy on dev instances for word Pressure", accuracy_on_dev_instances)

y_pred_test = clf.predict(model, test_x)
accuracy_on_test_instances = clf.evaluate_accuracy(test_y, y_pred_test)
print("Accuracy on test instances for word Pressure", accuracy_on_test_instances)

# WSD for word Group
chemical = open(f'{curr_dir}\\Group\\chemical group.txt', 'r')
chemical_data = np.unique(np.asarray(chemical.readlines()))
chemical_data = preprocess(chemical_data)

group_as_unit = open(f'{curr_dir}\\Group\\group as unit.txt', 'r')
group_as_unit_data = np.unique(np.asarray(group_as_unit.readlines()))
group_as_unit_data = preprocess(group_as_unit_data)

group_as_verb = open(f'{curr_dir}\\Group\\group as verb.txt', 'r')
group_as_verb_data = np.unique(np.asarray(group_as_verb.readlines()))
group_as_verb_data = preprocess(group_as_verb_data)

mathematical_group = open(f'{curr_dir}\\Group\\mathematical group.txt', 'r')
mathematical_group_data = np.unique(np.asarray(mathematical_group.readlines()))
mathematical_group_data = preprocess(mathematical_group_data)

y = np.concatenate((np.concatenate((np.concatenate((
    np.full(len(chemical_data), 1),
    np.full(len(group_as_verb_data), 3))),
                                    np.full(len(group_as_unit_data), 0))),
                    np.full(len(mathematical_group_data), 2)))

X_preprocessed = []
X_preprocessed.extend(chemical_data)
X_preprocessed.extend(group_as_verb_data)
X_preprocessed.extend(group_as_unit_data)
X_preprocessed.extend(mathematical_group_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.15, random_state=99)

accuracy_for_group = []
for i in range(3):
    clf = word_sense_disambiguation(i + 1)
    model = clf.fit(X_train, y_train)
    y_pred = clf.predict(model, X_test)
    acc_on_chagpt_test = clf.evaluate_accuracy(y_test, y_pred)
    accuracy_for_group.append(acc_on_chagpt_test * 100)
    print('With Itrations:', (i + 1), 'Accuracy for word Group:', acc_on_chagpt_test)

group_dev = extract_data("group", dev_instances, dev_key, group_mapping)
group_test = extract_data("group", test_instances, test_key, group_mapping)

dev_x = [value for index, value in enumerate(group_dev) if index % 2 == 0]
dev_y = [value for index, value in enumerate(group_dev) if index % 2 != 0]

test_x = [value for index, value in enumerate(group_test) if index % 2 == 0]
test_y = [value for index, value in enumerate(group_test) if index % 2 != 0]

clf = word_sense_disambiguation(1)
model = clf.fit(X_train, y_train)
y_pred_dev = clf.predict(model, dev_x)
accuracy_on_dev_instances = clf.evaluate_accuracy(dev_y, y_pred_dev)

print("Accuracy on dev instances for word Group", accuracy_on_dev_instances)

y_pred_test = clf.predict(model, test_x)
accuracy_on_test_instances = clf.evaluate_accuracy(test_y, y_pred_test)
print("Accuracy on test instances for word Group", accuracy_on_test_instances)

# WSD for word Burden
burden = open(f'{curr_dir}\\Burden\\burden.n.01.txt', 'r')
burden_data = np.unique(np.asarray(burden.readlines()))
burden_data = preprocess(burden_data)

effect = open(f'{curr_dir}\\Burden\\effect.n.04.txt', 'r')
effect_data = np.unique(np.asarray(effect.readlines()))
effect_data = preprocess(effect_data)

load = open(f'{curr_dir}\\Burden\\load.n.01.txt', 'r')
load_data = np.unique(np.asarray(load.readlines()))
load_data = preprocess(load_data)

y = np.concatenate((np.concatenate((
    np.full(len(burden_data), 0),
    np.full(len(load_data), 1))),
                    np.full(len(effect_data), 2)))

X_preprocessed = []
X_preprocessed.extend(burden_data)
X_preprocessed.extend(load_data)
X_preprocessed.extend(effect_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.15, random_state=99)

accuracy_for_burden = []
for i in range(3):
    clf = word_sense_disambiguation(i + 1)
    model = clf.fit(X_train, y_train)
    y_pred = clf.predict(model, X_test)
    acc_on_chagpt_test = clf.evaluate_accuracy(y_test, y_pred)
    accuracy_for_burden.append(acc_on_chagpt_test * 100)
    print('With Itrations:', (i + 1), 'Accuracy for word Burden:', acc_on_chagpt_test)

burden_dev = extract_data("burden", dev_instances, dev_key, burden_mapping)
burden_test = extract_data("burden", test_instances, test_key, burden_mapping)

dev_x = [value for index, value in enumerate(burden_dev) if index % 2 == 0]
dev_y = [value for index, value in enumerate(burden_dev) if index % 2 != 0]

test_x = [value for index, value in enumerate(burden_test) if index % 2 == 0]
test_y = [value for index, value in enumerate(burden_test) if index % 2 != 0]

clf = word_sense_disambiguation(1)
model = clf.fit(X_train, y_train)
y_pred_dev = clf.predict(model, dev_x)
accuracy_on_dev_instances = clf.evaluate_accuracy(dev_y, y_pred_dev)

print("Accuracy on dev instances for word Burden", accuracy_on_dev_instances)

y_pred_test = clf.predict(model, test_x)
accuracy_on_test_instances = clf.evaluate_accuracy(test_y, y_pred_test)
print("Accuracy on test instances for word Burden", accuracy_on_test_instances)

# WSD for word Game
game = open(f'{curr_dir}\\Game\\game.n.01.txt', 'r')
game_data = np.unique(np.asarray(game.readlines()))
game_data = preprocess(game_data)

single_play = open(f'{curr_dir}\\Game\\game.n.02.txt', 'r')
single_play_data = np.unique(np.asarray(single_play.readlines()))
single_play_data = preprocess(single_play_data)

plot = open(f'{curr_dir}\\Game\\plot.n.01.txt', 'r')
plot_data = np.unique(np.asarray(load.readlines()))
plot_data = preprocess(load_data)

y = np.concatenate((np.concatenate((
    np.full(len(game_data), 0),
    np.full(len(single_play_data), 1))),
                    np.full(len(plot_data), 2)))

X_preprocessed = []
X_preprocessed.extend(game_data)
X_preprocessed.extend(single_play_data)
X_preprocessed.extend(plot_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.15, random_state=99)

accuracy_for_game = []
for i in range(3):
    clf = word_sense_disambiguation(i + 1)
    model = clf.fit(X_train, y_train)
    y_pred = clf.predict(model, X_test)
    acc_on_chagpt_test = clf.evaluate_accuracy(y_test, y_pred)
    accuracy_for_game.append(acc_on_chagpt_test * 100)
    print('With Itrations:', (i + 1), 'Accuracy for word Game:', acc_on_chagpt_test)

game_dev = extract_data("game", dev_instances, dev_key, game_mapping)
game_test = extract_data("game", test_instances, test_key, game_mapping)

dev_x = [value for index, value in enumerate(game_dev) if index % 2 == 0]
dev_y = [value for index, value in enumerate(game_dev) if index % 2 != 0]

test_x = [value for index, value in enumerate(game_test) if index % 2 == 0]
test_y = [value for index, value in enumerate(game_test) if index % 2 != 0]

clf = word_sense_disambiguation(1)
model = clf.fit(X_train, y_train)
y_pred_dev = clf.predict(model, dev_x)
accuracy_on_dev_instances = clf.evaluate_accuracy(dev_y, y_pred_dev)

print("Accuracy on dev instances for word Game", accuracy_on_dev_instances)

y_pred_test = clf.predict(model, test_x)
accuracy_on_test_instances = clf.evaluate_accuracy(test_y, y_pred_test)
print("Accuracy on test instances for word Game", accuracy_on_test_instances)

# WSD for word Period
time_period = open(f'{curr_dir}\\Period\\time_period.n.01.txt', 'r')
time_period_data = np.unique(np.asarray(time_period.readlines()))
time_period_data = preprocess(time_period_data)

period_interval = open(f'{curr_dir}\\Period\\period.n.02.txt', 'r')
period_interval_data = np.unique(np.asarray(period_interval.readlines()))
period_interval_data = preprocess(period_interval_data)

geological_time = open(f'{curr_dir}\\Period\\period.n.03.txt', 'r')
geological_time_data = np.unique(np.asarray(geological_time.readlines()))
geological_time_data = preprocess(geological_time_data)

y = np.concatenate((np.concatenate((
    np.full(len(time_period_data), 0),
    np.full(len(period_interval_data), 1))),
                    np.full(len(geological_time_data), 2)))

X_preprocessed = []
X_preprocessed.extend(time_period_data)
X_preprocessed.extend(period_interval_data)
X_preprocessed.extend(geological_time_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.15, random_state=99)

accuracy_for_period = []
for i in range(3):
    clf = word_sense_disambiguation(i + 1)
    model = clf.fit(X_train, y_train)
    y_pred = clf.predict(model, X_test)
    acc_on_chagpt_test = clf.evaluate_accuracy(y_test, y_pred)
    accuracy_for_period.append(acc_on_chagpt_test * 100)
    print('With Itrations:', (i + 1), 'Accuracy for word Period:', acc_on_chagpt_test)

period_dev = extract_data("period", dev_instances, dev_key, period_mapping)
period_test = extract_data("period", test_instances, test_key, period_mapping)

dev_x = [value for index, value in enumerate(period_dev) if index % 2 == 0]
dev_y = [value for index, value in enumerate(period_dev) if index % 2 != 0]

test_x = [value for index, value in enumerate(period_test) if index % 2 == 0]
test_y = [value for index, value in enumerate(period_test) if index % 2 != 0]

clf = word_sense_disambiguation(1)
model = clf.fit(X_train, y_train)
y_pred_dev = clf.predict(model, dev_x)
accuracy_on_dev_instances = clf.evaluate_accuracy(dev_y, y_pred_dev)

print("Accuracy on dev instances for word Period", accuracy_on_dev_instances)

y_pred_test = clf.predict(model, test_x)
accuracy_on_test_instances = clf.evaluate_accuracy(test_y, y_pred_test)
print("Accuracy on test instances for word Period", accuracy_on_test_instances)

xx = np.array([1, 2, 3])
# Plotting the curves
plt.figure(figsize=(8, 6))  # Setting the figure size
plt.plot(xx, accuracy_for_pressure, label='Pressure')
plt.plot(xx, accuracy_for_group, label='Group')
plt.plot(xx, accuracy_for_game, label='Game')
plt.plot(xx, accuracy_for_burden, label='Burden')
plt.plot(xx, accuracy_for_period, label='Period')

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
