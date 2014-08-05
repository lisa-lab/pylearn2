"""
Designed to calculate certain statistics on the google question when vocab is clipped
"""
import cPickle
import numpy as np

#params:
vocab_size = 20000
questions_file = "/data/lisatmp3/pougetj/questions-words.txt"
vocab_file = "/data/lisatmp3/pougetj/vocab.pkl"
UNK = 1

#Load the vocabulary and binarize the questions
with open(vocab_file) as f:
	vocab = cPickle.load(f)

categories = []
binarized_questions = []
with open(questions_file) as f:
    for i, line in enumerate(f):
        words = line.strip().lower().split()
        if words[0] == ":":
		categories.append((i, words[1]))
		continue
	binarized_questions.append([vocab.get(word, UNK)
                                            for word in words])
print("done loading vocab and binarizing questions")

questions = np.asarray(binarized_questions, dtype="int32")
questions[questions >= vocab_size] = UNK
n_questions = len(questions)

#Count number of questions with UNK words
index = [UNK not in s for s in questions]
total_filtered_questions = questions[np.asarray(index)]
n_total_filtered_questions = len(total_filtered_questions)

#Count number of questions with UNK in the target
target_filtered_questions = questions[questions[:,3] != UNK]
n_target_filtered_questions = len(target_filtered_questions)

print "n_questions", n_questions
print "n_target_filtered_questions", n_target_filtered_questions
print "n_total_filtered_questions", n_total_filtered_questions
