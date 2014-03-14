import os
from pylearn2.utils.string_utils import preprocess
base = '${PYLEARN2_DATA_PATH}/esp_game/ESPGame100k/labels/'
base = preprocess(base)
paths = sorted(os.listdir(base))
assert len(paths)==100000

words = {}

for i, path in enumerate(paths):

    if i % 1000 == 0:
        print i
    path = base+path
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        word = line[:-1]
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

ranked_words = sorted(words.keys(), key=lambda x: -words[x])

ranked_words = [word + '\n' for word in ranked_words[0:4000]]


f = open('wordlist.txt','w')
f.writelines(ranked_words)
f.close()
