import torch
text: str = open('chat.txt', encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)


stoi= {}
for i,ch in enumerate(chars): #character -> position
    stoi[ch] = i
itos= {}
for i,ch in enumerate(chars): #position -> character
    itos[i] = ch

def encode(s):
    result = []
    for ch in s:
        result.append(stoi[ch])
    return result

def split_words(text):
    words = []
    current = ""
    for ch in text:
        if ch == " "  or ch == "\n":
            if current != "":
                words.append(current)
            current = ch
        else:
            current += ch
    if current != "":
        words.append(current)
    return words

words = split_words(text)
word_counts = {}
for w in words:
    word_counts[w] = word_counts.get(w, 0) + 1

word_ids = {}
for w in word_counts:
    word_ids[w] = encode(w)
def get_stats_words(word_ids, word_counts):
    counts = {}
    for w in word_ids:
        ids = word_ids[w]
        for a, b in zip(ids,ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + word_counts[w]
    return counts

def merge(ids,pair,new_id):
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            result.append(new_id)
            i = i+2
        else:
            result.append(ids[i])
            i = i+1
    return result

num_merges = 1000
merges = {}
vocab = dict(itos)
for k in range(num_merges):
    stats = get_stats_words(word_ids, word_counts)
    top = max(stats, key=stats.get)
    new_id = vocab_size + k
    for w in word_ids:
        word_ids[w] = merge(word_ids[w], top, new_id)
    merges[top] = new_id
    vocab[new_id] = vocab[top[0]]+vocab[top[1]]
    if k % 100 == 0:
        print(k, repr(vocab[new_id]),stats[top])

ids = []
for w in words:
    for t in word_ids[w]:
        ids.append(t)
print(len(text), "letters ->", len(ids), "token, ratio", len(text) / len(ids))

def get_stats(ids):
    counts = {}
    for a,b in zip(ids,ids[1:]):
        counts[(a,b)] = counts.get((a,b),0)+1
    return counts
stats = get_stats(ids)
top = max(stats, key=stats.get)

def bpe_encode_word(s):
    ids = encode(s)
    while True:
        stats = get_stats(ids)
        best = None
        for pair in stats:
            if pair in merges:
                if best is None or merges[pair] < merges[best]:
                    best = pair
        if best is None:
            break
        ids = merge(ids,best,merges[best])
    return ids

def bpe_encode(s):
    result = []
    for w in split_words(s):
        for t in bpe_encode_word(w):
            result.append(t)
    return result

def bpe_decode(ids):
    result = ''
    for i in ids:
        result+=vocab[i]
    return result


torch.save({"merges": merges, "chars": chars,"ids": ids}, "bpe2.pt")
print("saved!")

s = "<|user|> hi, how are you doing today?\n<|bot|> I'm good, thanks!"
t = bpe_encode(s)
for i in t:
    print(repr(vocab[i]), end=" | ")
print()
print(bpe_decode(t) == s)