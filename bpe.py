import torch
text: str = open('input.txt', encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)
num_merges = 435 #65+435 = 500 tokens

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
ids = encode(text)
print(len(ids))

def get_stats(ids):
    counts = {}
    for a,b in zip(ids,ids[1:]):
        counts[(a,b)] = counts.get((a,b),0)+1
    return counts
stats = get_stats(ids)
top = max(stats, key=stats.get)
print(top, stats[top])
print(repr(itos[top[0]] + itos[top[1]]))

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

merges = {}
vocab = dict(itos)

def bpe_encode(s):
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

def bpe_decode(ids):
    result = ''
    for i in ids:
        result+=vocab[i]
    return result

for k in range(num_merges):
    stats = get_stats(ids)
    top = max(stats, key=stats.get)
    new_id = vocab_size + k
    ids = merge(ids,top,new_id)
    merges[top] = new_id
    vocab[new_id] = vocab[top[0]] + vocab[top[1]]
    if k % 25 == 0:
        print(k, repr(vocab[new_id]),stats[top])
print("letters:", len(text))
print("tokens: ", len(ids))
print("ratio:  ", len(text) / len(ids))

for i in range(vocab_size + num_merges - 15, vocab_size + num_merges):
    print(i, repr(vocab[i]))

s = "ROMEO:\nBut soft, what light through yonder window breaks?"
t = bpe_encode(s)
print(len(s), "letters →", len(t), "tokens")
for i in t:
    print(repr(vocab[i]), end=" | ")
print()
print(bpe_decode(t) == s)

torch.save({"merges": merges, "ids": ids}, "bpe.pt")
loaded = torch.load("bpe.pt")
print(len(loaded["merges"]), "merges,", len(loaded["ids"]), "tokens")