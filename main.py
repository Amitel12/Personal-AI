import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256 #how many chars per chunk
batch_size = 64 #how many chunks per training step
n_embd = 384 # how many numbers describe each letter
n_head = 6 #attention heads per block
n_layer = 6 # how many blocks stacked on top of each other
dropout = 0.2 #randomly switches off 20% of the numbers while training
lr=1e-3
max_steps = 5000
eval_every = 500
text: str = open('input.txt', encoding='utf-8').read()
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
def decode(ids):
    result = ''
    for i in ids:
        result+=itos[i]
    return result

data = torch.tensor(encode(text)) #all encoded text into a tensor
##Splitting the Data to train/valid/test
n1 = int(0.8*len(data))
n2=int(0.9*len(data))
train_data = data[:n1] #0% ~ 80%
valid_data=data[n1:n2] #80% ~ 90%
test_data = data[n2:] #90% ~ 100%

def get_batch(split):
    if split == 'train':
        d = train_data
    else:
        d = valid_data
    ix = torch.randint(len(d) - block_size,(batch_size,))

    xs = []
    ys = []
    for i in ix:
        xs.append(d[i:i+block_size])
        ys.append(d[i+1:i+block_size+1])
    x=torch.stack(xs).to(device)
    y=torch.stack(ys).to(device)
    return x,y

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias = False)
        self.query = nn.Linear(n_embd,head_size,bias = False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) #store inside the model | not trainable!
        self.drop = nn.Dropout(dropout)  # stores it on the model
    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x) #what the letter contains?
        q = self.query(x) #what part of the word i want to know
        v = self.value(x) #what information it gives
        #1.Scores: every query compares with every key (B, T, T)
        att = q @ k.transpose(-1,-2)
        att = att*(k.shape[-1] ** -0.5) # scales down to avoid big number at softmax
        #2. no looking ahead
        att = att.masked_fill(self.tril[:T,:T]==0,float('-inf')) # swaps 0 with -infinity so the model ignores them
        #3.softmax-ing scores to precentages
        att = F.softmax(att,dim=-1)
        att = self.drop(att)  # randomly zero some attention weights
        #4.mixing the values using the precentages
        out = att @ v
        return out

class MultiHead(nn.Module):
    def __init__(self,n_head,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self,x):
        outs = []
        for h in self.heads:
            outs.append(h(x))
        out = torch.cat(outs,dim=-1)
        return self.drop(self.proj(out))

class FeedForward(nn.Module): #neural network to proccess the info for each letter
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( #chains the layers together so x goes through a then with a,b..
            nn.Linear(n_embd,4 * n_embd), #expands 32 numbers that represent dimensions to 128
            nn.GELU(),#bend so the model leans to more complex patterns
            nn.Linear(4 * n_embd, n_embd), #turns 128 back to 32 numbers that represent
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead(n_head,n_embd // n_head)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x)) #run attention on the noramlized version
        x = x + self.ff(self.ln2(x)) #normalize and feed forward then add back to the original
        return x


##The Model class
class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size,n_embd) #what the letter is and gives 32 numbers that describe the letter
        self.pos_emb = nn.Embedding(block_size,n_embd) #Where is the letter (0 to block_size-1) and gives 32 numbers that describe the letter
        self.head = nn.Linear(n_embd,vocab_size) #turns 32 numbers to 65 scores using trainable weights
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)]) #makes a list of 4 seperate blocks with weights for each one and chains them to blocks
        self.in_f = nn.LayerNorm(n_embd)


    def forward(self,idx,targets=None):
        B,T = idx.shape # B = how many chunks, T = how many letters per chunk
        tok = self.tok_emb(idx) # what each letter is (B,T,32)
        pos = self.pos_emb(torch.arange(T,device=idx.device)) #where each letter is (T,32)
        x = tok+pos #combines them (B,T,32)
        x = self.blocks(x)
        x = self.in_f(x)
        logits = self.head(x) #turns scores for the next character

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1,vocab_size),targets.view(-1)) #compare the scores to the real answers
        return logits,loss
    def generate(self,idx,n_new): #loop that predicts the next letter and add it to the text
        for _ in range(n_new):
            logits,loss = self(idx[:,-block_size:])
            last = logits[:,-1,:] #keeps only the scores for the last position
            probs = F.softmax(last,dim=-1) # turns scores to probability
            next_id = torch.multinomial(probs,1) #picks 1 random letter using those probabilities
            idx = torch.cat([idx,next_id],dim=1)  #adds the chosen letter to the end of the text
        return idx

model = Bigram()
model = model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "million parameters")

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'valid']:
        losses = []
        for _ in range(50):
            x,y = get_batch(split)
            logits,loss = model(x,y)
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses)
    model.train()
    return out



#training loop
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

for step in range(max_steps):
    x,y = get_batch('train')
    logits,loss = model(x,y)
    optimizer.zero_grad() #forgets the last corrections
    loss.backward() #checks for each number which gets less on the loss function
    optimizer.step() #move every number torward the right way that gets less on the loss function
    if step % eval_every == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.3f}, valid {losses['valid']:.3f}")

start = torch.zeros((1,1),dtype = torch.long,device = device) #one starting character
out = model.generate(start,300)
print(decode(out[0].tolist()))