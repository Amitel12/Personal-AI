# Personal AI

A small chatbot built from scratch in PyTorch: a GPT-style Transformer and a BPE tokenizer, trained on 100,000 conversations from Hugging Face.

I built this to understand how language models work. Claude guided me through the concepts and reviewed my code, and I wrote the code myself.

## What's inside

| | |
|---|---|
| Model | Transformer, about 26M parameters (8 layers, 8 attention heads, 512 embedding size, 256-token context) |
| Tokenizer | Word-level BPE with 1,000 merges and a vocabulary of 1,083 tokens |
| Data | 100,000 conversations from [SODA](https://huggingface.co/datasets/allenai/soda), about 18M tokens |
| Hardware | Trained on a single RTX 3070 (8 GB) |

## Files

| File | What it does |
|---|---|
| `make_chat.py` | Builds `chat.txt` from SODA and replaces speaker names with the placeholder "buddy" |
| `bpe2.py` | Trains the word-level tokenizer on `chat.txt` and saves `bpe2.pt` |
| `make_identity.py` | Generates conversations that teach the bot who it is and who made it |
| `chat2.py` | Trains or resumes the chatbot, then opens a chat in the terminal |

Earlier experiments are also here: `bpe.py` (character-level BPE), `main.py` (story model), `chat.py` (fine-tuning the story model on chat data) and `cut.py` (prepares `input.txt` from TinyStories).

## Running it

Models and datasets aren't in this repo because of their size, so you need to build them first.

1. Install the dependencies:
   ```
   pip install torch datasets
   ```
2. Create `input.txt`. `make_chat.py` only keeps characters that appear in this file. I made mine with `cut.py`, from the first 5M characters of TinyStories.
3. Run the scripts in this order:
   ```
   python make_chat.py
   python bpe2.py
   python make_identity.py
   python chat2.py
   ```

The settings in `chat2.py` are the ones I used to continue training an existing model (`lr = 1e-4`, `max_steps = 5000`). For a new model, use `lr = 3e-4` and `max_steps = 20000`. Set `train_more = False` to skip training and go straight to the chat.

## What I learned

- **More data reduces overfitting.** Going from 30,000 to 100,000 conversations shrank the gap between train and validation loss from about 0.73 to 0.22, and validation loss kept improving through the whole 20,000-step run (best: 1.664).
- **A better tokenizer lets the model see more.** BPE fits about 3.9 characters into each token, so the 256-token context covers roughly 1,000 characters instead of 256.
- **Word-level BPE makes large datasets practical.** Counting each unique word once, weighted by how often it appears, is what made it realistic to tokenize 100,000 conversations.
- **A model can only do what's in its data.** SODA is made of acted-out scenes between two people, so the bot learned to act out scenes rather than behave like an assistant.

## Limitations

The bot writes fluent small talk, but it often loses the topic and has no reliable knowledge. It also doesn't truly remember things: "buddy" is swapped for your name with a plain string replacement, and a model this size struggles to use information from earlier in the chat.
