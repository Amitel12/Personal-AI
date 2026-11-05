from datasets import load_dataset

allowed = set(open("input.txt",encoding = "utf-8").read())

ds = load_dataset("allenai/soda", split = "train", streaming = True) #stream -> read bit by bit without a huge download

out = open("chat.txt","w",encoding = "utf-8")
count = 0
for row in ds:
    lines = row["dialogue"]
    names = set(row["speakers"])
    text = ""
    for i in range(len(lines)):
        if i % 2 == 0: # even turns = the user
            tag = "<|user|> "
        else: #odd turns = the bot
            tag = "<|bot|> "
        line = lines[i]
        for name in names:
            line = line.replace(name, "buddy")
        text += tag + line + "\n"

    text += "<|endoftext|>\n"

    clean = ""
    for ch in text:
        if ch in allowed:
            clean += ch
    out.write(clean)

    count += 1
    if count == 100000:
        break
out.close()
print(count, "conversations  saved")



