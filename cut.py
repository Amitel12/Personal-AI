#cuts the tinystories to 5MB and paste it to input.txt
text = open("tinystories.txt", encoding="utf-8").read()
part = text[:5_000_000]                 # first 5 million characters
open("input.txt", "w", encoding="utf-8").write(part)
print(len(part), "characters saved")
