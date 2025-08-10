import re
import csv
import os

input_dir = "_novels/marple" 

marple_keywords = [
    "marple", 
    "miss marple",
    "ms. marple"
    "ms marple"
    "jane marple"
]

speak_verbs = [
    "says", "said", 
    "replies", "replied", 
    "asks", "asked", 
    "cries", "cried",
    "answers", "answered", 
    "remarks", "remarked", 
    "observes", "observed", 
    "shouts", "shouted",
    "comments", "commented"
]

marple_lines  = []

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".txt"):
        filepath = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        dialogue_pattern = r'["“](.*?)["”]'
        matches = list(re.finditer(dialogue_pattern, text, re.DOTALL))

        for match in matches:
            quote = match.group(1).strip()
            start, end = match.span()

            context_before = text[max(0, start-80):start].lower()
            context_after = text[end:end+80].lower()

            if (any(name in context_before for name in marple_keywords) and
                any(verb in context_before for verb in speak_verbs)):
                marple_lines.append(quote)
            elif (any(name in context_after for name in marple_keywords) and
                  any(verb in context_after for verb in speak_verbs)):
                marple_lines.append(quote)

# saved to csv
with open("lines/marple_lines_draft.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["number", "quote"])
    for i, line in enumerate(marple_lines, 1):
        writer.writerow([i, line])

print(f"Extraction completed, found {len(marple_lines)} lines in total")