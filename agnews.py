from datasets import load_dataset
dataset = load_dataset("ag_news", split="train[:10]")

print("Dataset features:", dataset.features)

top10_sentences = []
for item in dataset:
    # Try different access methods based on the dataset structure
    try:
        # Standard dictionary access
        top10_sentences.append(item['text'])
    except TypeError:
        # If items are returned as plain strings
        top10_sentences.append(str(item))

print("\nTop 10 sentences from AG News dataset:")
for i, sentence in enumerate(top10_sentences, 1):
    print(f"{i}. {sentence}")