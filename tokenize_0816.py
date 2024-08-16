import torch
from transformers import AutoTokenizer

# T5-samll tokenizer
checkpoint = "google-t5/t5-small" # name of the checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # load the tokenizer corresponding to that checkpoint

sentence1 = "Hello, how are you doing today?"
sentence2 = "We present BLOOMZ & mT0, a family of models capable of following human instructions in dozens of languages zero-shot. We finetune BLOOM & mT5 pretrained multilingual language models on our crosslingual task mixture (xP3) and find the resulting models capable of crosslingual generalization to unseen tasks & languages."
sentence3 = "华夏是人类的文明摇篮之一。 新石器时期，中原地区开始出现聚落组织"
sentence4 = "Helloween"

tokenized_output = tokenizer(sentence4) # {'input_ids': [8774, 6, 149, 33, 25, 692, 469, 58, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
input_ids1 = tokenized_output["input_ids"]
tokens1 = tokenizer.convert_ids_to_tokens(input_ids1) # ['▁Hello', ',', '▁how', '▁are', '▁you', '▁doing', '▁today', '?', '</s>']
# '</s>' is the end of sentence token

tokenizer2 = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1-mt")
tokenized_output2 = tokenizer2(sentence4)
input_ids2 = tokenized_output2["input_ids"]
tokens2 = tokenizer2.convert_ids_to_tokens(input_ids2)

print(f"sentence: {sentence3}")
print(f"T5-small input_ids: {input_ids1}")
print(f"Bloomz-mt-7b1 input_ids: {input_ids2}")
print(f"T5-small Tokens: {tokens1}")
print(f"Bloomz-mt-7b1 Tokens: {tokens2}")
breakpoint()

# Takeaway: LLMs think in tokens, not in words. Words are composed of tokens but are not tokens.
# Not all word/languages have valid tokens mapping in the tokenizer's vocabulary. E.g., for T5-small, there are no mappings for Chinese

pass