import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to obtain the [CLS] token embedding for a given sentence
def get_cls_embedding(sentence, model, tokenizer):
    # Tokenize the input sentence and add special tokens [CLS] and [SEP]
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')
    
    # Pass the input through the BERT model
    with torch.no_grad():
        outputs = model(input_ids)
    
    # The [CLS] token's embedding is the first token's embedding
    cls_embedding = outputs.last_hidden_state[0][0]
    
    return cls_embedding

# Function to compute the dot product between two embeddings
def dot_product(embedding1, embedding2):
    norm = torch.norm(embedding1) * torch.norm(embedding2)
    return torch.dot(embedding1, embedding2).item() / norm.item()

# Examples: similar sentences and dissimilar sentences
# Examples: similar tokens and dissimilar tokens
pairs = [
    (
        'Mathematics is a field of study that discovers and organizes methods, theories and theorems that are developed and proved for the needs of empirical sciences and mathematics itself.',
        'Mathematics involves the description and manipulation of abstract objects that consist of either abstractions from nature or—in modern mathematics—purely abstract entities that are stipulated to have certain properties, called axioms.'
    ),
    (
        'Mathematics is a field of study that discovers and organizes methods, theories and theorems that are developed and proved for the needs of empirical sciences and mathematics itself.',
        "William Shakespeare (c. 23[a] April 1564 – 23 April 1616)[b] was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist."
    ),
    (
        "He is often called England's national poet and the \"Bard of Avon\" (or simply \"the Bard\"). His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship.",
        "William Shakespeare (c. 23[a] April 1564 – 23 April 1616)[b] was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist."
    )
]
sentences = set()
for s1, s2 in pairs:
    sentences.add(s1)
    sentences.add(s2)

# Obtain CLS embeddings for each sentence
cls_embeddings = {sentence: get_cls_embedding(sentence, model, tokenizer) for sentence in sentences}

# Compute dot products and print results
print("Dot Product Similarity between Sentences:")
for sentence1, sentence2 in pairs:
    similarity = dot_product(cls_embeddings[sentence1], cls_embeddings[sentence2])
    print(f"Similarity between:\n'{sentence1}'\nand\n'{sentence2}': {similarity:.4f}\n")

