import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to obtain the word embedding for a given token
def get_word_embedding(token, model, tokenizer):
    # Tokenize the input token
    input_ids = tokenizer.encode(token, add_special_tokens=False, return_tensors='pt')
    
    # Pass the input through the BERT model
    with torch.no_grad():
        outputs = model(input_ids)
    
    # The last hidden state contains the embeddings
    # Since we're only interested in the token's embedding, we take the first (and only) token's embedding
    token_embedding = outputs.last_hidden_state[0][0]
    
    return token_embedding

# Function to compute the dot product between two embeddings
def dot_product(embedding1, embedding2):
    norm = torch.norm(embedding1) * torch.norm(embedding2)
    return torch.dot(embedding1, embedding2).item() / norm.item()

# Examples: similar tokens and dissimilar tokens
pairs = [
    ('king', 'man'),
    ('king', 'woman'),
    ('king', 'laptop'),
    ('computer', 'laptop'),
    ('US', 'rich'),
    ('US', 'poor'),
    ('China', 'rich'),
    ('China', 'poor'),
    ('India', 'rich'),
    ('India', 'poor'),
]
tokens = set()
for token1, token2 in pairs:
    tokens.add(token1)
    tokens.add(token2)

# Obtain embeddings for each token
embeddings = {token: get_word_embedding(token, model, tokenizer) for token in tokens}

# Compute dot products and print results
for token1, token2 in pairs:
    similarity = dot_product(embeddings[token1], embeddings[token2])
    print(f"Similarity between '{token1}' and '{token2}': {similarity:.2f}")

# What are the limitations?

