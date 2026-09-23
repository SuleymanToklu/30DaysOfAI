from transformers import AutoTokenizer, AutoModel
import torch
from functools import lru_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@lru_cache(maxsize=4)
def get_model_and_tokenizer(model_name):
    """Loads a model and tokenizer from Hugging Face, caching the result for performance."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def analyze_sentence(model_name, sentence):
    """
    Analyzes a sentence using a specified transformer model and extracts internal states.
    """
    if not sentence or not model_name:
        return None

    model, tokenizer = get_model_and_tokenizer(model_name)
    
    # Convert sentence to numerical IDs for the model.
    inputs = tokenizer(sentence, return_tensors="pt")

    # Disable gradient calculations for inference mode to save memory.
    with torch.no_grad():
        # Request attentions and hidden states from the model.
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Extract the tuples of tensors for each layer.
    attention_weights = outputs.attentions
    hidden_states = outputs.hidden_states

    analysis_data = {
        "tokens": tokens,
        "attention": attention_weights,
        "hidden_states": hidden_states
    }
    return analysis_data

def find_closest_words(embeddings, tokens):
    """
    Finds the two most semantically similar words in a sentence based on their embeddings.
    """
    # Ignore special tokens and punctuation for a cleaner analysis.
    ignore_list = ["[CLS]", "[SEP]", ".", ",", "?", "!"]
    
    valid_indices = [i for i, token in enumerate(tokens) if token not in ignore_list]
    if len(valid_indices) < 2:
        return "Anlamsal yakınlık analizi için yeterli kelime bulunamadı."

    valid_embeddings = embeddings[valid_indices]
    valid_tokens = [tokens[i] for i in valid_indices]
    
    # Calculate the cosine similarity matrix between all valid words.
    similarity_matrix = cosine_similarity(valid_embeddings)
    
    # Fill the diagonal with a low value to ignore self-similarity.
    np.fill_diagonal(similarity_matrix, -1)
    
    # Find the index of the highest similarity score.
    max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    
    word1 = valid_tokens[max_idx[0]]
    word2 = valid_tokens[max_idx[1]]
    similarity_score = similarity_matrix[max_idx]
    
    return f"💡 **Dinamik Analiz:** Model, bu cümlede anlamsal olarak birbirine en yakın iki kelimeyi **'{word1}'** ve **'{word2}'** olarak belirledi (Benzerlik Skoru: {similarity_score:.2f})."