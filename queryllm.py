import logging
import re
import string
import os

from llama_cpp import Llama
from transformers import AutoTokenizer
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if CUDA is available (this will be False for Apple Silicon)
print(f"CUDA available: {torch.cuda.is_available()}")

# Check if MPS (Metal Performance Shaders) is available
print(f"MPS available: {torch.backends.mps.is_available()}")

# Get the number of available GPUs
if torch.backends.mps.is_available():
    num_gpus = 1  # Apple Silicon GPUs are treated as a single device
else:
    num_gpus = 0

print(f"Number of available GPUs: {num_gpus}")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to remove punctuation and clean text
def clean_and_tokenize(text):
    # Remove punctuation using str.translate
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into tokens and remove stop words
    tokens = set(text.lower().split())
    return {word for word in tokens if word not in ENGLISH_STOP_WORDS}


def clean_text(text):
    # Remove unwanted spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove unnecessary characters except for URLs and email addresses
    text = re.sub(r'[^a-zA-Z0-9\s@:/._-]', '', text)
    # Handle spaces around URLs and email addresses
    text = re.sub(r'(?<=http) (?=:)', '', text)  # Remove space between 'http' and ':'
    text = re.sub(r'(?<=@) (?=[a-zA-Z])', '', text)  # Remove space after '@' in emails
    text = re.sub('-', ' ', text)

    return text


def vector_by_id():
    directory = "app/vector/ontologies.index"

    # Check if the directory exists
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return None

    try:
        # Load the vector database
        vector = Chroma(persist_directory=directory,
                        embedding_function=HuggingFaceEmbeddings(model_name='paraphrase-mpnet-base-v2'))

        # Get the metadata and document count
        data = vector.get()['metadatas']
        doc_count = len(data)
        logging.info(f"Vector database loaded with {doc_count} documents.")

        # Filter unique sources
        unique_data = []
        seen = set()
        for item in data:
            identifier = item['source']
            if identifier not in seen:
                seen.add(identifier)
                unique_data.append(item)

        vector.source = unique_data
        return vector

    except Exception as e:
        logging.error(f"Failed to load vector database: {e}")
        return None


def preprocess_text(text):
    return text.lower().replace(" ", "")

def get_refined_similarity_answer(vector, query) -> str:
    query_tokens = clean_and_tokenize(query)

    # Perform vector-based similarity search with preprocessed query
    vector_docs = vector.similarity_search_with_score(preprocess_text(query))

    if not vector_docs:
        print("No documents found with the given query. {}".format(query))  # Debugging output
        return "No relevant documents found.", ""

    # Prepare final context within token limit
    context = []
    current_token_count = 0

    max_token_size = 64
    for doc,vector_score in vector_docs:
        tokens = tokenizer(doc.page_content, return_tensors='pt', truncation=False, padding=False, max_length=max_token_size)
        token_length = len(tokens['input_ids'][0])

        if current_token_count + token_length > max_token_size:
            trimmed_tokens = tokens['input_ids'][0][:max_token_size - current_token_count]
            trimmed_content = tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
            context.append(trimmed_content)
            break
        else:
            context.append(doc.page_content)
            current_token_count += token_length
    return ' '.join(context)


llmMeta = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    verbose=False,
    n_ctx=1024,  # Set context length to 1024
    n_threads=6,  # Number of CPU threads to use
    n_batch=256,  # Batch size for input processing
    use_mmap=True,  # Use memory-mapped files for model (recommended)
    use_mlock=False,  # Disable mlock to avoid locking memory
    # embedding=True,  # Enable if you want to use embedding generation
    n_gpu_layers=12,  # Set to 0 since you are running on CPU
    seed=1,  # Set seed for reproducibility (-1 for random)
    logits_all=False,  # Whether to return logits for all tokens
    n_ubatch=128,
)


# read tsv file line by line
def read_tsv_line_by_line(file_path):
    px_terms = {}
    with open(file_path, 'r') as file:
        for line in file:
            line_arrays = line.split("\t")
            if len(line_arrays) == 2:
                px_terms[line_arrays[0].strip()] = line_arrays[1].strip()
            else:
                logger.debug("Error reading line: {}".format(line_arrays[0].strip()))
    return px_terms


logger.debug("Read terms...")
terms = read_tsv_line_by_line("instrument_files.tsv")
logger.debug("All the terms has been read, total number {}".format(len(terms)))

db = vector_by_id()
for key in terms:
    res = get_refined_similarity_answer(vector=db, query=terms[key])
    messages = f"""<|im_start|>system
    Return the most relevant and exact MS ontology accession and unProcessedlabel from the provided text.
    <|im_end|>
    <|im_start|>user
    Given the following text: {res}. return only **one** MS ontology accession and unProcessedlabel relevant to {terms[key]}.
    Do not include explanation or reasons, eg:  return **only** "MS:1001542, amaZon ETD"
    <|im_end|>
"""


    meta_reply = llmMeta(
        messages,
        max_tokens=150,
        stream=False,
        stop=['[INST]', '[/INST', 'Question:','</user>','</s>'],
        temperature=0,  # Lower temperature for more focused responses
        repeat_penalty=1.2  # Increase repeat penalty to discourage repetition
    )
    result = meta_reply ['choices'][0]['text']
    accession = re.findall(r"MS:.*", result)
    logging.info("{}, {}, {}".format(key, terms[key], accession))


