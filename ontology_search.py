import logging
import re
from difflib import get_close_matches

from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_obo_file(ontology_file, ontology_name=None):
    """
    Reads an OBO file and returns a list of OlsTerms
    @:param ontology_file: The name of the ontology
    @:param ontology_name: The name of the ontology
    """

    def split_terms(content):
        terms = content.split("[Term]")[1:]  # Skip the header and split by [Term]
        return terms

    def get_ontology_name(content):
        lines = content.split("\n")
        for line in lines:
            if line.startswith("ontology:"):
                return line.split("ontology:")[1].strip()
        return None

    def parse_term(term, ontology_name):
        term_info = {}
        lines = term.strip().split("\n")
        for line in lines:
            if line.startswith("id:"):
                term_info["accession"] = line.split("id:")[1].strip()
                term_info["ontology"] = ontology_name
            elif line.startswith("name:"):
                term_info["label"] = line.split("name:")[1].strip()
            elif line.startswith("def:"):
                term_info["description"] = preprocess(line.split("def:")[1].strip())
        return term_info

    with open(ontology_file, "r") as file:
        content = file.read()

    terms = split_terms(content)
    ontology_name = get_ontology_name(content) if ontology_name is None else ontology_name
    terms_info = [parse_term(term, ontology_name) for term in terms]

    return terms_info

def read_tsv_line_by_line(file_path):
    px_terms = {}
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Skip the header
            if count == 0:
                count += 1
                continue
            line_arrays = line.split("\t")
            if len(line_arrays) == 2:
                px_terms[line_arrays[0].strip()] = line_arrays[1].strip()
            else:
                logger.debug("Error reading line: {}".format(line_arrays[0].strip()))
    return px_terms

# 2. Preprocess text
def preprocess(text):
    # Convert to lowercase and remove special characters
    return re.sub(r'[^a-z0-9\s]', '', text.lower())


# 3. Implement similarity matching
def find_best_match(free_text, ontology_terms, threshold=70):
    processed_text = preprocess(free_text)

    processed_terms = {accession: preprocess(term["label"]) for accession, term in ontology_terms.items()}

    # First try exact matching
    for accession, term in processed_terms.items():
        if processed_text == term:
            return accession, ontology_terms[accession]

    # If no exact match, use fuzzy matching
    matches = get_close_matches(processed_text, processed_terms.values(), n=1, cutoff=0.6)

    if matches:
        best_match = matches[0]
        best_accession = [acc for acc, term in processed_terms.items() if term == best_match][0]
        similarity = fuzz.ratio(processed_text, best_match)

        if similarity >= threshold:
            return best_accession, ontology_terms[best_accession]

    return None, None


onto_terms = read_obo_file("psi-ms.obo")
# transform list to dictionary for easy access, accession is the key
onto_terms = {term["accession"]: term for term in onto_terms}
logger.info("All the terms has been read, total number {}".format(len(onto_terms)))


terms = read_tsv_line_by_line("instrument_files.tsv")
logger.info("All the terms has been read, total number {}".format(len(terms)))

# File with output results
output_file = "instrument_files_results.tsv"
output = open(output_file, 'w')
output.write("project\tFree text\tAccession\tTerm\n")

for key in terms:
    accession, term = find_best_match(terms[key], onto_terms)
    if accession:
        # logger.info(f"{key}, {terms[key]}, {accession}, {term['label']}")
        output.write(f"{key}\t{terms[key]}\t{accession}\t{term['label']}\n")
    else:
        logger.info(f"{key}, {terms[key]}, No match found, No match found")
        # output.write(f"{key}\t{terms[key]}\tNo match found\tNo match found\n")



