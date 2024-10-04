# Red the obo file and get the name, description and accession for each term.
import logging
import shutil

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
                term_info["description"] = line.split("def:")[1].strip()
        return term_info

    with open(ontology_file, "r") as file:
        content = file.read()

    terms = split_terms(content)
    ontology_name = get_ontology_name(content) if ontology_name is None else ontology_name
    terms_info = [parse_term(term, ontology_name) for term in terms]

    return terms_info

terms = read_obo_file("psi-ms.obo")
logger.info("All the terms has been read, total number {}".format(len(terms)))

# Create a vector datadase for mixtral LLM with the terms
docs = []
for term in terms:
    description = ""
    if "description" in term:
        description = " " + term["description"]
    content = "accession: {}, label: {}, description: {}".format(term["accession"], term["label"],description)
    content = content.replace("\"", "")
    new_doc = Document(
        page_content=content,
        metadata={'source': term["accession"],
                  'title': term["label"],
                  'id': term["accession"]
                  })
    docs.append(new_doc)

logger.debug("Number of Documents: %s", len(docs))
logger.debug("Creating HuggingFaceEmbeddings...")
embedding = HuggingFaceEmbeddings(model_name='paraphrase-mpnet-base-v2')
logger.debug("Embedding created")
logger.debug("Creating Chroma...")

shutil.rmtree("app", ignore_errors=True)

chroma = Chroma.from_documents(
    documents=docs,
        embedding=embedding,
        persist_directory="app/vector/ontologies.index"
)
logger.debug("Chroma created")



