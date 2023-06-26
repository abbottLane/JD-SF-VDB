import pinecone
import dotenv
import transformers
import os
import torch
import logging

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():

    # Init Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    # List indexes
    active_indexes = pinecone.list_indexes()
    logging.info(f"Active indexes: {active_indexes}")

    # Describe indexes
    index_description = pinecone.describe_index("salesforce")
    logging.info(f"Index description: {index_description}")

    # load docs to index from Salesforce.md
    docs = load_docs("Salesforce.md")

    # Embed the document with the best document embedding model ever created
    model = transformers.AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create an index
    index_name = "salesforce"
    if index_name not in active_indexes:
        pinecone.create_index(name=index_name, metric="cosine", shards=1)

    # Insert the docs into the index
    index = pinecone.Index(index_name=index_name)
    for i, doc in enumerate(docs):
        embedding = embed_doc(doc[:512], model, tokenizer, device)
        vector_docs = [{'id':str(i), 'values': [float(x) for x in embedding.cpu().numpy().tolist()[0]]}]
    upsert_response = index.upsert(vectors=vector_docs,namespace=index_name)
    logging.info(f"Upsert response: {upsert_response}")

def embed_doc(doc, model, tokenizer, device):
    # Tokenize the document
    inputs = tokenizer(doc, return_tensors="pt")
    inputs = inputs.to(device)
    # Embed the document
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = torch.mean(model_output.last_hidden_state, dim=1)
    return embedding




def load_docs(file_name):
    with open(file_name, "r") as f:
        text = f.readlines()
    # we assume marddown format, split into docs by single # header
    docs = []
    doc = ""
    for line in text:
        if line.startswith("## "):
            if doc:
                docs.append(doc)
            doc = line
        else:
            doc += line
    return docs

if __name__ == "__main__":
    main()