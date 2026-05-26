
from dataclasses import dataclass
from pydoc import doc
from typing import List, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Document Model
# ----------------------------
@dataclass
class Document:
    id: str
    text: str
    source: str = ""

# ----------------------------
# TF-IDF RAG (No DB / No Vector DB)
# ----------------------------
class SimpleTfidfRAG:
    def __init__(self, docs: List[Document]):
        self.docs = docs # Store documents - 2
        self.vectorizer = TfidfVectorizer( # performing TF-IDF vectorization - 3
            stop_words="english", # stop words English means common words like "the", "is", "and" will be ignored in the vectorization process,
                                  # which helps to focus on more meaningful words in the documents.
            ngram_range=(1, 2),
            max_features=5000
        )
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def retrieve(self, query: str, top_k: int = 3): # top_k is the number of top relevant documents to retrieve based on cosine similarity - 4
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self.docs[i], float(sims[i])) for i in idx if sims[i] > 0]

    def build_prompt(self, query, retrieved):
        if not retrieved:
            context = "No relevant context found."
        else:
            context = "\n---\n".join([d.text for d, _ in retrieved])

        prompt = f"""
You are a helpful assistant. Answer ONLY using context.

CONTEXT:
{context}

QUESTION:
{query}


ANSWER:
"""
        return prompt.strip()

# ----------------------------
# Main Program
# ----------------------------
def main():
    docs = [
        Document("1","Employees apply leave via HR portal. Sick leave needs certificate after 2 days.","HR Policy"),
        Document("2","Passwords must be 12 characters including uppercase, lowercase, number and symbol. MFA required.","IT Policy"),
        Document("3","Insurance covers spouse and two children including hospitalization and outpatient.","Benefits Guide")
    ]

    rag = SimpleTfidfRAG(docs) # Initialize RAG with documents - 1
    print (rag.doc_matrix)
# *****************************************
    print("=== Simple RAG Without Vector DB ===")

    while True:
        query = input("\nAsk question (type exit to quit): ").strip()
        if query.lower() == "exit":
            break

        results = rag.retrieve(query, top_k=2)

        print ("\n*************************************Length************************************:"  )
        print(f"Number of retrieved documents: {len(results)}")
        print("Retrieved Documents (with scores):")
# ****************************************************************
        print("\n*************************************************************************:")
        print("\nRetrieved Documents:")
        if not results:
            print("No matches found.")
        else:            
            for doc, score in results:
                print(f"{doc.source} -> score={score:.3f}")     
                
        # print("\nRetrieved Documents:")
        # if not results:
        #     print("No matches found.")
        # else:
        #     for doc, score in results:
        #         print(f"{doc.source} -> score={score:.3f}")

        prompt = rag.build_prompt(query, results)
        print("\nPrompt Sent To LLM:")
        print(prompt)

if __name__ == "__main__":
    main()
