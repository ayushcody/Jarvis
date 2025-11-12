from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RAGDoc:
    path: str
    text: str

class SimpleRAG:
    def __init__(self, corpus_dir: str):
        self.docs: List[RAGDoc] = []
        for p in sorted(Path(corpus_dir).glob("**/*")):
            if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
                self.docs.append(RAGDoc(str(p), p.read_text(encoding="utf-8")))
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in self.docs]) if self.docs else None

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[RAGDoc, float]]:
        if not self.docs:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_matrix)[0]
        ranked = sorted(zip(self.docs, sims), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def build_prompt(self, query: str, top_k: int = 3) -> str:
        ctx = self.retrieve(query, top_k=top_k)
        chunks = []
        for i, (doc, score) in enumerate(ctx, 1):
            chunks.append(f"[{i}] ({score:.3f}) {doc.path}\n{doc.text[:1200]}")
        context = "\n\n".join(chunks) if chunks else "No context docs."
        prompt = (
            "You are a helpful assistant. Answer the user's question using the CONTEXT.\n"
            "If the answer isn't in the context, say you don't know and offer next steps.\n\n"
            f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\n"
            "Answer in concise paragraphs with bullet points when helpful."
        )
        return prompt

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        return self.build_prompt(query, top_k=top_k)
