"""pip install langchain transformers torch sentence-transformers faiss-cpu langchain-community langchain-huggingface"""

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import threading as th
import tkinter as tk

lista_nomes = [
    "Ana", "Anderson", "Antonio", "Beatriz", "Bernardo", "Bruna", 
    "Carlos", "Camila", "Caio", "Daniel", "Diana", "Diego"
]

lista_nomes_minusculo = [name.lower() for name in lista_nomes]

# Carregar modelo de embeddings BERT
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Criar documentos com os nomes
documentos = [Document(page_content=nome) for nome in lista_nomes]

# Criar um banco de dados vetorial com FAISS
vectorstore = FAISS.from_documents(documentos, embedding_model)

def search_datas(prefixo, top_k=5):
    if not prefixo.strip():
        return []  # Retorna lista vazia se não houver entrada
    
    # Gerar o embedding do prefixo digitado
    query_embedding = embedding_model.embed_query(prefixo)
    
    # Buscar os itens mais similares no banco de dados
    resultados = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    
    # Retornar os nomes encontrados
    return [res.page_content for res in resultados]


class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Teste de pesquisa de nome com Bert A.I')
        self.root.geometry('500x500')

        self.searchEntry = tk.Entry(self.root, width=30)
        self.searchEntry.pack(pady=10)

        self.names = tk.Listbox(self.root)
        self.names.pack(pady=10, fill=tk.BOTH, expand=True)

        self.root.after(500, self.verify_search)
        
        self.root.mainloop()

    def actualize_names(self, lista):
        self.names.delete(0, tk.END)
        for item in lista:
            self.names.insert(tk.END, item)

    def verify_search(self):
        search_data = self.searchEntry.get()

        if search_data.lower() in lista_nomes_minusculo:
            self.actualize_names([search_data])
        else:
            datas_searched = search_datas(search_data)
            self.actualize_names(datas_searched)

        self.root.after(500, self.verify_search)


if __name__ == "__main__":
    interface = Interface()