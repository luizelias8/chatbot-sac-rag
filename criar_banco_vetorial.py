import os
import shutil
from langchain_community.document_loaders import CSVLoader # Importa o carregador de CSV do LangChain
from langchain_openai import OpenAIEmbeddings # Importa os embeddings da OpenAI para gerar os vetores
from langchain_community.vectorstores import FAISS # Importa o FAISS para criar o banco vetorial

# Define o caminho do arquivo CSV com as perguntas e respostas
arquivo_csv = 'perguntas_respostas.csv'

# Define o caminho da pasta onde o banco vetorial será salvo
pasta_faiss = 'faiss_perguntas_respostas'

# Verifica se a pasta já existe. Se sim, deleta a pasta e todo o conteúdo.
if os.path.exists(pasta_faiss):
    print(f"Removendo pasta existente: {pasta_faiss}")
    shutil.rmtree(pasta_faiss)

# Inicializa o carregador para ler o arquivo CSV
carregador = CSVLoader(file_path=arquivo_csv, encoding='utf-8')

# Carregar os documentos do CSV
documentos = carregador.load()
# Transforma cada linha do CSV em um documento que pode ser usado para gerar embeddings
# Cada linha do CSV será tratada como um único "documento" no contexto vetorial

# Exibir os documentos carregados
# for i, doc in enumerate(documentos, start=1):
#     print(f'{i}. {doc.page_content}')

# Inicializa os embeddings da OpenAI para transformar texto em vetores semânticos
embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

# Cria o banco vetorial FAISS a partir dos documentos e seus respectivos embeddings
armazenamento_vetorial = FAISS.from_documents(documents=documentos, embedding=embeddings)
# O FAISS otimiza a busca de similaridade entre os vetores

# Salva o banco vetorial no diretório local para reutilização em futuras execuções
armazenamento_vetorial.save_local(folder_path=pasta_faiss)