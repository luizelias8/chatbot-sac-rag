import os
from langchain_community.document_loaders import CSVLoader # Importa o carregador de CSV do LangChain
from langchain_openai import OpenAIEmbeddings # Importa os embeddings da OpenAI para gerar os vetores
from langchain_community.vectorstores import FAISS # Importa o FAISS para criar o banco vetorial

def carregar_arquivo_csv(caminho_arquivo_csv):
    # Inicializa o carregador para ler o arquivo CSV
    carregador = CSVLoader(file_path=caminho_arquivo_csv, encoding='utf-8')
    # Carregar os documentos do CSV

    documentos = carregador.load()
    # Transforma cada linha do CSV em um documento que pode ser usado para gerar embeddings
    # Cada linha do CSV será tratada como um único "documento" no contexto vetorial

    # Retorna a lista de todos os documentos carregados
    return documentos

def criar_banco_vetorial(documentos):
    """Cria embeddings para os documentos e salva no FAISS."""

    # Inicializa o modelo de embeddings da OpenAI
    modelo_embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

    # Cria o banco vetorial FAISS a partir dos documentos e do modelo de embeddings
    banco_vetorial = FAISS.from_documents(documentos, modelo_embeddings)

    # Retorna o banco vetorial
    return banco_vetorial

def salvar_banco_vetorial(banco_vetorial, caminho='banco_vetorial'):
    """Salva o banco vetorial em disco."""

    # Salva o banco vetorial no caminho especificado
    banco_vetorial.save_local(caminho)

    # Imprime o local de salvamento
    print(f'Banco vetorial salvo em: {caminho}')

# Bloco de execução principal, rodado apenas se o script for executado diretamente
if __name__ == '__main__':
    # Carrega o arquivo
    documentos = carregar_arquivo_csv('perguntas_respostas.csv')

    # Cria banco vetorial
    banco_vetorial = criar_banco_vetorial(documentos)

    # Salva o banco vetorial em disco
    salvar_banco_vetorial(banco_vetorial)