# Importa bibliotecas necessárias
import os
from langchain_openai import OpenAIEmbeddings # Para gerar vetores semânticos usando OpenAI
from langchain_community.vectorstores import FAISS # Para gerenciar um banco de dados vetorial FAISS
from langchain_groq import ChatGroq

# Define o caminho onde o banco vetorial está armazenado
caminho_banco_vetorial = 'faiss_perguntas_respostas'

# Inicializa os embeddings da OpenAI para transformar texto em vetores semânticos
embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

# Carrega o banco vetorial existente do FAISS no caminho especificado
# O parâmetro `allow_dangerous_deserialization=True` permite carregar dados potencialmente inseguros
banco_vetorial = FAISS.load_local(
    folder_path=caminho_banco_vetorial, # Caminho onde o banco está armazenado
    embeddings=embeddings, # Usa o mesmo método de embeddings com o qual o banco foi criado
    allow_dangerous_deserialization=True # Habilita carregamento de arquivos potencialmente inseguros
)
# Uso de um banco de informações externo:: utiliza o FAISS para recuperar textos relevantes com base em embeddings.
# O banco vetorial FAISS é utilizado para armazenar e buscar informações relevantes.
# O parâmetro embeddings é necessário para o método load_local porque o banco vetorial foi originalmente criado usando uma função específica de embeddings.
# A maneira como os textos são representados depende diretamente do modelo de embeddings utilizado.
# Para garantir que as consultas sejam representadas no mesmo espaço vetorial do banco de dados, o mesmo método de embeddings precisa ser aplicado.
# Se você não fornecer o mesmo método, o FAISS não saberá como gerar os vetores das novas consultas, e a correspondência semântica falhará ou produzirá resultados inconsistentes.

# Pergunta feita pelo usuário
pergunta_usuario = input('Pergunta: ')

# Realiza uma busca no banco vetorial com base na similaridade semântica da pergunta
# Retorna os 3 documentos mais similares, junto com seus scores
resultados = banco_vetorial.similarity_search_with_score(pergunta_usuario, k=3)
# Os textos recuperados serão organizados em um contexto, que será utilizado para enriquecer a resposta do modelo de linguagem.
# Modelos de linguagem têm um limite fixo de tokens que podem ser processados em uma única solicitação (entrada + saída).
# Por exemplo, um modelo pode suportar até 4096 tokens.
# Incorporar todos os dados disponíveis no banco de conhecimento ao prompt seria inviável porque poderia ultrapassar o limite de tokens.
# Redução de ruído e relevância: Selecionar apenas os trechos mais relevantes do banco vetorial garante que o contexto fornecido ao modelo seja útil e significativo.
# Isso evita que tokens sejam desperdiçados com informações irrelevantes, aumentando a eficiência e a qualidade da resposta.
# Recuperar trechos relevantes reduz a quantidade de dados que precisam ser incorporados ao prompt.
# Assim, o espaço de tokens disponível pode ser melhor aproveitado para a pergunta do usuário e a resposta gerada pelo modelo.

# Itera sobre os resultados para exibir os textos relevantes e seus scores
# for documento, score in resultados:
#     # Exibe o conteúdo do documento e o score de similaridade no console
#     print(f"Texto: {documento.page_content}, Score de similaridade: {score}")

# Configura o modelo de linguagem da API Groq
modelo = ChatGroq(
    model='llama-3.1-70b-versatile', # Nome do modelo de linguagem
    # temperature=0, # Parâmetro opcional para controlar a criatividade (mais baixo = menos criativo)
    api_key=os.getenv('GROQ_API_KEY') # Chave de API armazenada em variável de ambiente
)

# Inicializa o contexto, que será usado para formar uma resposta ao usuário
contexto = ''

# Concatena os textos relevantes obtidos do banco vetorial para formar o contexto
for documento, score in resultados:
    contexto += f'{documento.page_content}\n\n' # Adiciona cada conteúdo de documento ao contexto
# print(contexto)

# Define a lista de mensagens para o modelo de linguagem Groq
mensagens = [
    # Mensagem de configuração inicial informando o papel do modelo
    {'role': 'system', 'content': 'Você é um assistente de atendimento de um SAC de um banco digital.'},
    # Mensagem com o contexto gerado a partir dos resultados da busca no banco vetorial
    {'role': 'system', 'content': f'Para responder ao usuário, você deve se basear nas seguintes perguntas e respostas que já foram feitas anteriormente:\n{contexto}\n'}
]
# Criação de contexto dinâmico: as informações recuperadas do FAISS são injetadas dinamicamente no prompt do modelo de linguagem.

# Adiciona a pergunta feita pelo usuário à lista de mensagens
mensagens.append({'role': 'human', 'content': pergunta_usuario})

# Envia as mensagens ao modelo de linguagem e recebe a resposta
resposta = modelo.invoke(mensagens)
# Após adicionar o contexto ao prompt, o modelo de linguagem ChatGroq gera uma resposta com base na pergunta do usuário e nas informações recuperadas.
# Interação com um modelo generativo: o ChatGroq usa o contexto para gerar uma resposta específica e enriquecida para a consulta do usuário.

# Exibe a resposta gerada pelo modelo no console
print(resposta.content)
# Essa abordagem combina retrieval (para garantir que a resposta seja fundamentada em dados precisos)
# com generation (para criar respostas em linguagem natural), que é a essência de um sistema RAG.