# Importa bibliotecas necessárias
import os
from langchain_openai import OpenAIEmbeddings # Para gerar vetores semânticos usando OpenAI
from langchain_community.vectorstores import FAISS # Para gerenciar um banco de dados vetorial FAISS
from langchain_groq import ChatGroq

def carregar_banco_vetorial(caminho='banco_vetorial'):
    """Carrega o banco vetorial salvo."""

    # Verifica se o banco vetorial existe
    if not os.path.exists(caminho):
        raise ValueError(f'Banco vetorial não encontrado em: {caminho}')

    # Inicializa o modelo de embeddings
    modelo_embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

    # Carrega o banco vetorial
    banco_vetorial = FAISS.load_local(caminho, modelo_embeddings, allow_dangerous_deserialization=True)

    return banco_vetorial

def configurar_modelo_conversa():
    """Configura o modelo de linguagem para conversação."""

    modelo = ChatGroq(
        model='llama-3.1-70b-versatile', # Modelo de conversação
        temperature=0.2, # Baixa criatividade para respostas mais precisas
        max_tokens=500, # Limite de tokens na resposta
        api_key=os.getenv('GROQ_API_KEY')
    )

    return modelo

def montar_prompt(fragmentos, pergunta, contexto_memoria):
    """Monta manualmente o prompt com os fragmentos e a memória de contexto."""

    template = """
    Use os trechos fornecidos para responder a pergunta do usuário de forma clara e concisa.
    Se não souber a resposta, diga que não sabe.

    ### Contexto anterior:
    {contexto_memoria}

    ### Trechos:
    {fragmentos}

    ### Pergunta:
    {pergunta}

    ## Resposta:
    """

    # Juntar todos os fragmentos em um único texto
    # contexto = '\n'.join([f'{idx+1}. {doc.page_content}\n' for idx, doc in enumerate(fragmentos)])
    contexto = '\n'.join([f'{idx+1}. {doc[0].page_content}\n' for idx, doc in enumerate(fragmentos)])

    # Criar e formatar o prompt
    prompt = template.format(contexto_memoria=contexto_memoria, fragmentos=contexto, pergunta=pergunta)

    return prompt

def conversar():
    """Interface principal para conversar com os documentos carregados, mantendo memória de contexto."""

    # Inicializa a memória de contexto como uma string vazia
    contexto_memoria = ''

    try:
        # Carregar banco vetorial
        banco_vetorial = carregar_banco_vetorial()

        # Configurar modelo de conversação
        modelo = configurar_modelo_conversa()

        # Loop de conversação
        while True:
            # Solicitar pergunta ao usuário
            pergunta = input("\n📄 Qual sua pergunta sobre os documentos? (ou 'sair' para encerrar): ")

            # Condição de saída
            if pergunta.lower() == 'sair':
                print('Encerrando conversa. Até logo! 👋')
                break

            # Processar pergunta
            try:
                # Recuperar fragmentos relevantes
                # recuperador = banco_vetorial.as_retriever(search_kwargs={'k': 3})
                # documentos_relevantes = recuperador.invoke(pergunta)
                # documentos_relevantes = banco_vetorial.similarity_search(pergunta, k=3)
                documentos_relevantes = banco_vetorial.similarity_search_with_score(pergunta, k=3)

                # Montar o prompt
                prompt = montar_prompt(documentos_relevantes, pergunta, contexto_memoria)

                print('\n🛠️ Prompt Montado Manualmente:\n')
                print(prompt) # Visualizar o prompt completo

                # Obter resposta do modelo
                resposta = modelo.invoke(prompt)

                # Adicionar a interação ao contexto de memória
                contexto_memoria += f'Pergunta: {pergunta}\nResposta: {resposta.content}\n------'

                # Manter apenas as últimas 5 interações
                interacoes = contexto_memoria.split('------') # Divide em interações
                interacoes_filtradas = interacoes[-5:] # Filtra as últimas 5
                contexto_memoria = '------\n'.join(interacoes_filtradas) # Reconstrói o contexto

                # Imprimir resposta
                print('\n🤖 Resposta:')
                print(resposta.content)

                # Mostrar fontes (opcional)
                print('\n📍 Fontes:')
                # for idx, documento in enumerate(documentos_relevantes, 1):
                for idx, (documento, score) in enumerate(documentos_relevantes, 1):
                    print(f"{idx}. Página {documento.metadata.get('page', 'N/A')}: {documento.metadata.get('source', 'N/A')}")
                    print(f'  Score de similaridade: {score:.4f}') # Mostra o score com 4 casas decimais
                    # Scores próximos a 0 indicam documentos muito similares
                    # Scores tipicamente variam entre 0 e 1
                    # Scores acima de 0.5 geralmente indicam baixa similaridade
                    # Um documento idêntico teria score 0
                    print(f'  Trecho: {documento.page_content[:200]}...\n')

            except Exception as erro:
                print(f'Erro ao processar pergunta: {erro}')

    except Exception as erro:
        print(f'Erro ao iniciar conversação: {erro}')

# Execução principal
if __name__ == '__main__':
    conversar()