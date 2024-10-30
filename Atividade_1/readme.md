# Atividade: Aplicação do Algoritmo BPE

## Objetivo

Nesta atividade, o objetivo é reproduzir o algoritmo de *Byte Pair Encoding* (BPE) conforme demonstrado no vídeo [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) e aplicá-lo a um conjunto de dados fornecido pelo professor. A atividade tem como foco o entendimento prático de compressão de texto e subpalavras em Processamento de Linguagem Natural.

## Conjunto de Dados

- O arquivo contendo uma coleção de textos pode ser baixado no link [Aqui!](https://unbbr-my.sharepoint.com/:u:/g/personal/thiagodepaulo_unb_br/ETRbkBjaKihNmsQI0eWq9RkB3I9tE-SluKccadGOFJYqmA?e=V9k4Vb).
- O corpus é formado por vários arquivos no formato '.json'. Processe o BPE em todos os arquivos. 

## Estrutura do código

### tatiktoken.py
- **Objetivo**: Implementa um tokenizador com Byte Pair Encoding (BPE) utilizando regex inspirada no GPT-4.

- **Classes**:
    - NotTrainedError: Exceção levantada se o tokenizador for usado sem treinamento.
    - Tokenizer: Responsável por treinar e aplicar BPE para tokenização e codificação.
- **Métodos Principais**:
    - train: Treina o tokenizador para criar merges e vocabulário.
    - encode: Codifica um texto para IDs de tokens.
    - decode: Converte IDs de tokens de volta para texto.

### tester.py
- **Objetivo**: Testa tokenizadores utilizando múltiplos textos e processamento paralelo.

- **Classes e Atributos**  
  - `TokenizerTester`  
    - Atributo: `tokenizer` – Instância do tokenizador.

- **Métodos Públicos**  
  - `__init__(self, tokenizer)`  
    - Inicializa a classe com um tokenizador.
  
  - `load_texts(self, folder_path)`  
    - **Descrição**: Carrega textos a partir de arquivos JSON em uma pasta.  
    - **Retorna**: Lista de textos.

  - `test_tokenizer(self, texts, verbose=False)`  
    - **Descrição**: Testa o tokenizador paralelamente em uma lista de textos.  
    - **Funcionamento**:
      - Utiliza processamento paralelo para tokenização e decodificação.
      - Exibe relatório com tempo total e erros encontrados.

- **Métodos Privados**  
  - `_get_vocab_size(self, text)`  
    - **Descrição**: Calcula o tamanho do vocabulário adaptativo com base nos caracteres únicos.  

  - `_process_text(self, i, original_text, tokenizer, get_vocab_size, verbose)`  
    - **Descrição**: Tokeniza e decodifica um texto individualmente.  
    - **Funcionamento**:
      - Treina, codifica e decodifica o texto.
      - Verifica erros e registra exceções, se necessário.

## Demais arquivos
- **run.ipynb**: notebook onde se aplica o algoritmo BPE no conjunto de dados fornecido pelo professor, mais descrições no próprio notebook.
- **requirements.txt**: lista de bibliotecas necessárias para executar os códigos desenvolvidos para esta atividade.