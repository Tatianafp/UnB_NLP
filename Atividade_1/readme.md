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
    - `NotTrainedError`: Exceção levantada se o tokenizador for usado sem treinamento.
    - `Tokenizer`: Responsável por treinar e aplicar BPE para tokenização e codificação.
- **Métodos Principais**:
    - `train(self, text, vocab_size, verbose=False)`: 
        - Treina o tokenizador para criar merges e vocabulário.
    - `encode(self, text)`: 
        - Codifica um texto para IDs de tokens.
    - `decode`: 
        - Converte IDs de tokens de volta para texto.

### tester.py
- **Objetivo**: Testa tokenizadores utilizando múltiplos textos e processamento paralelo.

- **Classes**  
  - `TokenizerTester`  
    - Atributo: `tokenizer` – Instância do tokenizador.

- **Métodos Principais**  
  - `load_texts(self, folder_path)`  
    - Carrega textos a partir de arquivos JSON em uma pasta.  

  - `_get_vocab_size(self, text)`  
    - Calcula o tamanho do vocabulário adaptativo com base nos caracteres únicos.  

  - `_process_text(self, i, original_text, tokenizer, get_vocab_size, verbose)`  
    - Treina, codifica e decodifica o texto.
    - Verifica erros e registra exceções, se necessário.

  - `test_tokenizer(self, texts, verbose=False)`  
    - Testa o tokenizador paralelamente em uma lista de textos.
    - Exibe relatório com tempo total e erros encontrados.


## Demais arquivos
- **run.ipynb**: notebook onde se aplica o algoritmo BPE no conjunto de dados fornecido pelo professor, mais descrições no próprio notebook.
- **requirements.txt**: lista de bibliotecas necessárias para executar os códigos desenvolvidos para esta atividade.