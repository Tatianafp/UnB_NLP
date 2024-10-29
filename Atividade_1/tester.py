import os
import json
import concurrent.futures
from time import time
from tqdm import tqdm

class TokenizerTester:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_texts(self, folder_path):
        """Carrega todos os textos dos arquivos JSON na pasta."""
        texts = []
        for filename in tqdm(os.listdir(folder_path)):
            # if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                texts.append(data["text"]) 
        return texts
    
    def _get_vocab_size(self, text):
        """Calcula dinamicamente o tamanho do vocabulário baseado no número de caracteres únicos no texto."""
        base_vocab = 256  # Tamanho mínimo do vocabulário
        unique_chars = len(set(text))-1
        adaptive_size = min(base_vocab + unique_chars // 10, 10000)  # Limita até 10.000
        return adaptive_size

    def _process_text(self, i, original_text, tokenizer, get_vocab_size, verbose):
        """Testa a tokenização de um texto."""
        try:
            if original_text == '':
                return None
            
            vocab_size = get_vocab_size(original_text)

            tokenizer.train(original_text, vocab_size, verbose=verbose)

            encoded = tokenizer.encode(original_text)  # Tokeniza o texto
            decoded = tokenizer.decode(encoded)  # Decodifica de volta

            if original_text != decoded:
                return (i, 'diff')  # Erro de decodificação
            return None  # Sucesso sem erros

        except Exception as e:
            return (i, f'exception: {str(e)}') 



    def test_tokenizer(self, texts, verbose=False):
        """Testa o tokenizador paralelamente."""
        list_errors = []  

        start_time = time() 

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._process_text, i, text, self.tokenizer, self._get_vocab_size, verbose
                ): i
                for i, text in enumerate(texts)
            }

            # Itera sobre os resultados conforme eles ficam prontos
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts)):
                result = future.result()
                if result:  # Se houver erro
                    list_errors.append(result)

        total_time = time() - start_time  # Tempo total
        errors = len(list_errors)

        # Exibe o relatório
        print(f"\nTeste concluído: {len(texts)} textos processados")
        print(f"Erros de decodificação: {errors}")
        print(f"Tempo total: {total_time:.2f} segundos")
        if list_errors:
            print(f"Erros encontrados: {list_errors}")
