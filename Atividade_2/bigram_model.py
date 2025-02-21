import math
from collections import Counter
from tqdm import tqdm

class BigramModel:
    def __init__(self, tokenizer, alpha=0.1):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.vocab = set()
        self.tokenizer = tokenizer
        self.alpha = alpha  # Parâmetro de suavização

    def train(self, data):
        for sequence in tqdm(data):
            self.unigram_counts.update(sequence)
            self.bigram_counts.update(zip(sequence[:-1], sequence[1:]))
            self.vocab.update(sequence)

    def bigram_probability(self, w1, w2):
        vocab_size = len(self.vocab)
        bigram_count = self.bigram_counts[(w1, w2)] + self.alpha
        unigram_count = self.unigram_counts[w1] + (self.alpha * vocab_size)

        return bigram_count / unigram_count
    
    def bigram_probability_with_context(self, context, w2): #ainda faltam melhorias
        """
        Calcula a probabilidade de uma palavra w2, dada uma sequência de palavras anteriores (contexto).
        O contexto é uma lista das últimas n palavras.
        """
        context = tuple(context)  # Converte o contexto para tupla para garantir que seja imutável
        context_len = len(context)
        
        # Verifica se o contexto é longo o suficiente
        if context_len == 0:
            return 1 / (len(self.vocab) + 1)  # Probabilidade uniforme quando não há contexto

        prob = 0
        for i in range(context_len):
            bigram = (context[i], w2)
            prob += self.bigram_counts[bigram]
        
        # Normaliza a probabilidade
        return prob / sum(self.bigram_counts.values())

    def calculate_perplexity(self, test_tokens):
        total_log_prob = 0
        total_N = 0

        for sentence in tqdm(test_tokens):
            N = len(sentence)
            if N < 2:
                continue

            log_prob = 0
            for i in range(1, N):
                w1, w2 = str(sentence[i-1]), str(sentence[i])
                prob = self.bigram_probability(w1, w2)

                log_prob += math.log2(prob)

            total_log_prob += log_prob
            total_N += N

        if total_N == 0:
            return float('inf')

        return 2 ** (-total_log_prob / total_N)

    def sample_next_token(self, w1):
        """
        Prevê a próxima palavra mais provável após w1.
        """
        if w1 not in self.unigram_counts:  # Se w1 nunca apareceu, escolha um token aleatório
            return list(self.vocab)[0]

        probabilities = {w2: self.bigram_probability(w1, w2) for w2 in self.vocab}
        return max(probabilities, key=probabilities.get)
    
    def generate_text(self, start_text, max_tokens=20):
        generated_tokens = self.tokenizer.encode(start_text)
        for _ in range(max_tokens):
            w1 = generated_tokens[-1]
            next_token = self.sample_next_token(w1)

            if next_token in generated_tokens[-3:]:
                continue

            generated_tokens.append(next_token)

            if next_token == self.tokenizer.encode("</s>")[0]:
                break
        
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def sample_next_token_with_context(self, w1):
        """
        Prevê a próxima palavra mais provável após w1.
        """
        if w1 not in self.unigram_counts:  # Se w1 nunca apareceu, escolha um token aleatório
            return list(self.vocab)[0]

        probabilities = {w2: self.bigram_probability_with_context(w1, w2) for w2 in self.vocab}
        return max(probabilities, key=probabilities.get)

    def generate_text_with_context(self, start_text, max_tokens=20):
        generated_tokens = self.tokenizer.encode(start_text)
        for _ in range(max_tokens):
            w1 = generated_tokens[-3:-1]
            next_token = self.sample_next_token_with_context(w1)

            if next_token in generated_tokens[-3:]:
                continue

            generated_tokens.append(next_token)

            if next_token == self.tokenizer.encode("</s>")[0]:
                break
        
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text
