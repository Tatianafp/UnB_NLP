import regex as re
from tqdm import tqdm

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class NotTrainedError(Exception):
    """Exceção para indicar que o tokenizador não foi treinado."""
    def __init__(self, message="O tokenizador precisa ser treinado antes de ser usado."):
        super().__init__(message)

class Tokenizer():
	def __init__(self):
		self.pattern =  re.compile(GPT4_SPLIT_PATTERN)
		self.frequency = None
		self.merges = None
		self.vocab = None

	def _get_pairs_frequency(self, tokens, frequency):
		"""Calcula a frequência dos pares consecutivos de tokens."""
		for i in range(len(tokens) - 1):
			pair = (tokens[i], tokens[i+1])
			if pair in frequency:
				frequency[pair] += 1
			else: 
				frequency[pair] = 1
		return frequency

	def _get_most_frequent_pair(self, tokens, frequency={}):
		"""Obtém o par de tokens mais frequente."""
		self.frequency = self._get_pairs_frequency(tokens, frequency)
		return max(self.frequency, key=self.frequency.get)

	def _get_least_frequent_pair(self, tokens):
		"""Obtém o par menos frequente, considerando merges prévios."""
		self.frequency = self._get_pairs_frequency(tokens, {})
		return min(self.frequency, key=lambda p: self.merges.get(p, float("inf")))

	def _merge(self, tokens, pair, new_token): 
		"""Funde pares de tokens em um novo token."""
		updated_tokens = []
		i = 0
		while i < len(tokens):
			if i <len(tokens) -1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
				updated_tokens.append(new_token)
				i += 2
			else:
				updated_tokens.append(tokens[i])
				i += 1
		return updated_tokens
	
	def _convert_text_to_tokens(self, text):
		"""Converte texto em uma lista de bytes."""
		tokens = text.encode("utf-8") # raw bytes
		return list(map(int, tokens))

	def train(self, text, vocab_size, verbose=False):
		"""Treina o tokenizador utilizando Byte Pair Encoding (BPE)."""
		if vocab_size < 256:
			raise ValueError("O tamanho do vocabulário deve ser pelo menos 256.")

		num_merges = vocab_size - 256
		text_chunks = re.findall(self.pattern, text)
		print('ids')
		ids = [list(ch.encode("utf-8")) for ch in tqdm(text_chunks)]
		print('Total ids: ', len(ids))
		self.merges = {}
		self.vocab = {idx: bytes([idx]) for idx in range(256)}
		
		# Realiza as fusões necessárias para o vocabulário desejado.
		print('merges')
		for i in tqdm(range(num_merges)):
			self.frequency = {}
			for chunk_ids in tqdm(ids[:-1]):
				self._get_pairs_frequency(chunk_ids, self.frequency)
			pair = self._get_most_frequent_pair(chunk_ids, self.frequency)
			idx = 256 + i
			ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in tqdm(ids)]

			self.merges[pair] = idx
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

			if verbose:
				print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {self.frequency[pair]} occurrences")

	def _encode_chunk(self, text):
		"""Codifica um trecho de texto em tokens."""
		if not self.merges:
			self.merges = {}
		tokens = list(text.encode("utf-8"))

		# Aplica merges até não ser mais possível.
		while len(tokens) >= 2:
			pair = self._get_least_frequent_pair(tokens)
			if pair not in self.merges:
				break # nothing else can be merged
			idx = self.merges[pair]
			tokens = self._merge(tokens, pair, idx)
		return tokens
	
	def encode(self, text):
		"""Codifica o texto em uma lista de IDs de tokens usando regex para identificar os grupos de tokens."""
		text_chunks = re.findall(self.pattern, text)
		ids = []
		for chunk in text_chunks:
			chunk_ids = self._encode_chunk(chunk)
			ids.extend(chunk_ids)
		return ids

	def decode(self, ids):
		"""Decodifica uma lista de IDs de tokens para o texto original."""
		if not self.vocab:
			raise NotTrainedError()
		tokens = b"".join(self.vocab[idx] for idx in ids)
		text = tokens.decode("utf-8", errors="replace")
		return text