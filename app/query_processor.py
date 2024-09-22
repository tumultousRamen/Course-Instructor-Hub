from nltk.tokenize import word_tokenize
from typing import Dict, Callable, Set

class QueryProcessor:
    def __init__(self, stop_words: Set[str], predefined_queries: Dict[str, Callable]):
        self.stop_words = stop_words
        self.predefined_queries = predefined_queries

    def process_query(self, query: str) -> str:
        tokens = word_tokenize(query.lower())
        tokens = [w for w in tokens if w not in self.stop_words]
        
        best_match = None
        max_overlap = 0
        
        for key in self.predefined_queries.keys():
            key_tokens = word_tokenize(key.lower())
            overlap = len(set(tokens) & set(key_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = key
        
        if best_match and max_overlap > 0:
            return self.predefined_queries[best_match](query)
        else:
            return "I'm sorry, I don't understand that query. Could you please rephrase it?"