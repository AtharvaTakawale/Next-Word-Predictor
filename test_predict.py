from model_utils import ngram_predict_next, transformer_predict_next

tests = [
    "I am learning",
    "Machine learning models",
    "People read books",
    "Technology continues to"
]

for t in tests:
    print("INPUT:", t)
    print("NGRAM:", ngram_predict_next(t, top_k=5))
    print("TRANSFORMER:", transformer_predict_next(t, top_k=5))
    print()
