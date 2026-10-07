"""TF-IDF по whitelist терминов механизмов."""
from sklearn.feature_extraction.text import TfidfVectorizer

WHITELIST = [
    "overflow", "underflow", "wraparound", "truncation",х
    "oob", "out-of-bounds", "out of bounds",
    "null", "null_deref", "nullptr", "dereference",
    "use-after-free", "uaf", "double-free", "double free",
    "race", "toctou", "reentrancy",
    "format string", "format_string",
    "uninitialized", "uninit",
    "bounds", "bounds check",
    "privilege", "access control", "permission",
    "deserializ", "deserialize",
    "memcpy", "strcpy", "strcat", "malloc", "calloc", "free",
    "index", "offset", "length", "size",
    "signed", "unsigned", "integer",
]


def build_tfidf(texts: list[str], max_features: int = 60):
    """Возвращает (matrix, vectorizer, terms)."""
    # Препроцессинг: whitelist слова оставляем, стоп-слова убираем sklearn'ом
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,   # log(TF), уменьшает влияние повторов
        token_pattern=r"[A-Za-z][A-Za-z0-9_\-]*",
    )
    matrix = vec.fit_transform(texts)
    terms = list(vec.get_feature_names_out())
    return matrix, vec, terms