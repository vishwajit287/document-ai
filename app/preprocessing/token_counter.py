import tiktoken

DEFAULT_MODEL = "gpt-4o-mini"

def get_tokenizer(model_name: str = DEFAULT_MODEL):

    try:
        return tiktoken.encoding_for_model(model_name)

    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(
    text: str,
    model_name: str = DEFAULT_MODEL
) -> int:

    tokenizer = get_tokenizer(model_name)

    return len(tokenizer.encode(text))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model_name: str = DEFAULT_MODEL
) -> str:

    tokenizer = get_tokenizer(model_name)

    tokens = tokenizer.encode(text)

    return tokenizer.decode(tokens[:max_tokens])
