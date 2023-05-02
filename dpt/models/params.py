params = {
    'bert': 'bert-large-uncased',
    'fs_bert_with_label_marker': 'bert-large-uncased',
    'fs_bert_with_label_init': 'bert-large-uncased',
    'electra': 'google/electra-large-discriminator',
    'fs_electra_with_label_marker': 'google/electra-large-discriminator',
    'fs_electra_with_label_itself': 'google/electra-large-discriminator',
    'fs_electra_add_label_itself': 'google/electra-large-discriminator',
    'fs_electra_with_label_init': 'google/electra-large-discriminator',
    'fs_electra_add_label_init': 'google/electra-large-discriminator',
    'begin_with_labels_and_downline': 'google/electra-large-discriminator',
    'begin_with_labels': 'google/electra-large-discriminator',
    'begin_with_sentiment_and_downline': 'google/electra-large-discriminator',
    'begin_with_sentiment': 'google/electra-large-discriminator',
}

def labelTokenizer(data, tokenizer, label_tokens, max_len, label_positions):
    kv = {"input_ids": [], "attention_mask": [], "positions": []}
    for sp in data:
        st = sentence_pair_tokenizer(tokenizer, sp, max_len, len(label_tokens))
        bolt = len(st)
        st = st + label_tokens
        sti = tokenizer.convert_tokens_to_ids(st)
        lvt = len(sti)
        sti = sti + [0] * (max_len - lvt)
        kv["input_ids"].append(sti)

        att_mask = [1] * lvt + [0] * (max_len - lvt)
        kv["attention_mask"].append(att_mask)

        positions = [t + bolt for t in label_positions]
        kv["positions"].append(positions)
    return kv


def sentence_pair_tokenizer(t, sp, ml, ol=0):
    s0 = sp[0]
    s0_t = t.tokenize(s0)
    s1 = sp[1]
    s1_t = [] if s1 is None else t.tokenize(s1)
    cl = len(s0_t) + len(s1_t)
    cl = cl + 2 if s1 is None else cl + 3

    ofl = cl + ol - ml
    if ofl > 0:
        if s1 is None:
            s0_t = s0_t[: len(s0_t) - ofl]
        else:
            s1_t = s1_t[: len(s1_t) - ofl]

    s1_t = [] if s1 is None else s1_t + ["[SEP]"]
    s_t = ["[CLS]"] + s0_t + ["[SEP]"] + s1_t
    return s_t