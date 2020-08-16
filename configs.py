from attrdict import AttrDict  # type: ignore

config = {
    "encoder_path": "models/encoder_model.h5",
    "decoder_path": "models/decoder_model.h5",
    "input_word_index": "pickles/input_word_index.pkl",
    "target_word_index": "pickles/target_word_index.pkl",
    "url": "https://api.mymemory.translated.net/get",
    "max_length_src": 47,
    "max_length_tar": 47,
}
config = AttrDict(config)
