import streamlit as st  # type: ignore

# FIXME: AttributeError: Can't get attribute 'Lang' on <module '__main__' from '/content/shakespare_translation_streamlit/streamlit_pytorch_app.py'>
# Error caused with pickled object was created in main module
# FIXME: ModuleNotFoundError: No module named 'torch' on local machine
# Maybe conda verion of pytorch
import pickle
import random
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    import pytorch_app  # import Foo into main_module's namespace explicitly
    from pytorch_app import Lang
except AttributeError:
    class CustomUnpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if name == 'Lang':
                from pytorch_app import Lang
                return Lang
            return super().find_class(module, name)

    print('Using CustomUnpickler')
    input_lang = CustomUnpickler(open('Input_outputs_langs/input_lang.pkl', 'rb')).load()
    output_lang = CustomUnpickler(open('Input_outputs_langs/output_lang.pkl', 'rb')).load()


from pytorch_app import EncoderRNN, AttnDecoderRNN, encoder1, attn_decoder1, evaluate
# from app import decode_sequence, decode_sequence_beam_search
# from configs import config
# from server import get_english_translation, get_spanish_translation


# pytorch functions
def evaluateRandomly(encoder, decoder, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        st.write('\>', pair[0])
        st.write('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        st.write('<', output_sentence)
        st.write('')
        # return output_sentence, pair[0], pair[1]


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'pairs':
            from pytorch_app import Lang
            return Lang
        return super().find_class(module, name)

# FIXME: Cant use CustomUnpickler for non-classes


try:
    pairs = open('Input_outputs_langs/pairs.pkl', 'rb')
    pairs = pickle.load(pairs)
    pairs.close()
    # pairs = pairs.read()
except AttributeError:
    pass
    print('Using CustomUnpickler')
    # pairs = CustomUnpickler(io.open('Input_outputs_langs/pairs.pkl', 'rb')).load()
    # pairs = pairs.read()


# TODO: insert attention matrrics

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    st.pyplot(fig)


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


# title
st.title("English to Shakespare Translator")
st.markdown(
    """
This is demo English to Shakespare Translator. The template for this code was from [insert name here]
"""
)

# get input text
input_text = st.text_input(label="English text.")

# get beam width if beam search selected
# beam_width = None
# if st.checkbox("Beam Search"):
#     beam_width = st.slider(label="Beam Width", min_value=3, max_value=7, value=3)

# my return response
if st.button("Submit"):
    with st.spinner("Translating..."):
        st.subheader("Predicted Shakespare Translation")
        predicted_seq, attentions = evaluate(encoder1, attn_decoder1, input_text)
        formatted_predicted_seq = ' '.join(predicted_seq)
        st.success(formatted_predicted_seq)
        st.pyplot(showAttention(input_text, predicted_seq, attentions))

        # #correct Translation
        # st.subheader("Actual Shakespare Translation")
        # st.success(f"{get_spanish_translation(input_text)}")

# insert random pairs
if st.button("Random Pairs"):
    with st.spinner("Generating..."):
        for i in range(5):
            evaluateRandomly(encoder1, attn_decoder1)
        # output_sentence, pair_1, pair_2 = evaluateRandomly(encoder1, attn_decoder1)
        # for i in range(10):
        #     st.write('>>', pair_1)
        #     st.write('=', pair_2)
        #     st.write('<', output_sentence)
        #     st.write('')


# evaluateAndShowAttention("elle a cinq ans de moins que moi .")
#
# evaluateAndShowAttention("elle est trop petit .")
#
# evaluateAndShowAttention("je ne crains pas de mourir .")
#
# evaluateAndShowAttention("c est un jeune directeur plein de talent .")


# # return response
# if st.button("Submit"):
#     with st.spinner("Translating ..."):
#         if beam_width != None:
#             spanish_sequences = decode_sequence_beam_search(
#                 input_text, beam_width=beam_width
#             )
#             st.subheader("Predicted Spanish Translation")
#             for x in spanish_sequences:
#                 st.success(f"{x[:-4]}")
#             spanish_seq = spanish_sequences[0][:-4]
#         elif beam_width == None:
#             st.subheader("Predicted Spanish Translation")
#             spanish_seq = decode_sequence(input_text)[:-4]
#             st.success(f"{spanish_seq[:-4]}")
#     # actual spanish translation
#     st.subheader("Actual Spanish Translation")
#     st.success(f"{get_spanish_translation(input_text)}")
#     # predicted spanish back to english
#     st.subheader("Predicted Spanish to English Translation")
#     st.success(f"{get_english_translation(spanish_seq)}")


st.markdown(
    "The interactive app is created using [Streamlit](https://streamlit.io/), an open-source framework that lets users creating apps for machine learning projects very easily."
)
