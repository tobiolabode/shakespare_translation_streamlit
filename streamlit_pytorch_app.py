import streamlit as st  # type: ignore


from pytorch_app import evaluate, encoder1, attn_decoder1, Lang, EncoderRNN, AttnDecoderRNN
# from app import decode_sequence, decode_sequence_beam_search
# from configs import config
# from server import get_english_translation, get_spanish_translation

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

        # #correct Translation
        # st.subheader("Actual Shakespare Translation")
        # st.success(f"{get_spanish_translation(input_text)}")


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
