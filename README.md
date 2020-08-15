# Translate App

![App Demo](resources/streamlit_app.gif)

A simple sanic app which translates english sentence to spanish. It uses the attention based Seq2Seq model trained.Built using [Streamlit](https://www.streamlit.io/) and deployed on [heroku](https://www.heroku.com/)

You can try out the heroku app here - https://engish-to-spanish-translation.herokuapp.com/

## Word Level Seq2Seq Model
Sequence-to-sequence Neural Machine Translation is an example of Conditional Language Model.

* Language Model - Decoder is predicting the next word of the target sentence based on the sequence generated so far
* Conditional - The predictions are conditioned on the source sentence x and the generated target seq

It calculate `P(y∣x)` where `x` is the source sentence & `y` is the target sentence

I have written a [blog](https://adimyth.github.io/notes/seq2seq/nlp/machine-translation/2020/08/11/Seq2Seq.html), explaining in detail the inner working as well as training mechanism.
Also, a jupyter notebook can be found in my [github repository](https://adimyth.github.io/notes/seq2seq/nlp/machine-translation/2020/08/11/Seq2Seq.html)

## Running Locally
There are two flavors to test this app locally. 
* Streamlit - WebApp
* Sanic App - CLI App
### Streamlit App
[Streamlit](https://streamlit.io) is an open source framework that let's users create web apps very easily.
1. Install streamlit
2. Clone the repo & navigate to *translate_app*directory
3. Run the app
```bash
streamlit run streamlit_app.py
```
4. Go to localhost:8051

### Sanic App
1. Install tensorflow 2.0 and sanic
2. Clone the git repo
3. Navigate to *translate_app* directory
4. Run server on *localhost:5000/translate*. 
```python
python server.py
```
5. Evaluate by making curl request at the above url with `input_sentence` as a parameter. Pass in the desired English sentence as value with it. 
```bash
curl -X POST -H "Content-Type: application/json"  -d '{"input_sentence":"What did you decide?"}' "localhost:5000/translate"
```
6. Alternatively, you could run the script `make_requests.py` and change the sentences in `sentences` list.

Kindly, :star: the repository, if you find it useful!
