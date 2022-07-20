from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def fun1():
    return render_template('index.html')


co = pickle.load(open('count_vec.pkl', 'rb'))
sol = pickle.load(open('twitter_analysis.pkl', 'rb'))
@app.route('/predict', methods=['GET', 'POST'])
def fun2():
    if request.method == 'POST':
        text = request.form["message"]
        d = [text]
        vectors = co.transform(d)
        vectors = vectors.toarray()
        final = sol.predict(vectors)
        final = final[0]
        if final == 0:
            return render_template('index.html', prediction_text="positive tweet")
        else:
            return render_template('index.html', prediction_text="negative tweet")


if __name__ == '__main__':
    app.run(debug=True)