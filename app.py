from flask import Flask, request, render_template 
import pickle

app=Flask(__name__)
model=pickle.load(open('modelRL.pkl','rb'))

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
	features=[float(x) for x in request.form.values()]
	import numpy as np 
	features_finals=[np.array(features)]
	prediction=model.predict(features_finals)
	output=round(prediction[0],2)
	return render_template("index.html", 
		prediction_text=f"La charge d'assurance est {output}")

if __name__=="__main__":
	app.run(debug=True)
