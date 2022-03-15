from flask import Flask, request, render_template
import pickle
import numpy as np
from scipy.special import inv_boxcox

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/estimate',methods=['POST','GET'])
def estimate():
    if request.method=='POST':
        result=request.form
        location = result['location']
        openings = result['openings']
        applicants = result['applicants']
        title = result['title']
        perks = result.getlist('perks')
        duration = result['duration']
        skills = result.getlist('skills')

        pkl_file = open('flask-app\index_dict.pkl', 'rb')
        index_dict = pickle.load(pkl_file)
        cat_vector = np.zeros(len(index_dict))
        
        skill_list = []
        for i in skills:
            skill_list.append(index_dict[(i)])

        perk_list = []
        for j in perks:
            perk_list.append(index_dict[(j)])


        try:
            cat_vector[index_dict['Duration']] = int(duration)
        except:
            pass
        try:
            cat_vector[index_dict['Number of Openings']] = int(openings)
        except:
            pass
        try:
            cat_vector[perk_list] = 1
        except:
            pass
        try:
            cat_vector[index_dict['Location_'+str(location)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['Title_'+str(title)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['Applicants_'+str(applicants)]] = 1
        except:
            pass
        try:
            cat_vector[skill_list] = 1
        except:
            pass

        pkl_file = open('flask-app\internshala_lgbm_model.pkl', 'rb')
        model = pickle.load(pkl_file)

        scaler_file = open("flask-app\scaler.pkl", "rb")
        scaler = pickle.load(scaler_file)

        prediction = model.predict(scaler.transform(cat_vector.reshape(1,-1)))
        lmda = 0.2731531291764417
        prediction = inv_boxcox(prediction, lmda)
        pred = "Rs. " + str(int((prediction).tolist()[0])) + " approx."
        
        return render_template('result.html',prediction=pred)

    
if __name__ == '__main__':
	app.debug = False
	app.run()
