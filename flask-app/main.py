from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/estimate',methods=['POST','GET'])
def estimate():
    if request.method=='POST':
        result=request.form
        location = result['location']
        title = result['title']
        perks = result.getlist('perks')
        duration = result['duration']
        skills = result.getlist('skills')

        pkl_file = open('index_dict.pkl', 'rb')
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
            cat_vector[skill_list] = 1
        except:
            pass

        pkl_file = open('internshala_lgbm_model.pkl', 'rb')
        model = pickle.load(pkl_file)

        scaler_file = open("scaler.pkl", "rb")
        scaler = pickle.load(scaler_file)

        prediction = model.predict(scaler.transform(cat_vector.reshape(1,-1)))
        pred_range = "Rs. " + str(int((prediction-1963).tolist()[0])) + " to " + str(int((prediction+1963).tolist()[0])) + " approx."
        
        return render_template('result.html',prediction=pred_range)

    
if __name__ == '__main__':
	app.debug = False
	app.run()
