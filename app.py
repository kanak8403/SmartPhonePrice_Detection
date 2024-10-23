from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    brand = request.form.get('brand')
    ratings = float(request.form['Ratings'])
    ram = float(request.form['RAM'])
    rom = float(request.form['ROM'])
    mobile_size = float(request.form['Mobile_Size'])
    primary_cam = float(request.form['Primary_Cam'])
    selfi_cam = float(request.form['Selfi_Cam'])
    battery_power = float(request.form['Battery_Power'])
    features = np.array([[ratings, ram, rom, mobile_size, primary_cam, selfi_cam, battery_power]])
    # Make prediction
    predicted_price = model.predict(features)

    # Format the output price
    output = '{0:.2f}'.format(predicted_price[0])  # Adjust formatting as needed

    return render_template('forest_fire.html',
                               pred=f'Predicted Price: {output}',
                               brand=brand,
                               ram=ram,
                               rom=rom,
                               ratings=ratings,
                               mobile_size=mobile_size,
                               primary_cam=primary_cam,
                               selfi_cam=selfi_cam,
                               battery_power=battery_power)

    return render_template("forest_fire.html")


if __name__ == '__main__':
    app.run(debug=True)
