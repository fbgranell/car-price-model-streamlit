###############################
# This program lets you       #
# - Create a dashboard        #
# - Evevry dashboard page is  #
# created in a separate file  #
###############################

# Python libraries
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
# User module files


st.image('./images/header.png')
st.markdown("<h1 style='text-align: center;'>Predictor of used car prices</h1>", unsafe_allow_html=True)

# List of acceptable values
valid_fuels = ['Gasoline','Diesel','Hybrid']
valid_gearbox = ['Manual','Automatic']
valid_colors = ['White','Gray','Black','Blue','Red','Silver','Brown','Beige','Orange','Yellow','Other']
valid_brands = ['BMW',  'Peugeot',  'Audi',  'Volkswagen',  'Seat',  'Mercedes',  'Citroen',  'Opel',  'Renault',
  'Ford',  'Nissan',  'Kia',  'Hyundai',  'Fiat',  'Skoda',  'Mazda',  'Toyota',  'Mini',  'Volvo',  'Land',  'Dacia',
    'Jeep',  'Ds',  'Mitsubishi',  'Honda',  'Jaguar',  'Αlfa',  'Suzuki',  'Abarth',  'Smart',  'Cupra',  'Infiniti',
      'Porsche',  'Lexus',  'Subaru',  'Chevrolet',  'Maserati',  'Ssangyong',  'Lancia',  'Cadillac',  'Saab',
        'Chrysler',  'Dodge',  'Other',  'Aston',  'Bentley',  'Ferrari']
valid_types = ['Standard','Sport','4x4 (4WD)','Commercial Van']
valid_locations = ['Barcelona',  'Madrid',  'Valencia',  'Sevilla',  'Malaga',  'Vizcaya',  'Castellon',  'Zaragoza',
  'Murcia',  'Navarra',  'Ciudad Real',  'Cantabria',  'Cordoba',  'Ourense',  'Toledo',  'Pontevedra',  'Girona', 
   'Coruña',  'Tarragona',  'Alicante',  'Badajoz',  'Granada',  'Almeria',  'Guipuzcoa',  'Jaen',  'Huelva',  'Alava',
     'Guadalajara',  'Caceres',  'Valladolid',  'Asturias',  'Albacete',  'Lleida',  'Cadiz',  'Las Palmas',  'Lugo',
       'Tenerife',  'Islas Baleares',  'Leon',  'Burgos',  'Salamanca',  'Avila',  'Zamora',  'Huesca',  'Segovia',  'La Rioja',
         'Palencia',  'Teruel',  'Cuenca',  'Soria']
# Get input values

t_brand = st.selectbox("Please select the brand of the car: ",valid_brands, key = '1')

t_type = st.selectbox("Please select the type of  car: ",valid_types, key = '2')

t_age = st.number_input("Please enter the age of the car in years: ",format='%i',step=1, min_value=1, max_value=20, value = 5)

t_cv = st.number_input("Please enter the horsepower of the car in CV: ",format='%i',step=1, min_value=40, max_value=720, value = 136)

t_km = st.number_input("Please enter the mileage of the car in km: ",format='%i',step=1000, min_value=0, max_value=1700000,value=80000)

t_fuel = st.selectbox("Please select the type of fuel: ",valid_fuels, key = '3')

t_gearbox = st.selectbox("Please select the type of gearbox: ",valid_gearbox, key = '5')

t_color = st.selectbox("Please select the color of the car: ",valid_colors, key = '6')

t_location = st.selectbox("Please select the location where you want to buy the car: ", valid_locations, key = '7')

t_length = st.number_input("Please enter the length of the car in cm: ",format='%i',step=1, min_value=250, max_value=537, value = 436)

t_width = st.number_input("Please enter the width of the car in cm: ",format='%i',step=1, min_value=124, max_value=251, value = 180)

t_cmixto = st.number_input("Please enter the mixed fuel consumption of the car in L/km: ",format='%.1f',step=0.1, min_value=0.6, max_value=21.0, value = 5.0)

normalizer = pd.read_pickle('./machine-learning/normalizer.p')
encoder = pd.read_pickle('./machine-learning/encoder.p')
model = pd.read_pickle('./machine-learning/xgboost.p')

col1, col2, col3 = st.columns([1,1,1])
if col2.button("Get Your Prediction"):
    
    x = pd.DataFrame({'year':[t_age],
                     'cv':[t_cv],
                     'km':[t_km],
                     'fuel':[t_fuel],
                     'gearbox':[t_gearbox],
                     'color':[t_color],
                     'brand':[t_brand],
                     'cmixto':[t_cmixto],
                     'class':[t_type],
                     'location':[t_location],
                     'area':[t_width*t_length]
                     })

    # Normalize
    x_num = x.select_dtypes(np.number)
    x_norm = normalizer.transform(x_num)
    x_norm = pd.DataFrame(x_norm, columns = x_num.columns)

    # Onehot-encode
    x_cat = x.select_dtypes(object)
    x_oh = encoder.transform(x_cat).toarray()
    x_oh = pd.DataFrame(x_oh,columns=encoder.get_feature_names_out(x_cat.columns))

    x_scaled = pd.concat([x_norm,x_oh],axis=1)
    prediction = model.predict(x_scaled)

    st.success(f"The model predicted a price of {prediction[0]:.0f}€ for the specified car.")
    
    # Tengo que rehacer los pickles con las categorias igual en un lado y en otro,
    # y tambien no tener la id en el transformer del pickle. Después de todo 
    # quedarme con el repositorio del mac. 