#import standard libraries
from cProfile import label
import pickle
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import base64
from PIL import Image
from sklearn.linear_model import LinearRegression

label=LabelEncoder()


image1=Image.open('Acura.jpeg')
image2=Image.open('Audi.jpeg')
image3=Image.open('BMW.jpeg')
image4=Image.open('Buick.jpeg')
image5=Image.open('Cadillac.jpeg')
image6=Image.open('Chevrolet.jpeg')
image7=Image.open('Chrysler.jpeg')
image8=Image.open('Dodge.jpeg')
image9=Image.open('Ford.jpeg')
image10=Image.open('GMC.jpeg')
image11=Image.open('Honda.jpeg')
image12=Image.open('Hummer.jpeg')
image13=Image.open('Hyundai.jpeg')
image14=Image.open('Infiniti.jpeg')
image15=Image.open('Isuzu.jpeg')
image16=Image.open('Jaguar.jpg')
image17=Image.open('Jeep.jpeg')
image18=Image.open('Kia.jpeg')
image19=Image.open('Land Rover.jpeg')
image20=Image.open('Lexus.jpeg')
image21=Image.open('Lincoln.jpeg')
image22=Image.open('Mazda.jpeg')
image23=Image.open('Mercedes-Benz.jpeg')
image24=Image.open('Mercury.jpeg')
image25=Image.open('MINI.jpeg')
image26=Image.open('Mitsubishi.jpeg')
image27=Image.open('Nissan.jpeg')
image28=Image.open('Oldsmobile.jpeg')
image29=Image.open('Pontiac.jpeg')
image30=Image.open('Porsche.jpg')
image31=Image.open('Saab.jpeg')
image32=Image.open('Saturn.jpeg')
image33=Image.open('Scion.jpeg')
image34=Image.open('Subaru.jpeg')
image35=Image.open('Suzuki.jpeg')
image36=Image.open('Toyota.jpeg')
image37=Image.open('Volkswagen.jpeg')
image38=Image.open('Volvo.jpg')
#set the title page
st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered",
    initial_sidebar_state="auto"
)
#set the background color
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: 100 cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('Lamborghini.png')
data=pd.read_csv('Car_data.csv')
EDA=st.sidebar.selectbox("Explore Data Analysis",['EDA'])
if st.sidebar.checkbox('EDA'):
  make_type=data.groupby(['Make','Type']).size().reset_index().rename(columns={0:'diifernt_type_makes'})
  st.set_option('deprecation.showPyplotGlobalUse', False)
  ax=plt.axes()
  ax.set(facecolor='skyblue')
  sns.set(rc={'figure.figsize':(17,9)},style='darkgrid')
  ax.set_title('different car brad have differnt types prices',fontsize=15,fontweight='bold')
  sns.barplot(x='Make',y='diifernt_type_makes',hue='Type',data=make_type[:50],palette='rainbow')
  plt.xlabel('Car Brands')
  plt.ylabel('Count the cars')
  plt.show()
  st.pyplot()
  make_origin=data.groupby(['Make','Origin']).size().reset_index().rename(columns={0:'diifernt_origin_makes'})
  ax=plt.axes()
  ax.set(facecolor='yellow')
  sns.set(rc={'figure.figsize':(17,9)},style='darkgrid')
  ax.set_title('different car brad with high sales rate in differnt origin',fontsize=15,fontweight='bold')
  sns.barplot(x='Make',y='diifernt_origin_makes',hue='Origin',data=make_origin[:50],palette='rainbow')
  plt.xticks(rotation=45)
  plt.xlabel('Car Brands')
  plt.ylabel('Count the cars')
  plt.show()
  st.pyplot()
  make_driveTrain=data.groupby(['Make','DriveTrain']).size().reset_index().rename(columns={0:'diifernt_make_driveTrain'})
  ax=plt.axes()
  ax.set(facecolor='orange')
  sns.set(rc={'figure.figsize':(17,9)},style='darkgrid')
  ax.set_title('different car brad with different drivtrains ',fontsize=25,fontweight='bold')
  sns.barplot(x='Make',y='diifernt_make_driveTrain',hue='DriveTrain',data=make_driveTrain[:50],palette=['black','red','green'])
  plt.xticks(rotation=45)
  plt.xlabel('Car Brands')
  plt.ylabel('Count the cars')
  plt.show()
  st.pyplot()
  st.write("The distribution of the MSRP")
  ax=plt.axes()
  ax.set(facecolor='grey')
  sns.set(rc={'figure.figsize':(17,9)},style='darkgrid')
  ax.set_title('Distribution of the car price in $ ',fontsize=25,fontweight='bold')
  sns.distplot(data['MSRP'])
  plt.xticks(rotation=45)
  plt.xlabel('Car price')
  plt.ylabel('Count of the values')
  plt.show()
  st.pyplot()
  st.write("The distribution of the Invoice")
  ax=plt.axes()
  ax.set(facecolor='red')
  sns.set(rc={'figure.figsize':(17,9)},style='darkgrid')
  ax.set_title('Distribution of the car invoice in $ ',fontsize=25,fontweight='bold')
  sns.distplot(data['Invoice'],color='black',hist=True)
  plt.xticks(rotation=45)
  plt.xlabel('Car invoice')
  plt.ylabel('Count of the values')
  plt.show()
  st.pyplot()
#Read the data

data=data.drop(['Unnamed: 0'],axis=1)
data['Model']=LabelEncoder().fit_transform(data['Model'])
data['Make']=LabelEncoder().fit_transform(data['Make'])
data['Type']=LabelEncoder().fit_transform(data['Type'])
data['Origin']=LabelEncoder().fit_transform(data['Origin'])
data['DriveTrain']=LabelEncoder().fit_transform(data['DriveTrain'])
X=data.drop(['MSRP'],axis=1)
y=data['MSRP']
linear=LinearRegression()
linear.fit(X,y)
#define the function
def main():
    Make=st.selectbox("Enter company brand",['Acura', 'Audi', 'BMW','Buick', 'Cadillac', 'Chevrolet',
       'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Hummer', 'Hyundai',
       'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover',
       'Lexus', 'Lincoln', 'MINI', 'Mazda', 'Mercedes-Benz', 'Mercury',
       'Mitsubishi', 'Nissan', 'Oldsmobile', 'Pontiac', 'Porsche', 'Saab',
       'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen',
       'Volvo'])
    if Make == "Acura":
      Make_1 = 0
    elif Make == "Audi":
      Make_1 = 1
    elif Make == "BMW":
      Make_1 = 2
    elif Make == "Buick":
      Make_1 = 3
    elif Make == "Cadillac":
      Make_1 = 4
    elif Make == "Chevrolet":
      Make_1 = 5
    elif Make == "Chrysler":
      Make_1 = 6
    elif Make == "Dodge":
      Make_1 = 7
    elif Make == "Ford":
      Make_1 = 8
    elif Make == "GMC":
      Make_1 = 9
    elif Make == "Honda":
      Make_1 = 10
    elif Make == "Hummer":
      Make_1 = 11
    elif Make == "Hyundai":
      Make_1 = 12
    elif Make == "Infiniti":
      Make_1 = 13
    elif Make == "Isuzu":
      Make_1 = 14
    elif Make == "Jaguar":
      Make_1 = 15
    elif Make == "Jeep":
      Make_1 = 16
    elif Make == "Kia":
      Make_1 = 17
    elif Make == "Land Rover":
      Make_1 = 18
    elif Make == "Lexus":
      Make_1 = 19
    elif Make == "Lincoln":
      Make_1 = 20
    elif Make == "MINI":
      Make_1 = 21
    elif Make == "Mazda":
      Make_1 = 22
    elif Make == "Mercedes-Benz":
      Make_1 = 23
    elif Make == "Mercury":
      Make_1 = 24
    elif Make == "Mitsubishi":
      Make_1 = 25
    elif Make == "Nissan":
      Make_1 = 26
    elif Make == "Oldsmobile":
      Make_1 = 27
    elif Make == "Pontiac":
      Make_1 = 28
    elif Make == "Porsche":
      Make_1 = 29
    elif Make == "Saab":
      Make_1 = 30
    elif Make == "Saturn":
      Make_1 = 31
    elif Make == "Saturn":
      Make_1 = 31
    elif Make == "Saturn":
      Make_1 = 31
    elif Make == "Scion":
      Make_1 = 32
    elif Make == "Subaru":
      Make_1 = 33
    elif Make == "Suzuki":
      Make_1 = 34
    elif Make == "Toyota":
      Make_1 = 35
    elif Make == "Volkswagen":
      Make_1 = 36
    elif Make == "Volvo":
      Make_1 = 37
    #model
    Model=st.selectbox(" ",['MDX', 'RSX Type S 2dr', 'TSX 4dr', 'TL 4dr', '3.5 RL 4dr',
       '3.5 RL w/Navigation 4dr', 'NSX coupe 2dr manual S', 'A4 1.8T 4dr',
       'A41.8T convertible 2dr', 'A4 3.0 4dr',
       'A4 3.0 Quattro 4dr manual', 'A4 3.0 Quattro 4dr auto',
       'A6 3.0 4dr', 'A6 3.0 Quattro 4dr', 'A4 3.0 convertible 2dr',
       'A4 3.0 Quattro convertible 2dr', 'A6 2.7 Turbo Quattro 4dr',
       'A6 4.2 Quattro 4dr', 'A8 L Quattro 4dr', 'S4 Quattro 4dr',
       'RS 6 4dr', 'TT 1.8 convertible 2dr (coupe)',
       'TT 1.8 Quattro 2dr (convertible)',
       'TT 3.2 coupe 2dr (convertible)', 'A6 3.0 Avant Quattro',
       'S4 Avant Quattro', 'X3 3.0i', 'X5 4.4i', '325i 4dr', '325Ci 2dr',
        '325Ci convertible 2dr', '325xi 4dr', '330i 4dr', '330Ci 2dr',
       '330xi 4dr', '525i 4dr', '330Ci convertible 2dr', '530i 4dr',
       '545iA 4dr', '745i 4dr', '745Li 4dr', 'M3 coupe 2dr',
       'M3 convertible 2dr', 'Z4 convertible 2.5i 2dr',
       'Z4 convertible 3.0i 2dr', '325xi Sport', 'Rainier',
       'Rendezvous CX', 'Century Custom 4dr', 'LeSabre Custom 4dr',
       'Regal LS 4dr', 'Regal GS 4dr', 'LeSabre Limited 4dr',
       'Park Avenue 4dr', 'Park Avenue Ultra 4dr', 'Escalade', 'SRX V8',
       'CTS VVT 4dr', 'Deville 4dr', 'Deville DTS 4dr', 'Seville SLS 4dr',
       'XLR convertible 2dr', 'Escalade EXT', 'Suburban 1500 LT',
       'Tahoe LT', 'TrailBlazer LT', 'Tracker', 'Aveo 4dr',
       'Aveo LS 4dr hatch', 'Cavalier 2dr', 'Cavalier 4dr',
       'Cavalier LS 2dr', 'Impala 4dr', 'Malibu 4dr', 'Malibu LS 4dr',
       'Monte Carlo LS 2dr', 'Impala LS 4dr', 'Impala SS 4dr',
       'Malibu LT 4dr', 'Monte Carlo SS 2dr', 'Astro', 'Venture LS',
       'Corvette 2dr', 'Corvette convertible 2dr', 'Avalanche 1500',
       'Colorado Z85', 'Silverado 1500 Regular Cab', 'Silverado SS',
       'SSR', 'Malibu Maxx LS', 'PT Cruiser 4dr',
       'PT Cruiser Limited 4dr', 'Sebring 4dr', 'Sebring Touring 4dr',
       '300M 4dr', 'Concorde LX 4dr', 'Concorde LXi 4dr',
       'PT Cruiser GT 4dr', 'Sebring convertible 2dr',
       '300M Special Edition 4dr', 'Sebring Limited convertible 2dr',
       'Town and Country LX', 'Town and Country Limited', 'Crossfire 2dr',
       'Pacifica', 'Durango SLT', 'Neon SE 4dr', 'Neon SXT 4dr',
       'Intrepid SE 4dr', 'Stratus SXT 4dr', 'Stratus SE 4dr',
       'Intrepid ES 4dr', 'Caravan SE', 'Grand Caravan SXT',
       'Viper SRT-10 convertible 2dr', 'Dakota Regular Cab',
       'Dakota Club Cab', 'Ram 1500 Regular Cab ST', 'Excursion 6.8 XLT',
       'Expedition 4.6 XLT', 'Explorer XLT V6', 'Escape XLS',
       'Focus ZX3 2dr hatch', 'Focus LX 4dr', 'Focus SE 4dr',
       'Focus ZX5 5dr', 'Focus SVT 2dr', 'Taurus LX 4dr',
       'Taurus SES Duratec 4dr', 'Crown Victoria 4dr',
       'Crown Victoria LX 4dr', 'Crown Victoria LX Sport 4dr',
       'Freestar SE', 'Mustang 2dr (convertible)',
       'Mustang GT Premium convertible 2dr',
       'Thunderbird Deluxe convert w/hardtop 2d', 'F-150 Regular Cab XL',
       'F-150 Supercab Lariat', 'Ranger 2.3 XL Regular Cab', 'Focus ZTW',
       'Taurus SE', 'Envoy XUV SLE', 'Yukon 1500 SLE',
       'Yukon XL 2500 SLT', 'Safari SLE', 'Canyon Z85 SL Regular Cab',
       'Sierra Extended Cab 1500', 'Sierra HD 2500', 'Sonoma Crew Cab',
       'Civic Hybrid 4dr manual (gas/electric)',
       'Insight 2dr (gas/electric)', 'Pilot LX', 'CR-V LX', 'Element LX',
       'Civic DX 2dr', 'Civic HX 2dr', 'Civic LX 4dr', 'Accord LX 2dr',
       'Accord EX 2dr', 'Civic EX 4dr', 'Civic Si 2dr hatch',
       'Accord LX V6 4dr', 'Accord EX V6 2dr', 'Odyssey LX', 'Odyssey EX',
       'S2000 convertible 2dr', 'H2', 'Santa Fe GLS', 'Accent 2dr hatch',
       'Accent GL 4dr', 'Accent GT 2dr hatch', 'Elantra GLS 4dr',
       'Elantra GT 4dr', 'Elantra GT 4dr hatch', 'Sonata GLS 4dr',
       'Sonata LX 4dr', 'XG350 4dr', 'XG350 L 4dr', 'Tiburon GT V6 2dr',
       'G35 4dr', 'G35 Sport Coupe 2dr', 'I35 4dr', 'M45 4dr',
       'Q45 Luxury 4dr', 'FX35', 'FX45', 'Ascender S', 'Rodeo S',
       'X-Type 2.5 4dr', 'X-Type 3.0 4dr', 'S-Type 3.0 4dr',
       'S-Type 4.2 4dr', 'S-Type R 4dr', 'Vanden Plas 4dr', 'XJ8 4dr',
       'XJR 4dr', 'XK8 coupe 2dr', 'XK8 convertible 2dr', 'XKR coupe 2dr',
       'XKR convertible 2dr', 'Grand Cherokee Laredo', 'Liberty Sport',
       'Wrangler Sahara convertible 2dr', 'Sorento LX', 'Optima LX 4dr',
       'Rio 4dr manual', 'Rio 4dr auto', 'Spectra 4dr',
       'Spectra GS 4dr hatch', 'Spectra GSX 4dr hatch',
       'Optima LX V6 4dr', 'Amanti 4dr', 'Sedona LX', 'Rio Cinco',
       'Range Rover HSE', 'Discovery SE', 'Freelander SE', 'GX 470',
       'LX 470', 'RX 330', 'ES 330 4dr', 'IS 300 4dr manual',
       'IS 300 4dr auto', 'GS 300 4dr', 'GS 430 4dr', 'LS 430 4dr',
       'SC 430 convertible 2dr', 'IS 300 SportCross', 'Navigator Luxury',
       'Aviator Ultimate', 'LS V6 Luxury 4dr', 'LS V6 Premium 4dr',
       'LS V8 Sport 4dr', 'LS V8 Ultimate 4dr', 'Town Car Signature 4dr',
       'Town Car Ultimate 4dr', 'Town Car Ultimate L 4dr', 'Cooper',
       'Cooper S', 'Tribute DX 2.0', 'Mazda3 i 4dr', 'Mazda3 s 4dr',
       'Mazda6 i 4dr', 'MPV ES', 'MX-5 Miata convertible 2dr',
       'MX-5 Miata LS convertible 2dr', 'RX-8 4dr automatic',
       'RX-8 4dr manual', 'B2300 SX Regular Cab', 'B4000 SE Cab Plus',
       'G500', 'ML500', 'C230 Sport 2dr', 'C320 Sport 2dr', 'C240 4dr',
       'C320 Sport 4dr', 'C320 4dr', 'C32 AMG 4dr', 'CL500 2dr',
       'CL600 2dr', 'CLK320 coupe 2dr (convertible)',
       'CLK500 coupe 2dr (convertible)', 'E320 4dr', 'E500 4dr',
       'S430 4dr', 'S500 4dr', 'SL500 convertible 2dr', 'SL55 AMG 2dr',
       'SL600 convertible 2dr', 'SLK230 convertible 2dr', 'SLK32 AMG 2dr',
       'C240', 'E320', 'E500', 'Mountaineer', 'Sable GS 4dr',
       'Grand Marquis GS 4dr', 'Grand Marquis LS Premium 4dr',
       'Sable LS Premium 4dr', 'Grand Marquis LS Ultimate 4dr',
       'Marauder 4dr', 'Monterey Luxury', 'Sable GS', 'Endeavor XLS',
       'Montero XLS', 'Outlander LS', 'Lancer ES 4dr', 'Lancer LS 4dr',
       'Galant ES 2.4L 4dr', 'Lancer OZ Rally 4dr auto',
       'Diamante LS 4dr', 'Galant GTS 4dr', 'Eclipse GTS 2dr',
       'Eclipse Spyder GT convertible 2dr', 'Lancer Evolution 4dr',
       'Lancer Sportback LS', 'Pathfinder Armada SE', 'Pathfinder SE',
       'Xterra XE V6', 'Sentra 1.8 4dr', 'Sentra 1.8 S 4dr',
       'Altima S 4dr', 'Sentra SE-R 4dr', 'Altima SE 4dr',
       'Maxima SE 4dr', 'Maxima SL 4dr', 'Quest S', 'Quest SE',
       '350Z coupe 2dr', '350Z Enthusiast convertible 2dr',
       'Frontier King Cab XE V6', 'Titan King Cab XE', 'Murano SL',
       'Alero GX 2dr', 'Alero GLS 2dr', 'Silhouette GL', 'Aztekt',
       'Sunfire 1SA 2dr', 'Grand Am GT 2dr', 'Grand Prix GT1 4dr',
       'Sunfire 1SC 2dr', 'Grand Prix GT2 4dr', 'Bonneville GXP 4dr',
       'Montana', 'Montana EWB', 'GTO 2dr', 'Vibe', 'Cayenne S',
       '911 Carrera convertible 2dr (coupe)',
       '911 Carrera 4S coupe 2dr (convert)', '911 Targa coupe 2dr',
       '911 GT2 2dr', 'Boxster convertible 2dr',
       'Boxster S convertible 2dr', '9-3 Arc Sport 4dr', '9-3 Aero 4dr',
       '9-5 Arc 4dr', '9-5 Aero 4dr', '9-3 Arc convertible 2dr',
       '9-3 Aero convertible 2dr', '9-5 Aero', 'VUE', 'Ion1 4dr',
       'lon2 4dr', 'lon3 4dr', 'lon2 quad coupe 2dr',
       'lon3 quad coupe 2dr', 'L300-2 4dr', 'L300 2', 'xA 4dr hatch',
       'xB', 'Impreza 2.5 RS 4dr', 'Legacy L 4dr', 'Legacy GT 4dr',
       'Outback Limited Sedan 4dr', 'Outback H6 4dr',
       'Outback H-6 VDC 4dr', 'Impreza WRX 4dr', 'Impreza WRX STi 4dr',
       'Baja', 'Forester X', 'Outback', 'XL-7 EX', 'Vitara LX',
       'Aeno S 4dr', 'Aerio LX 4dr', 'Forenza S 4dr', 'Forenza EX 4dr',
       'Verona LX 4dr', 'Aerio SX', 'Prius 4dr (gas/electric)',
       'Sequoia SR5', '4Runner SR5 V6', 'Highlander V6', 'Land Cruiser',
       'RAV4', 'Corolla CE 4dr', 'Corolla S 4dr', 'Corolla LE 4dr',
       'Echo 2dr manual', 'Echo 2dr auto', 'Echo 4dr', 'Camry LE 4dr',
       'Camry LE V6 4dr', 'Camry Solara SE 2dr', 'Camry Solara SE V6 2dr',
       'Avalon XL 4dr', 'Camry XLE V6 4dr', 'Camry Solara SLE V6 2dr',
       'Avalon XLS 4dr', 'Sienna CE', 'Sienna XLE Limited',
       'Celica GT-S 2dr', 'MR2 Spyder convertible 2dr', 'Tacoma',
       'Tundra Regular Cab V6', 'Tundra Access Cab V6 SR5', 'Matrix XR',
       'Touareg V6', 'Golf GLS 4dr', 'GTI 1.8T 2dr hatch',
       'Jetta GLS TDI 4dr', 'New Beetle GLS 1.8T 2dr',
       'Jetta GLI VR6 4dr', 'New Beetle GLS convertible 2dr',
       'Passat GLS 4dr', 'Passat GLX V6 4MOTION 4dr',
       'Passat W8 4MOTION 4dr', 'Phaeton 4dr', 'Phaeton W12 4dr',
       'Jetta GL', 'Passat GLS 1.8T', 'Passat W8', 'XC90 T6', 'S40 4dr',
       'S60 2.5 4dr', 'S60 T5 4dr', 'S60 R 4dr', 'S80 2.9 4dr',
       'S80 2.5T 4dr', 'C70 LPT convertible 2dr',
       'C70 HPT convertible 2dr', 'S80 T6 4dr', 'V40', 'XC70'])
    
    Type=st.selectbox("Enter Car type ",['SUV', 'Sedan', 'Sports', 'Wagon','Truck', 'Hybrid'])
    if Type == "SUV":
      Type_1 = 0
    elif Type == "Sedan":
      Type_1 = 1
    elif Type == "Sports":
      Type_1 = 2
    elif Type == "Wagon":
      Type_1 = 3
    elif Type == "Truck":
      Type_1 = 4
    elif Type == "Hybrid":
      Type_1 =5
    Origin=st.selectbox("Enter your origin",['Asia', 'Europe', 'USA'])
    if Origin == "Asia":
      Origin_1=0
    elif Origin == "Europe":
      Origin_1 =1
    elif Origin == "USA":
      Origin_1 =2
    DriveTrain=st.selectbox("Enter driven tupe",['All', 'Front', 'Rear'])
    if DriveTrain == "All":
      DriveTrain_1 =0
    elif DriveTrain == "Front":
      DriveTrain_1 =1
    elif DriveTrain == "Rear":
      DriveTrain_1 =2
    Invoice=st.slider("Enter Invoice amount Ex:33337.0",9875.0,173560.0)
    EngineSize=st.slider("Enter  EngineSize Ex:3.5",1.3,8.3)
    Cylinders=st.slider("Enter Cylinders Ex:6.0",3.0,12.0)
    Horsepower=st.slider("Enter Horsepower Ex:265.0",73.0,500.0)
    MPG_City=st.slider("Enter MPG_City Ex:17.0",10.0,60.0)
    MPG_Highway=st.slider("Enter MPG_Highway, Ex:23.0",12.0,66.0)
    Weight=st.slider("Enter Weight Ex:4451.0",1850.0,7190.0)
    Wheelbase=st.slider("Enter Wheelbase Ex:106.0",89.0,144.0)
    Length=st.slider("Enter Length Ex:189.0",143.0,238.0)
    if st.button("Predict"):
      Model=label.fit_transform([Model])
      
      result=linear.predict([[Make_1,Model, Type_1, Origin_1, DriveTrain_1,Invoice,
       EngineSize,Cylinders,Horsepower,MPG_City,MPG_Highway,
       Weight, Wheelbase,Length]])
      st.success(print(result)
      st.balloons()
      if Make == "Acura":
        st.image(image1,caption="Acura Car Model ",width=1000)
      elif Make == "Audi":
        st.image(image2,caption="Audi Car Model ",width=1000)
      elif Make == "BMW":
        st.image(image3,caption="BMW Car Model ",width=1000)
      elif Make == "Buick":
        st.image(image4,caption="Buick Car Model ",width=1000)
      elif Make == "Cadillac":
        st.image(image5,caption="Cadillac Car Model ",width=1000)
      elif Make == "Chevrolet":
        st.image(image6,caption="Chevrolet Car Model ",width=1000)
      elif Make == "Chrysler":
        st.image(image7,caption="Chrysler Car Model ",width=1000)
      elif Make == "Dodge":
        st.image(image8,caption="Dodge Car Model ",width=1000)
      elif Make == "Ford":
        st.image(image9,caption="Ford Car Model ",width=1000)
      elif Make == "GMC":
        st.image(image10,caption="GMC Car Model ",width=1000)
      elif Make == "Honda":
        st.image(image11,caption="Honda Car Model ",width=1000)
      elif Make == "Hummer":
        st.image(image12,caption="Hummer Car Model ",width=1000)
      elif Make == "Hyundai":
        st.image(image13,caption="Hyundai Car Model ",width=1000)
      elif Make == "Infiniti":
        st.image(image14,caption="Infiniti Car Model ",width=1000)
      elif Make == "Isuzu":
        st.image(image15,caption="Isuzu Car Model ",width=1000)
      elif Make == "Jaguar":
        st.image(image16,caption="Jaguar Car Model ",width=1000)
      elif Make == "Jeep":
        st.image(image17,caption="Jeep Car Model ",width=1000)
      elif Make == "Kia":
        st.image(image18,caption="Kia Car Model ",width=1000)
      elif Make == "Land Rover":
         st.image(image19,caption="Land Rover Car Model ",width=1000)
      elif Make == "Lexus":
         st.image(image20,caption="Lexus Car Model ",width=1000)
      elif Make == "Lincoln":
          st.image(image21,caption="Lincoln Car Model ",width=1000)
      elif Make == "MINI":
        st.image(image22,caption="MINI Car Model ",width=1000)
      elif Make == "Mazda":
        st.image(image23,caption="Mazda Car Model ",width=1000)
      elif Make == "Mercedes-Benz":
        st.image(image24,caption="Mercedes-Benz Car Model ",width=1000)
      elif Make == "Mercury":
        st.image(image25,caption="Mercury Car Model ",width=1000)
      elif Make == "Mitsubishi":
        st.image(image26,caption="Mitsubishi Car Model ",width=1000)
      elif Make == "Nissan":
        st.image(image27,caption="Nissan Car Model ",width=1000)
      elif Make == "Oldsmobile":
        st.image(image28,caption="Oldsmobile Car Model ",width=1000)
      elif Make == "Pontiac":
         st.image(image29,caption="Pontiac Car Model ",width=1000)
      elif Make == "Porsche":
        st.image(image30,caption="Porsche Car Model ",width=1000)
      elif Make == "Saab":
        st.image(image31,caption="Saab Car Model ",width=1000)
      elif Make == "Saturn":
        st.image(image32,caption="Saturn Car Model ",width=1000)
      
      elif Make == "Scion":
        st.image(image33,caption="Scion Car Model ",width=1000)
      elif Make == "Subaru":
        st.image(image34,caption="Subaru Car Model ",width=1000)
      elif Make == "Suzuki":
        st.image(image35,caption="Suzuki Car Model ",width=1000)
      elif Make == "Toyota":
        st.image(image36,caption="Toyota Car Model ",width=1000)
      elif Make == "Volkswagen":
        st.image(image37,caption="Volkswagen Car Model ",width=1000)
      elif Make == "Volvo":
        st.image(image38,caption="Volvo Car Model ",width=1000)
       


      


if __name__ == "__main__":
   main()

 
