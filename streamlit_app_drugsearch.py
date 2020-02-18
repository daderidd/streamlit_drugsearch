import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from PIL import Image
import altair as alt
import pydeck as pdk
st.image('https://reseau-delta.ch/assets/ci_content/images/logo.png',width = 180)

st.title('Recherche de données médicaments')


@st.cache
def load_data(nrows,path,DATE_COLUMN = None):
    data = pd.read_csv(path, nrows = nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    if DATE_COLUMN is not None:
    	data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
data_folder = Path("../../../../../Switchdrive/PhD/Data/Delta/").resolve()
DATA_URL = data_folder/'Clean_data'/'20200802_drug.csv'
path = data_folder/'Clean_data'/'20200802_geometries.csv'
#Import geometries
df_geometries = load_data(108162,path = path) 
 # Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(3576919,path = DATA_URL,DATE_COLUMN = 'delivereddate')
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

if st.sidebar.checkbox('Show raw prescription data',key = 'Drug prescription data'):
    st.subheader('Raw data')
    st.write(data)

DATE_COLUMN = None
DATA_URL = ('./Data/atc_nomenclature.csv')
#Import the ATC nomenclature from WHO
atc_data = load_data(8329,path = DATA_URL) #ATC nomenclature 
atc_data = atc_data.fillna('') 

atc_filter = st.sidebar.text_input('Filter ATC name list').lower() #Add text box input for ATC filtering
##########################
atc = st.sidebar.selectbox("ATC choices", atc_data[atc_data.nameen.str.contains(atc_filter)]['nameen'].sort_values().unique().tolist(), 0) #Add a select box to choose an ATC among the filtered ones
##########################
#Import ATC levels
df_atc_levels = pd.read_csv('./Data/atc_levels.csv') 
dict_atc_levels= dict(zip(df_atc_levels.atc, df_atc_levels.level))
if atc != '':
	DATE_COLUMN = 'delivereddate'
	atc_level_on = dict_atc_levels[atc]
	st.subheader('ATC ontology for "%s"' % atc)
	st.write(data[data[atc_level_on] == atc].filter(regex='atc_').drop_duplicates())
	###BAR CHART
	st.subheader('Nombre de prescriptions de %s par mois' % atc)
	hist_values = np.histogram(data[data[atc_level_on] == atc][DATE_COLUMN].dt.month, bins=12, range=(0,12))[0]
	st.bar_chart(hist_values)
	###MAP
	# Some number in the range 0-12
	month_to_filter = st.slider('Mois', 0, 12, 6)
	filtered_data = data[(data[DATE_COLUMN].dt.month == month_to_filter)&(data[atc_level_on] == atc)]
	filtered_data = pd.merge(filtered_data,df_geometries[['id','lat','lon']],how = 'left',left_on = 'patientid',right_on = 'id')
	filtered_data = filtered_data[filtered_data.lat.isnull()==False]
	st.subheader('Carte des prescriptions de %s en %s/2018' % (atc,month_to_filter))
	st.map(filtered_data)
	### RATIO MAP
	##########################
	grouping_level = st.sidebar.selectbox("See prescribing by:", ['Prescribers','Cercles de qualité','Distributeurs'], 0) #Add a select box to choose the grouping level
	##########################
	versus_filter = st.sidebar.text_input('Filter versus name list').lower() #Add text box input for ATC filtering
	##########################
	versus =  st.sidebar.selectbox("ATC choices", atc_data[atc_data.nameen.str.contains(versus_filter)]['nameen'].sort_values().unique().tolist(), 0)
	##########################
	group_dict = {'Prescribers':'prescriberid','Distributeurs':'distributorid'}
if versus != '':

	atc_level_vs = dict_atc_levels[versus]

	y = data[data[atc_level_on] == atc].groupby(group_dict[grouping_level]).count().patientid/data[data[atc_level_vs] == versus].groupby(group_dict[grouping_level]).count().patientid
	y = pd.DataFrame(y.dropna().sort_values().reset_index(drop = True)).reset_index().rename(columns = {'index':'mpr','patientid':'fraction'})
	
	st.subheader('Number of patients that received %s per 1,000 patients that received %s' % (atc, versus))
	y['fraction'] = y['fraction']*1000
	st.bar_chart(y['fraction'].to_numpy())
	# c = alt.Chart(y).mark_bar().encode(x = alt.X('mpr:O',axis = alt.Axis(title = 'mpr')),y = alt.Y('fraction',axis = alt.Axis(title = 'Number of patients that received %s per 1,000 patients that received %s' % (atc, versus))))
	# st.altair_chart(c)
	buildings_ge = pd.read_pickle('./Data/buildings_ge.pkl')
	LAND_COVER  = [[[-74.0, 40.7], [-74.02, 40.7], [-74.02, 40.72], [-74.0, 40.72]]]
	material = {'ambient': 0.5,
  	'diffuse': 0.6,
  	'shininess': 40,
  	'specularColor': [60, 64, 70]}
	layer = pdk.Layer(
    	'HexagonLayer',
    	filtered_data[['lon','lat']],
    	get_position=['lon', 'lat'],
    	auto_highlight=True,
    	elevation_scale=20,
    	pickable=True,
    	elevation_range=[0, 100],
    	extruded=True,                 
    	coverage=1)
	scatter = pdk.Layer(
	    'ScatterplotLayer',     # Change the `type` positional argument here
	    filtered_data[['lon','lat']],
	    get_position=['lon', 'lat'],
	    auto_highlight=True,
	    get_radius=30,          # Radius is given in meters
	    get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
	    pickable=True)
	polygon = pdk.Layer(
	    'PolygonLayer',
	    LAND_COVER,
	    stroked=False,
	    # processes the data as a flat longitude-latitude pair
	    get_polygon='-',
	    get_fill_color=[180, 0, 0, 0],
	    pickable=True
	)
	# Set the viewport location
	view_state = pdk.ViewState(
	    longitude=filtered_data.lon.mean(),
	    latitude=filtered_data.lat.mean(),
	    zoom=10,
	    min_zoom=3,
	    max_zoom=30,
	    pitch=45,
	    bearing=-25)
	layers = [scatter,polygon]
	if st.sidebar.checkbox('Show buildings blueprint',key = 'Buildings'):
		buildings_ge = pd.read_pickle('./Data/buildings_ge.pkl')
		buildings = pdk.Layer(
	        'PolygonLayer',
	        buildings_ge,
	        extruded = True,
	        wireframe = False,
	        opacity =  0.5,
	        get_polygon = 'polygon',
	        get_elevation = 'elevation',
	        get_fill_color = [74, 80, 87],
	        material = material,
	        pickable=True
	      )
		layers = [scatter,polygon,buildings]
	st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state,mapbox_key = 'pk.eyJ1IjoiZGFkZXJpZGQiLCJhIjoiY2pmYmI4bTF2MjU0dDJ4bW1pdGFkaGpodSJ9.XhxTSKh9k5zQ1ysmB9g2gQ'))

