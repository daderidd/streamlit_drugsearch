#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from PIL import Image
import altair as alt
import pydeck as pdk
import numpy as np
from api_key import mapbox_key
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeckmapping
from importlib import reload
reload(pydeckmapping)
from shapely.geometry import Point
from pydeckmapping import build_map
import datetime
import sys
sys.path.append('/Users/david/Dropbox/PhD/Scripts/Spatial analyses')
import pyspace
reload(pyspace)
from pyproj import Transformer
transformer = Transformer.from_crs("epsg:2056", "epsg:4326")


st.image('https://reseau-delta.ch/assets/ci_content/images/logo.png',width = 180)
st.markdown(st.__version__)

st.title("Plateforme d'analyse des données du réseau de soins Delta")
text_intro = """ Les données analysées sur cette plateforme correspondent aux données Delta de l'année {} et portent sur plus de {} patients dont {} à Genève. Il y a {} prescriteurs dont {} MPR Delta, {} distributeurs et {} cercles. """
################################
###########LOAD DATA############
################################
@st.cache(allow_output_mutation=True)
def load_data(path,DATE_COLUMN = None):
	"""Load data into DataFrame"""
	data = pd.read_csv(path)
	lowercase = lambda x: str(x).lower()
	data.rename(lowercase, axis='columns', inplace=True)
	if DATE_COLUMN is not None:
		data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
	return data
@st.cache(allow_output_mutation=True)
def load_gdf(path):
	"""Load data into GeoDataFrame"""
	data = gpd.read_file(path)
	lowercase = lambda x: str(x).lower()
	data.rename(lowercase, axis='columns', inplace=True)
	return data
date = '20200318'
#####################################GENERAL DATASETS########################################### 
data_folder = Path("../Data").resolve()
buildings_ge = pd.read_pickle('../Data/buildings_ge.pkl')
drug_path = data_folder/'Clean_data'/'{}_drug.csv'.format(date)
geom_path = data_folder/'Clean_data'/'{}_geometries.geojson'.format(date)
patient_path = data_folder/'Clean_data'/'{}_patient.geojson'.format(date)
cercle_path = data_folder/'Clean_data'/'{}_cercle.csv'.format(date)
event_path = data_folder / 'Clean_data'/'{}_event.geojson'.format(date)
mpr_path = data_folder / 'Clean_data'/'{}_mpr.geojson'.format(date)
distributor_path = data_folder/'Clean_data/{}_distributor.geojson'.format(date)
prescriber_path = data_folder / 'Clean_data/{}_prescriber.geojson'.format(date)
provider_path = data_folder / 'Clean_data/{}_provider.geojson'.format(date)
animator_path = data_folder / 'Clean_data/{}_animator.geojson'.format(date)
prestation_path = data_folder/'Clean_data'/'{}_prestation.csv'.format(date)


data_load_state = st.text('Loading data...') # Create a text element and let the reader know the data is loading.
df_geometries = load_gdf(path = geom_path) #Import geometries
gdf_distributor = load_gdf(distributor_path)
gdf_prescriber = load_gdf(prescriber_path)
gdf_provider = load_gdf(provider_path)
gdf_animator = load_gdf(animator_path)
gdf_event = load_gdf(event_path)
gdf_mpr = load_gdf(mpr_path)
df_cercle = load_data(cercle_path)
df_drug = load_data(path = drug_path,DATE_COLUMN = 'delivereddate') #Load drug data
gdf_patient = load_gdf(path = patient_path) # Load patient data
data_load_state.text('Loading data...done!') # Notify the reader that the data was successfully loaded.
atc_data = load_data(path = '../Data/atc_nomenclature.csv') #Load the ATC nomenclature from WHO
df_atc_levels = pd.read_csv('../Data/atc_levels.csv') #Import ATC levels
cantons = gpd.read_file('/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp')
communes = gpd.read_file('/Users/david/Dropbox/PhD/Data/Databases/SITG/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp')
communes_ge = communes[communes.KANTONSNUM == 25]
MIP = ['indometacin','acemetacin','ketoprofen','phenylbutazon','piroxicam','meloxicam','etoricoxib','pethidin','chinidin','flecainid','sotalol','nitrofurantoin','zolpidem','estradiol','trimipamine','acémétacine','amiodarone']
#################################################################################################
################################## DATA PREPARATION #############################################
atc_data = atc_data.fillna('') 
gdf_mpr['mpr_yy_bth'] = gdf_mpr['mprbirthday'].str.split('.').str[2]
gdf_mpr['mpr_yy_entry'] = gdf_mpr['mprentrydate'].str.split('.').str[2]
gdf_mpr['mpr_yy_exit'] = gdf_mpr['mprexitdate'].str.split('.').str[2]
gdf_mpr = gdf_mpr.drop(['mprentrydate','mprbirthday','mprexitdate'],axis = 1).drop_duplicates()
gdf_mpr[['mpr_yy_bth','mpr_yy_entry','mpr_yy_exit']] = gdf_mpr[['mpr_yy_bth','mpr_yy_entry','mpr_yy_exit']].astype('float')
no_dupli = gdf_mpr.groupby(['id']).mean().reset_index()
no_dupli = no_dupli.drop(['e','n'],axis = 1)
gdf_mpr = gdf_mpr.drop(['mpr_yy_bth','mpr_yy_entry','mpr_yy_exit'],axis = 1).merge(no_dupli, on = 'id').drop_duplicates().reset_index()
gdf_mpr = gdf_mpr[['id','name','mprsex','mpr_yy_bth','mpr_yy_entry','mpr_yy_exit','e','n','geometry']].drop_duplicates(subset = ['id'])
gdf_mpr['age'] = 2018-gdf_mpr.mpr_yy_bth
gdf_mpr.loc[gdf_mpr.age > 200, 'age'] = 65 ###To be changed (better to change in Data Preparation and replace yy_bth before age calculation)
gdf_mpr.loc[gdf_mpr.age < 0,'age'] = np.nan
bins = [30, 45, 60, 75]
gdf_mpr['age_cat'] = pd.cut(gdf_mpr['age'], bins)
dict_atc_levels= dict(zip(df_atc_levels.atc, df_atc_levels.level)) 
gdf_event_cercle = pd.merge(gdf_event,df_cercle, left_on = 'id',right_on = 'eventid', how = 'left')
uniq_cercle_geom = gdf_event_cercle.drop_duplicates(subset = 'circlename',keep='first').reset_index(drop = True)
# uniq_cercle_geom['longitude'],uniq_cercle_geom['latitude'] = uniq_cercle_geom.to_crs(epsg = 4326).geometry.x,uniq_cercle_geom.to_crs(epsg = 4326).geometry.y
uniq_cercle_geom[['latitude','longitude']]= uniq_cercle_geom.apply(lambda x: transformer.transform(x.e,x.n),axis = 1,result_type = 'expand')

geojson_file_CQ = '../Data/CQ_polygons.geojson'
bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115]
gdf_patient['age_cat'] =  pd.cut(gdf_patient['age'], bins)
#################################################################################################
################################ PERSONAL ID ##############################################
st.sidebar.markdown('## Insert your personal ID')

st.sidebar.markdown('Example : 40e8ac4dbc023d86a815f8476a1884e4')

personal_id = st.sidebar.text_input('Personal ID')
if personal_id not in df_drug.prescriberid.values:
	st.sidebar.markdown('### *Invalid ID*')
else:
	st.sidebar.markdown('### *Valid ID*')

colors_id = {}
for i in df_drug.prescriberid.values:
	colors_id[i] = 'blue'
	if i == personal_id:
		colors_id[i] = 'red'

#################################################################################################
################################ INTRODUCTION TEXT ##############################################
text_intro = text_intro.format(2018,gdf_patient.id.nunique(),gdf_patient[gdf_patient.networkname == 'Delta Genève'].id.nunique(),df_drug.prescriberid.nunique(),df_drug[df_drug.mpr_delta == 1].prescriberid.nunique(),df_drug.distributorid.nunique(),52)
st.markdown(text_intro)
st.sidebar.markdown('# *Analyses*')

#################################################################################################
################################ SHOW RAW DATA ##################################################
if st.sidebar.checkbox('Show raw prescription data',key = 'Drug prescription data'):
    st.subheader('Raw data')
    st.write(df_drug.head())
#################################################################################################
#################################AGE FILTERING ##################################################
st.markdown('### Age filtering')
age_filter = st.slider("Patient Age", 0, 110, (25, 75), step = 5)

patients = gdf_patient[(gdf_patient.age >= age_filter[0])&(gdf_patient.age <= age_filter[1])].id.values
filtered_drug = df_drug[df_drug.patientid.isin(patients)]
filtered_drug = pd.merge(filtered_drug,df_geometries[['id','lat','lon']],how = 'left',left_on = 'patientid',right_on = 'id')
filtered_drug = filtered_drug[filtered_drug.lat.isnull()==False]
filtered_drug = filtered_drug.rename(columns = {'lon' : 'longitude','lat':'latitude'})
filtered_drug['month_delivery'] = filtered_drug['delivereddate'].dt.month_name()
filtered_drug['month_delivery'] = pd.Categorical(filtered_drug['month_delivery'], ['January','February',"March", "April",'May','June','July','August','September','October','November', "December"])


st.markdown('### Time period selection')

start_date = str(st.date_input('Start date',  datetime.date(2018, 1, 1)))
end_date = str(st.date_input('End date', filtered_drug.delivereddate.max()))
if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')
filtered_drug = filtered_drug[(filtered_drug.delivereddate >start_date)&(filtered_drug.delivereddate < end_date)]
#################################################################################################
if st.sidebar.checkbox('Time and space distribution of a drug',key = 'Time space drug A'):
	st.sidebar.markdown('## Drug group A')
	atc_filter = st.sidebar.text_input('Filter ATC name list').lower() #Add text box input for ATC filtering
	atc_on = st.sidebar.selectbox("ATC choices", atc_data[atc_data.nameen.str.contains(atc_filter)]['nameen'].sort_values().unique().tolist(), 0) #Add a select box to choose an ATC among the filtered ones
	if atc_on != '':
		DATE_COLUMN = 'delivereddate'
		atc_level_on = dict_atc_levels[atc_on]
		st.subheader('ATC ontology for "%s"' % atc_on)
		st.write(filtered_drug[filtered_drug[atc_level_on] == atc_on].filter(regex='atc_').drop_duplicates())
		################################# BAR CHART ###################################################
		st.subheader('Nombre de prescriptions de %s par mois' % atc_on)
		hist_values = pd.DataFrame(filtered_drug[filtered_drug[atc_level_on] == atc_on].month_delivery.value_counts().sort_index()).reset_index()
		# st.bar_chart(hist_values,height = 300)
		fig = px.bar(hist_values, x='index', y='month_delivery',
             hover_data=['month_delivery'], color='month_delivery', color_continuous_scale=[(0, "rgb(222,235,247)"), (0.5, "rgb(158,202,225)"), (1, "rgb(49,130,189)")],
             labels={'month_delivery':'Number of prescriptions'}, height=600,width = 780)
		st.plotly_chart(fig)
		###MAP
		# Some number in the range 0-12
		month_to_filter = st.slider('Mois', 1, 12, 6)
		filtered_drug_atc = filtered_drug[(filtered_drug[DATE_COLUMN].dt.month == month_to_filter)&(filtered_drug[atc_level_on] == atc_on)]

		mapstyle = st.sidebar.radio("Mapping style",('Simple', 'Advanced'))
		st.markdown('## Geographic distribution')
		st.subheader('Carte des prescriptions de %s au %s/2018' % (atc_on,month_to_filter))
		if mapstyle == 'Simple':
			st.map(filtered_drug_atc)
		if mapstyle == 'Advanced':
			st.sidebar.markdown('### Map Options')
			##MAP OPTIONS
			show_buildings = False
			show_grid = False
			# show_geojson = False
			if st.sidebar.checkbox('Show buildings blueprint',key = 'Buildings'):
				show_buildings = True
			if st.sidebar.checkbox('Show grid layer',key = 'Grid'):
				show_grid = True
			layers, view_state = build_map(filtered_drug_atc[['longitude','latitude']],buildings_ge,geojson_file_CQ,show_buildings = show_buildings,show_grid = show_grid)
			r = pdk.Deck(map_style = 'mapbox://styles/mapbox/light-v9',layers=layers, initial_view_state=view_state,mapbox_key = mapbox_key())
			st.pydeck_chart(r)
		##########################
		grouping_level = st.sidebar.selectbox("See prescribing by:", ['Prescribers','Distributors'], 0,key = 'group level') #Add a select box to choose the grouping level
		group_dict = {'Prescribers':'prescriberid','Distributors':'distributorid'}
		group_var = group_dict[grouping_level]
		##########################
		st.sidebar.markdown('## Drug group B')
		versus_filter = st.sidebar.text_input('Filter versus name list').lower() #Add text box input for ATC filtering
		atc_vs =  st.sidebar.selectbox("ATC choices", atc_data[atc_data.nameen.str.contains(versus_filter)]['nameen'].sort_values().unique().tolist(), 0,key = 'atc_choice')
		##########################
		if atc_on in MIP: 
			st.markdown('# Médicaments Potentiellement Inadéquats (MIP)')
			st.markdown("""Pour la définition des médicaments potentiellement inadéquats, la liste Beers et PRISCUS a été opérationnalisée en fonction des limitations données. Les recommandations font référence aux "personnes âgées". Selon le principe actif et l'affection, la limite d'âge selon les listes peut varier légèrement. Dans ce contexte, le dénominateur commun est l'application des règles à tous les assurés qui ont atteint l'âge de 65 ans au cours du trimestre concerné.
	En principe, les deux listes contiennent des catégories similaires pour l'évaluation d'une substance active :
	La liste des bières distingue essentiellement 9 critères différents qui définissent la MIP : 1. l'ATC est toujours le MIP
	2. seule la préparation à courte durée d'action du médicament est une MIP (donc pas de produits retardés) 3. seules les doses élevées sont des MIP
	4. seul l'usage à long terme est PIM
	5. pas de MIP en cas de certains diagnostics
	6. uniquement les MIP avec une forme de dosage spéciale (par exemple, orale) 7. seuls les médicaments spéciaux sont des MIP
	8. seulement PIM si sans utilisation simultanée d'autres substances actives
	9. la MIP si des critères cliniques sont présents.
	Outre ces classifications, il existe également des médicaments qui doivent répondre à plusieurs critères d'exigence ou combinaisons de conditions pour être considérés comme des MIP. Ces groupes sont les suivants : 2/6, 3/9, 4/9, 6/7, 8/4 et 8/3/4.
	Comme il n'y a pas de diagnostics et de critères cliniques dans les données d'Helsana,
	 les groupes 5 et 9 ne sont pas utilisés, c'est-à-dire que ces MIP 
	 ne peuvent pas être déterminées. Dans le groupe 7 également, aucun médicament 
	 n'est défini pour cette évaluation. Le groupe 8 n'apparaît qu'en combinaison avec d'autres groupes. 
	 Le groupe 4 n'est utilisé qu'avec les bières Version 2015 en combinaison avec les valeurs cliniques. 
	 Le groupe 2 n'est actuellement pas pertinent sur le marché suisse.""")
		st.markdown('## Prescription of {} vs {}'.format(atc_on, atc_vs))
		if atc_vs != '':
			atc_level_vs = dict_atc_levels[atc_vs]
			show_by = st.radio("Show by :",('Patient', 'Item'))
			if show_by == 'Patient':
				if st.sidebar.checkbox('Only Delta MPR',key = 'MPR'):
					df_onvs_frac = filtered_drug[[group_var,atc_level_on,'patientid']][(filtered_drug['mpr_delta']==1)&(filtered_drug[atc_level_on] == atc_on)].groupby(group_var).patientid.nunique() / filtered_drug[[group_var,atc_level_vs,'patientid']][(filtered_drug['mpr_delta']==1)&(filtered_drug[atc_level_vs] == atc_vs)].groupby(group_var).patientid.nunique()
				else:
					df_onvs_frac = filtered_drug[[group_var,atc_level_on,'patientid']][filtered_drug[atc_level_on] == atc_on].groupby(group_var).patientid.nunique() / filtered_drug[[group_var,atc_level_vs,'patientid']][filtered_drug[atc_level_vs] == atc_vs].groupby(group_var).patientid.nunique()
				df_onvs_frac = pd.DataFrame(df_onvs_frac.dropna().sort_values()).rename(columns = {'patientid':'fraction'}).reset_index()
				title = 'Number of patients that received {} per <br> 1,000 patients that received {}'.format(atc_on, atc_vs)
			if show_by == 'Item':
				if st.sidebar.checkbox('Only Delta MPR',key = 'MPR'):
					df_onvs_frac = filtered_drug[[group_var,atc_level_on]][(filtered_drug['mpr_delta']==1)&(filtered_drug[atc_level_on] == atc_on)].groupby(group_var).count()[atc_level_on] / filtered_drug[[group_var,atc_level_vs,]][(filtered_drug['mpr_delta']==1)&(filtered_drug[atc_level_vs] == atc_vs)].groupby(group_var).count()[atc_level_vs]
				else:
					df_onvs_frac = filtered_drug[[group_var,atc_level_on]][filtered_drug[atc_level_on] == atc_on].groupby(group_var).count()[atc_level_on] / filtered_drug[[group_var,atc_level_vs]][filtered_drug[atc_level_vs] == atc_vs].groupby(group_var).count()[atc_level_vs]
				df_onvs_frac = pd.DataFrame(df_onvs_frac.dropna().sort_values()).rename(columns = {0:'fraction'}).reset_index()
				title = 'Items for {} per 1,000 {}'.format(atc_on, atc_vs)
			df_onvs_frac['fraction'] = df_onvs_frac['fraction']*1000
			fig = go.Figure(data=[go.Bar(x=df_onvs_frac[group_var], y=df_onvs_frac['fraction'])])
			fig.update_traces(marker_color=df_onvs_frac[group_var].map(colors_id).values, marker_line_width=0, opacity=1)
			xaxis_title = "ID of the {}".format(grouping_level[:-1])
			fig.update_layout(xaxis_title=xaxis_title, yaxis_title= 'n', width=780, height=600,title={'text':title})
			st.plotly_chart(fig)
			if grouping_level == 'Prescribers':
				df_onvs_frac_by_cercle = pd.DataFrame(pd.merge(df_onvs_frac,gdf_event_cercle, left_on = 'prescriberid', right_on = 'participantid',how = 'left').groupby('circlename').mean().fraction.sort_values()).reset_index()
				fig = px.bar(df_onvs_frac_by_cercle, x='circlename', y='fraction',hover_data=['fraction'], color='fraction',labels={'fraction':'n','circlename':'Name of the Quality Circle'}, width=780, height=600)
				st.plotly_chart(fig)

				df_onvs_frac_by_cercle = gpd.GeoDataFrame(pd.merge(df_onvs_frac_by_cercle,uniq_cercle_geom[['circlename','longitude','latitude','networkname','geometry']],on = 'circlename',how = 'left'))
				ax = df_onvs_frac_by_cercle[df_onvs_frac_by_cercle.networkname == 'Delta Genève'].plot('fraction',legend = True,cmap = 'magma',markersize = 90,figsize = (8,5))
				df_onvs_frac_by_cercle[df_onvs_frac_by_cercle.networkname == 'Delta Genève'].apply(lambda x: ax.annotate(s=x.circlename, xy=x.geometry.centroid.coords[0], ha='center',size = 2),axis=1);
				communes_ge.plot(ax = ax,alpha = 0.2,color = 'lightgrey')
				ax.set_axis_off()
				st.pyplot(height = 800,dpi = 800)
				# layers, view_state = build_map(df_onvs_frac_by_cercle[['longitude','latitude']],buildings_ge,geojson_file_CQ,show_buildings = False,show_grid = False,show_geojson = True)
				# r = pdk.Deck(map_style = 'mapbox://styles/mapbox/light-v9',layers=layers,tooptip = True, initial_view_state=view_state,mapbox_key = mapbox_key())
				# st.write(r)


if st.sidebar.checkbox('Generic and original drugs analysis',key = 'Generic drug'):
	atc_filter = st.sidebar.text_input('Filter ATC name list',key = 'atc 2').lower() #Add text box input for ATC filtering
	atc_on_generic = st.sidebar.selectbox("ATC choices", atc_data[atc_data.nameen.str.contains(atc_filter)]['nameen'].sort_values().unique().tolist(),key = 'atc 2 selectbox') #Add a select box to choose an ATC among the filtered ones
	if atc_on_generic != '':
		atc_generic_level_on = dict_atc_levels[atc_on_generic]

		st.markdown('# Generic drugs usage')
		grouping_level = st.sidebar.selectbox("See prescribing by:", ['Prescribers','Distributors'],0,key = 'group level drug') #Add a select box to choose the grouping level
		group_dict = {'Prescribers':'prescriberid','Distributors':'distributorid'}
		group_var = group_dict[grouping_level]
		generic_status = filtered_drug[['drugatcname','druggeneric']].drop_duplicates().dropna()
		drugs_with_generic = generic_status[generic_status.drugatcname.duplicated()].sort_values('drugatcname').dropna().drugatcname.unique()
		prescriptions_gene_orig = filtered_drug[filtered_drug.drugatcname.isin(drugs_with_generic)]
		# drugs_to_study = prescriptions_gene_orig[['drugatcname','atc_lvl1','atc_lvl2','atc_lvl3','atc_lvl4','atc_lvl5']].drop_duplicates()
		# st.write(drugs_to_study.head())
		prescriptions_gene_orig_nonull = prescriptions_gene_orig[prescriptions_gene_orig.druggeneric.isnull()==False]
		ratio_gene_orig = pd.DataFrame(prescriptions_gene_orig_nonull.groupby([group_var,atc_generic_level_on,'druggeneric']).drugname.count()).unstack().reset_index().fillna(0)
		st.write(ratio_gene_orig)
		ratio_gene_orig.columns = [group_var,atc_generic_level_on,'Générique','Original']
		ratio_gene_orig['total'] = ratio_gene_orig[['Générique','Original']].sum(axis = 1)
		ratio_gene_orig['perc_generique'] = ((ratio_gene_orig['Générique']/ratio_gene_orig[['Original','Générique']].sum(axis = 1))*100).round(1)
		ratio_gene_orig['perc_original'] = ((ratio_gene_orig['Original']/ratio_gene_orig[['Original','Générique']].sum(axis = 1))*100).round(1)

		drug_gene_orig_ratio = ratio_gene_orig[ratio_gene_orig[atc_generic_level_on] == atc_on_generic].sort_values('perc_generique')

		if grouping_level == 'Prescribers':
		    prescription_info = pd.merge(gdf_prescriber,drug_gene_orig_ratio,left_on = 'id',right_on = group_var)
		    if st.sidebar.checkbox('Only Delta MPR',key = 'MPR'):
		        prescription_info = prescription_info[prescription_info.mpr_delta == 1]
		if grouping_level == 'Distributors':
		    prescription_info = pd.merge(gdf_distributor,drug_gene_orig_ratio,left_on = 'id',right_on = group_var)
		max_prescription = int(prescription_info.total.max())
		min_prescription = int(prescription_info.total.min())

		n_prescri_filter = st.slider("Number of prescriptions delivered by {}".format(grouping_level[:-1]), min_prescription,max_prescription, (min_prescription, max_prescription), step = 5)
		prescription_info = prescription_info[(prescription_info.total >= n_prescri_filter[0]) & (prescription_info.total <= n_prescri_filter[1]) ]
		####
		barplot_option = st.sidebar.selectbox('Show bar plot with: ',('Absolute values', 'Percentages'))
		####
		if barplot_option == 'Absolute values':
			title = 'Number of "Generic" and "Original" {} prescriptions by {}'.format(atc_on_generic,grouping_level)
			labels= prescription_info.sort_values('total')[group_var].values
			fig = go.Figure(data=[    
			    go.Bar(name='Generic', x=labels, y=prescription_info.sort_values('total').Générique.values),
			    go.Bar(name='Original', x=labels, y=prescription_info.sort_values('total').Original.values)])
			xaxis_title = "ID of the {}".format(grouping_level[:-1])
			fig.update_layout(barmode='stack',xaxis_title=xaxis_title, yaxis_title= 'n', width=780, height=600,title={'text':title})
			# Change the bar mode
		if barplot_option == 'Percentages':
			title = 'Percentage of "Generic" and "Original" {} prescriptions by {}'.format(atc_on_generic,grouping_level)
			labels= prescription_info[group_var].values
			fig = go.Figure(data=[
			go.Bar(name='Générique', x=labels, y=prescription_info.sort_values('perc_generique').perc_generique.values),
			go.Bar(name='Original', x=labels, y=prescription_info.sort_values('perc_generique').perc_original.values)])
			xaxis_title = "ID of the {}".format(grouping_level[:-1])
			fig.update_layout(barmode='stack',xaxis_title=xaxis_title, yaxis_title= '%', width=780, height=600,title={'text':title})
		# fig.update_traces(marker_color=prescription_info[group_var].map(colors_id).values, marker_line_width=0, opacity=1)
		st.plotly_chart(fig)
		if grouping_level == 'Prescribers':

			drug_gene_orig_per_cq = pd.DataFrame(pd.merge(prescription_info,gdf_event_cercle[['participantid','circlename']].drop_duplicates(), left_on = 'prescriberid', right_on = 'participantid',how = 'left').groupby('circlename').mean().perc_generique.sort_values()).reset_index()
			fig = px.bar(drug_gene_orig_per_cq, x='circlename', y='perc_generique',hover_data=['perc_generique'], color='perc_generique',labels={'perc_generique':'%','circlename':'Name of the Quality Circle'}, width=780, height=600)
			st.plotly_chart(fig)    

		prescription_info[['lat','lon']]= prescription_info.apply(lambda x: transformer.transform(x.e,x.n),axis = 1,result_type = 'expand')
		prescription_info = pyspace.add_random_noise(prescription_info)

		distance = st.slider('Distance',100,2000, value = 1200, step = 100)
		prescription_info,weights = pyspace.get_distanceBandW(prescription_info,distance)
		getis = pyspace.compute_getis(prescription_info,'perc_generique',weights,9999,0.05,star = True)
		colors_cl = {'Cold Spot - p < 0.01':'#2166ac', 'Cold Spot - p < 0.05':'#67a9cf','Cold Spot - p < 0.1':'#d1e5f0', 'Hot Spot - p < 0.01':'#b2182b','Hot Spot - p < 0.05':'#ef8a62','Hot Spot - p < 0.1':'#fddbc7','Not significant':'#bdbdbd'}
		prescription_info['perc_generique_G_cl'] = pd.Categorical(prescription_info['perc_generique_G_cl'], ['Cold Spot - p < 0.01','Cold Spot - p < 0.05','Cold Spot - p < 0.1','Hot Spot - p < 0.01','Hot Spot - p < 0.05','Hot Spot - p < 0.1','Not significant'])
		px.set_mapbox_access_token(mapbox_key())
		specialty = group_var[:-2]+'specialty'
		fig = px.scatter_mapbox(prescription_info.sort_values('perc_generique_G_cl'), lat="lat", lon="lon",hover_data = [specialty,'perc_generique'],color = 'perc_generique_G_cl',
		                        color_discrete_map = colors_cl,size = 'perc_generique', size_max=10, zoom=8)
		st.plotly_chart(fig)

		fig = px.scatter_mapbox(prescription_info.sort_values('perc_generique'), lat="lat", lon="lon",color = 'perc_generique',size = 'perc_generique', size_max=10, zoom=8)
		st.plotly_chart(fig)
st.markdown('# Polymedication')
st.markdown("""La polymédication est souvent définie comme l'administration ou la prise de 5 médicaments ou 
	agents différents ou plus. Plus les médicaments sont combinés, plus le 
	risque d'effets secondaires indésirables est élevé. Par 
	exemple, le risque d'interaction est déjà de 38% pour 4 médicaments différents, alors 
	qu'il passe à 82% s'il y a 7 médicaments différents ou plus (Blozik, et al., 2013).

	### Définition de la polymédication dans l'ensemble des données d'analyse

	Pour l'identification d'un assuré avec polymédication, les codes ATC réglés 
	par assuré sont comptés dans un trimestre d'évaluation. Si un code ATC à moins de 
	7 chiffres est trouvé dans un autre code ATC à plus de chiffres, il n'est compté 
	qu'une fois par assuré et par trimestre. Pour un drapeau positif de polymédication,
	 6 codes ATC différents ou plus doivent avoir été comptabilisés avec un assuré.""")				
if st.checkbox('Only Delta MPR',key = 'MPR2'):
	df_onvs_meanatc_perpatient = pd.DataFrame(filtered_drug[(filtered_drug['mpr_delta']==1)].groupby(['prescriberid','patientid']).drugatcname.nunique().groupby('prescriberid').mean().sort_values()).reset_index()
	#################
	patient_n_atc = pd.DataFrame(filtered_drug[(filtered_drug['mpr_delta']==1)].groupby(['patientid']).drugatcname.nunique()).reset_index()
	patient_n_atc.columns = ['patientid','n_cat']
	patient_n_atc = pd.merge(gdf_patient[['id','age_cat']],patient_n_atc, left_on = 'id',right_on = 'patientid', how = 'left').drop('patientid',axis = 1)
	patient_n_atc['n_cat']= patient_n_atc['n_cat'].fillna(0)
	patient_n_atc.loc[patient_n_atc.n_cat == 0, 'atc_cat'] = '0. No medication'
	patient_n_atc.loc[patient_n_atc.n_cat > 4, 'atc_cat'] = '2. 5 or more medications'
	patient_n_atc.loc[(patient_n_atc.n_cat < 5)& (patient_n_atc.n_cat > 0), 'atc_cat'] = '1. 1-4 medications'
	patient_n_atc['age_cat'] = patient_n_atc['age_cat'].astype(str)
	patient_n_atc = pd.DataFrame(patient_n_atc.groupby(['age_cat','atc_cat']).size().mul(100)/patient_n_atc.groupby(['age_cat']).size()).reset_index()
	patient_n_atc.columns = ['age_cat','atc_cat','perc']
	patient_n_atc['perc'] = patient_n_atc['perc'].round(1)
	patient_n_atc['age_cat'] = pd.Categorical(patient_n_atc['age_cat'], ['(0.0, 5.0]','(5.0, 10.0]', '(10.0, 15.0]',
       '(15.0, 20.0]', '(20.0, 25.0]', '(25.0, 30.0]', '(30.0, 35.0]',
       '(35.0, 40.0]', '(40.0, 45.0]', '(45.0, 50.0]',
       '(50.0, 55.0]', '(55.0, 60.0]', '(60.0, 65.0]', '(65.0, 70.0]',
       '(70.0, 75.0]', '(75.0, 80.0]', '(80.0, 85.0]', '(85.0, 90.0]',
       '(90.0, 95.0]', '(95.0, 100.0]','(100.0, 105.0]', '(105.0, 110.0]', 'nan'])
	patient_n_atc = patient_n_atc.sort_values(['age_cat','atc_cat'])
	fig = px.bar(patient_n_atc, x="age_cat", y="perc", color='atc_cat', barmode='group',
             height=400,title = 'Proportion de personnes assurées en fonction du nombre de médicaments',labels={'age_cat':'Age category','perc':'Percentage (%)'})
	st.plotly_chart(fig)
else:
	df_onvs_meanatc_perpatient = pd.DataFrame(filtered_drug.groupby(['prescriberid','patientid']).drugatcname.nunique().groupby('prescriberid').mean().sort_values()).reset_index()
title = 'Mean number of unique ATC prescribed by patient for each prescriber'
fig = go.Figure(data=[go.Bar(x=df_onvs_meanatc_perpatient['prescriberid'], y=df_onvs_meanatc_perpatient['drugatcname'])])
fig.update_traces(marker_color='rgb(8,48,107)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=1)
fig.add_shape(
        # Line Horizontal
            type="line",
            x0=0,
            y0=6,
            x1=len(df_onvs_meanatc_perpatient['prescriberid'].values),
            y1=6,
            line=dict(
                color="LightSeaGreen",
                width=4,
                dash="dashdot",
            ),
    )
xaxis_title = "ID of the Prescriber"
fig.update_layout(xaxis_title=xaxis_title, yaxis_title= 'n', width=780, height=600,title={'text':title})
st.plotly_chart(fig)
##########################
df_onvs_meanatc_perpatient = pd.merge(df_onvs_meanatc_perpatient,gdf_mpr[['id','mprsex','age_cat']], left_on = 'prescriberid',right_on = 'id',how = 'left')
df_onvs_meanatc_perpatient.age_cat = df_onvs_meanatc_perpatient.age_cat.astype(str)
##########################
title = 'Number of unique ATC prescribed by patient by MPR age category and sex'
fig = px.box(df_onvs_meanatc_perpatient.dropna().sort_values('age_cat'), x="age_cat", y="drugatcname", color="mprsex",title = title,labels={'age_cat':'Age category','drugatcname':'Number of prescriptions by patient'})
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
st.plotly_chart(fig)