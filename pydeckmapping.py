import pydeck as pdk
import geopandas as gpd
import pandas as pd

def build_map(data,buildings_data,geojson_file,show_buildings = False, show_grid = False,show_geojson = False):
    LAND_COVER = [[[6.33, 46.11], [6.33, 46.35], [5.93, 46.35], [5.93, 46.11]]]
    material = {'ambient': 0.5,
    	'diffuse': 0.6,
    	'shininess': 40,
    	'specularColor': [60, 64, 70]}
    scatter = pdk.Layer(
        'ScatterplotLayer',     # Change the `type` positional argument here
        data,
        get_position=['longitude', 'latitude'],
        auto_highlight=True,
        get_radius=30,          # Radius is given in meters
        get_fill_color=[180, 0, 0, 140],  # Set an RGBA value for fill
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
        longitude=6.14909,
        latitude=46.193,
        zoom=10,
        min_zoom=3,
        max_zoom=30,
        pitch=45,
        bearing=-25)
    layers = [scatter,polygon]
    if show_geojson == True:
        geojson = pdk.Layer(
            'GeoJsonLayer',
            geojson_file,
            opacity = 0.8,
            stroked = False,
            filled = True,
            extruded = True,
            wireframe = True,
            get_elevation = 'properties.fraction',
            get_fill_color = 'properties.fraction',
            get_line_color = [255,255,255],
            pickable = True)
        layers.append(geojson)
    if show_grid == True:
        grid_layer = pdk.Layer(
            'GridLayer',
            data,
            get_position=['longitude', 'latitude'],
            auto_highlight=True,
            cell_size = 800,
            elevation_scale=50,
            pickable=True,
            elevation_range=[0, 300],
            extruded=True,                 
            coverage=1,
            on_hover = 'count')
        layers.append(grid_layer)
    if show_buildings == True:
    	buildings = pdk.Layer(
            'PolygonLayer',
            buildings_data,
            extruded = True,
            wireframe = False,
            opacity =  0.8,
            get_polygon = 'polygon',
            get_elevation = 'elevation',
            get_fill_color = [74, 80, 87],
            material = material,
            pickable=True
          )
    	layers.append(buildings)

    return layers,view_state

