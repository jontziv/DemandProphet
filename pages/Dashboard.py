#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="DataCo Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################
# Load data
df1= pd.read_csv('data/DATACOFinal_2R1.csv')
df2= pd.read_csv('data/DATACOFinal_2R2.csv')
df3= pd.read_csv('data/DATACOFinal_2R3.csv')
df_init=pd.concat([df1, df2,df3], 
                  axis = 1)

#df_init = pd.read_csv('data/DATACOFinal_2R.csv')
df_choropleth = pd.read_csv('data/choropleth.csv')

#######################


#######################
# Plots

# Heatmap
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900,height=400
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    
    return heatmap

# Choropleth map
def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column,locationmode='country names',
                               color_continuous_scale=input_color_theme,
                               range_color=(0, max(input_df.OrderSumOCountry)),
                               labels={'OrderSumOCountry':'OrderSumOCountry'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth


# Donut chart
def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=55, cornerRadius=35).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=150, height=150)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=55, cornerRadius=35).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=150, height=150)
  return plot_bg + plot + text

# Calculation year-over-year population migrations
def calculate_ordersum(input_df, input_year):
  selected_year_data = input_df[input_df['Delivery_Year'] == input_year].reset_index()
  previous_year_data = input_df[input_df['Delivery_Year'] == input_year - 1].reset_index()
  selected_year_data['revenue_difference'] = selected_year_data.OrderSumOCountry.sub(previous_year_data.OrderSumOCountry, fill_value=0)
  selected_year_data['locations']=selected_year_data.OrderSumOCountry
  return pd.concat([selected_year_data.OrderCountry, selected_year_data.OrderSumOCountry,selected_year_data.locations, selected_year_data.revenue_difference], axis=1).sort_values(by="OrderSumOCountry", ascending=False)

def cal_ordersum_choropl(input_df, input_year):
  selected_year_data = input_df[input_df['Delivery_Year'] == input_year].reset_index()
  previous_year_data = input_df[input_df['Delivery_Year'] == input_year - 1].reset_index()
  selected_year_data['locations']=input_df['alpha-3']
  selected_year_data['OrderCountry']=input_df['name_y']
  return pd.concat([selected_year_data.OrderCountry, selected_year_data.OrderSumOCountry,selected_year_data.locations,selected_year_data.Delivery_Year], axis=1).sort_values(by="OrderSumOCountry", ascending=False)

# Calculation KPIs
def calculate_kpis(input_df, input_year):
  selected_year_data = input_df[input_df['Delivery_Year'] == input_year].reset_index()
  previous_year_data = input_df[input_df['Delivery_Year'] == input_year - 1].reset_index()
  ontimestat=['Shipping on time','Advance shipping']
  selected_year_Ontimedata=input_df[(input_df['Delivery_Year'] == input_year) & (input_df['DeliveryStatus'].isin(ontimestat))]['DelStatus'].unique().sum()
  selected_year_Cancelleddata=input_df[(input_df['Delivery_Year'] == input_year) & (input_df['DeliveryStatus']=='Shipping canceled')]['DelStatus'].unique().sum()
  selected_year_Latedata=input_df[(input_df['Delivery_Year'] == input_year) & (input_df['DeliveryStatus']=='Late delivery')]['DelStatus'].unique().sum()
  return selected_year_Ontimedata,selected_year_Cancelleddata,selected_year_Latedata

#######################
# Dashboard Main Panel
col = st.columns((4.5, 9, 4.5), gap='large')

with col[0]:
    st.title(':bar_chart: DataCo Dashboard')
    
    year_list = list(df_init['Delivery_Year'].dropna().astype(int).unique())
    
    selected_year = st.selectbox('Select a year:', year_list)
    df_selected_year = df_init[df_init['Delivery_Year'] == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="OrderSumOCountry", ascending=False)


    df_selected_year_dist=df_init.drop_duplicates(subset=['OrderSumOCountry','OrderCountry'], keep="first")
    df_selected_year_sorted_dist = df_selected_year_dist.sort_values(by="OrderSumOCountry", ascending=False)


    color_theme_list = ['blues', 'greens', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme:', color_theme_list)

    projection=['equirectangular','mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa']
    selected_projection_theme = st.selectbox('Select a projection:', projection)


    st.markdown('#### Best/Lower Performing Market')

    df_ordersum = calculate_ordersum(df_init, selected_year)
    df_kpis=calculate_kpis(df_init, selected_year)

    if selected_year > 2010:
        first_state_name = df_ordersum.OrderCountry.iloc[0]
        first_state_revenue = df_ordersum.OrderSumOCountry.iloc[0]
        first_state_delta = df_ordersum.revenue_difference.iloc[0]
    else:
        first_state_name = '-'
        first_state_population = '-'
        first_state_delta = ''
    st.metric(label=first_state_name, value=first_state_revenue, delta=first_state_delta)

    if selected_year > 2010:
        last_state_name = df_ordersum.OrderCountry.iloc[-1]
        last_state_population = df_ordersum.OrderSumOCountry.iloc[-1]   
        last_state_delta = df_ordersum.revenue_difference.iloc[-1] 
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    


    if selected_year > 2010:    

        Ontime1 = df_kpis[0]
        Late1=df_kpis[1]
        Cancelled1=df_kpis[2]
        donut_chart_ontime = make_donut(Ontime1, 'Orders Ontime', 'green')
        donut_chart_late = make_donut(Late1, 'Orders Late', 'red')
        donut_chart_cancelled = make_donut(Cancelled1, 'Orders Cancelled','blue')
    else:
        Ontime1 = 0
        Late1 = 0
        Cancelled1=0
        donut_chart_ontime = make_donut(Ontime1, 'Orders Ontime', 'green')
        donut_chart_late = make_donut(Late1, 'Orders Late', 'red')
        donut_chart_cancelled = make_donut(Cancelled1, 'Orders Cancelled','blue')




with col[1]:
    st.markdown('#### Net Revenue')
    choropleth_data = cal_ordersum_choropl(df_choropleth, selected_year)
    choropleth = px.choropleth(choropleth_data, 
                    locations="locations", 
                    color="OrderSumOCountry", 
                    hover_name="OrderCountry", 
                    animation_frame="Delivery_Year",
                    projection=selected_projection_theme,  
                    title="Revenue per Country",
                    labels={"OrderSumOCountry": "Rev per Country"},
                    color_continuous_scale="Viridis")
    st.plotly_chart(choropleth, use_container_width=True)
    
    heatmap = make_heatmap(df_init, 'Delivery_Year', 'OrderCountry', 'OrderSumOCountry', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)


    st.markdown('#### On Time/Late/Cancelled')
    col4, col5, col6 = st.columns([3,3,3])
    with col4:
        st.header("Ontime")
        st.altair_chart(donut_chart_ontime)

    with col5:
        st.header("Late")
        st.altair_chart(donut_chart_late)

    with col6:
        st.header("Cancelled")
        st.altair_chart(donut_chart_cancelled)
    
with col[2]:
    st.markdown('#### Top Markets')

    st.dataframe(df_selected_year_sorted_dist,
                 column_order=("OrderCountry", "OrderSumOCountry"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "OrderCountry": st.column_config.TextColumn(
                        "OrderCountry",
                    ),
                    "OrderSumOCountry": st.column_config.ProgressColumn(
                        "OrderSumOCountry",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_year_sorted_dist.OrderSumOCountry),
                     )}
                 )
    
    with st.expander('About', expanded=True):
        st.write('''
            - :orange[**Best/Lower performing Market**]: States the market with the highest final order value and lowest respectively.Underneath we see the delta to previous year
            - :orange[**On Time/Late/Cancelled**]: percentage of orders that have been delivered or with advanced shipping are classed Ontime, Late and Cancelled are given based on the status from the raw data.
            - :orange[**Parametrization is enabled by the user sepcifying Year, Map Type and colorization of heat components.]''')
