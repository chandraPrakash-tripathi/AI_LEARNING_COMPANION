import streamlit as st
import pandas as pd
st.set_page_config(page_title="My App", layout="wide") #page config
st.title("Optimization Visualization App") #bold title
st.write("Visualize optimization algorithms in action!") #normal text
if st.button("Start Visualization"): #button
    st.write("Visualization started...") #text after button click

#sidebar
st.sidebar.title("Settings") #sidebar title
mode = st.sidebar.radio("Mode",["Demo","Custom"]) #sidebar radio buttons
st.write(f"Selected mode: {mode}") #display selected mode

#Layout: columns, tabs, expanders, containers
col1, col2  = st.columns([2,1]) #two columns
with col1:
    st.subheader("Main Visualization Area") #subheader in column 1
    st.write("This area will display the main visualizations.") #text in column 1
with col2:
    st.subheader("Controls") #subheader in column 2
    st.write("This area will have control widgets.") #text in column 2

tabs = st.tabs(["Tab 1", "Tab 2"]) #two tabs
with tabs[0]:
    st.write("Content for Tab 1") #content in tab 1
with tabs[1]:
    st.write("Content for Tab 2") #content in tab 2
with st.expander("More Options"): #expander
    st.write("Additional settings can go here.") #text in expander


# Common widgets (inputs & outputs)
#st.button, st.checkbox, st.radio, st.selectbox, st.multiselect, st.slider,
# st.number_input, st.text_input, 
# st.text_area, st.date_input, st.file_uploader.
name = st.text_input("Name")
age = st.slider("Age", 0, 100, 25)
if st.checkbox("Show greeting"):
    st.write(f"Hello {name}, age {age}!")

#Forms (group inputs, submit once)
with st.form("my_form"):
    city = st.text_input("City")
    rating = st.slider("Rating", 1, 5, 3)
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.success(f"Submitted: {city}, {rating}")


#st.session_state — preserve values between reruns
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1

st.write("Count:", st.session_state.count)

#File upload and reading CSV
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())


#Plotting: Matplotlib, Altair, Plotly
#matplolib
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 200)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)


#altair
import altair as alt
import pandas as pd
df = pd.DataFrame({"x": x, "y": y})
chart = alt.Chart(df).mark_line().encode(x="x", y="y")
st.altair_chart(chart, use_container_width=True)
