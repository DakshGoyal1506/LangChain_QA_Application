import streamlit as st 
import pandas as pd 

# Text Input

name = st.text_input('Your name: ')
if name:
    st.write(f'Hello {name}!')

# NUMBER INPUT
x = st.number_input('Enter a number', min_value=1, max_value=99, step=1)
st.write(f'The current number is {x}')

st.divider()

# Button
clicked = st.button('Click me!')

if clicked:
    st.write(':ghost:' * 3)

st.divider()

# Checkbox

agree = st.checkbox('I agree')
if agree:
    "Great, you agreed!"

checked = st.checkbox('Continue', value=True)
if checked:
    ':+1:' * 5

df = pd.DataFrame({'Name': ['Anne', 'Mario', 'Douglas'],
                   'Age': [30, 25, 40]
                   })

if st.checkbox('Show data'):
    st.write(df)

#Radio

pets = ['cat', 'dog', 'fish', 'turtle']
pet = st.radio('Favourite pet', pets, index=2, key='your_pet')
st.write(f'Your favourite pet: {pet}')
st.write(f'Your favorite pet: {st.session_state.your_pet * 3}')

st.divider()

# SELECT BOXES
cities = ['London', 'Berlin', 'Paris', 'Madrid']
city = st.selectbox('Your city', cities, index=1)
st.write(f'Your city is {city}')

# Slider
x = st.slider('x', value = 15, min_value=12, max_value=78, step=3)
st.write(f'x is {x}')

#File uploader

uploaded_file = st.file_uploader('Upload a file:', type=['txt', 'csv'])
if uploaded_file:
    st.write(uploaded_file)
    if uploaded_file.type == 'text/plain':
        from io import StringIO
        stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
        string_data = stringio.read()
        st.write(string_data)
    
    elif uploaded_file.type == 'text/csv':
        df = pd.read_csv(uploaded_file)
        st.write(df)

# Camera

camera_photo = st.camera_input('Take a photo')
if camera_photo:
    st.image(camera_photo)

st.image('https://static/streamlit.io/examples/owl.jpg')