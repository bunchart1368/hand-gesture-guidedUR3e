import streamlit as st

# streamlit run webpage.py

# Streamlit frontend
st.set_page_config(
    page_title="Tic-Tac-Toe",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Tic-Tac-Toe 🎮")

with st.sidebar:
    st.title(f"Welcome")
    st.markdown('''
    This is your Tic-Tac-Toe game.
    You can play with ROBOT!
    
    🎥 💌 🤖 ✨
    ''')     
    st.title('''To play:''')
    st.markdown('''          
    point your index finder to the position you want to play
    The robot will draw the symbol for you ;)
    ''')


empty_path = 'image/empty.png'
x_path = 'image/x.png'
o_path = 'image/o.png'
board = [" " for _ in range(9)]
