import data_loader as dl
import preprocessing as p
import streamlit as st

st.title("Automated Machine Learning Pipeline for Small datasets")
st.write("Upload your file link below: (max File size 5MB)")
file_path = st.text_input("File path")

if st.button("Load File"):
    if file_path:
        loader = dl.load_file(file_path)
        data = loader.Load()
        if isinstance(data, str) and data == "File size exceeds 5MB limit.":
            st.write("File size exceeds 5MB limit.")
        elif isinstance(data, str) and data == "File not found.":
            st.write("File not found. Please check the file path.")
        else:
            try:
                st.write("Data Loaded successfully!")
                st.dataframe(data)
            except Exception as e:
                st.write("Error Loading Data: ", e)
    else:
        st.write("Please enter a valid file path.")