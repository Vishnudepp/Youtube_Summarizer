import validators
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
import streamlit as st
from langchain_ollama import OllamaLLM

model=OllamaLLM(model="llama3.2")
template = """
    Give a 600 words summary of the video url the user shares using the video transcripts.
    here is the context: {text}

    """
prompt = ChatPromptTemplate.from_template(template)

st.set_page_config(page_title="Youtube Summarizer MelCows Thee")
st.title(" Vishnudeep's Conversational Youtube Summarizer MelCows Thee")
st.write("Copy the Youtube URL and get the summarized report")

with st.form("chat_form"):
    url = st.text_input("Type your URL here:")
    submitted = st.form_submit_button("Submit URL")

if submitted:
    try:
        with st.spinner("Waiting..."):
            if not validators.url(url):
                st.error("The URL provided is not valid. Please enter a proper YouTube URL.")
                st.stop()
            if "youtube.com" or "youtu.be" in url:
                loader=YoutubeLoader.from_youtube_url(url, add_video_info=True)
                data=loader.load()
                if not data:
                    st.error("Failed to load video transcripts. Please try another video.")
                    st.stop()
                chains=load_summarize_chain(model, chain_type="stuff",prompt=prompt )
                summary_yt=chains.invoke(data)
                st.success("Summary generated successfully!")
                st.write(summary_yt["output_text"])

    except Exception as e:
        st.error(f"Exception:{e}")

