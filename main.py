import streamlit as st
from streamlit_chat import message
from url import url,run_llm


st.title("URL Question Answering App")
url_link = st.text_input("Given your URL")
if st.button('submit'):
    if url_link is not None:
        url(url_link)
        st.success("URL loaded!")

question = st.text_input("Ask a question about the URL:")
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if question:
        st.spinner("loading your answer...")
        answer= run_llm(question,chat_history=st.session_state["chat_history"])
        formatted_response = (
             f"{answer['answer']}"
         )
        st.session_state["user_prompt_history"].append(question)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", question))
        st.session_state["chat_history"].append(("ai", answer["answer"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)