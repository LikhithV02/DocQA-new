import streamlit as st
import pandas as pd
from PIL import Image
import os, json
from dotenv import load_dotenv
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
# from langchain_groq import ChatGroq
from groq import Groq
# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
# import streamlit.components.v1 as components
# from pymongo import MongoClient
# from bson.objectid import ObjectId
# from datetime import datetime
# from pymongo.server_api import ServerApi
from donut_inference import *
from classification import *
from non_form_llama_parse import *
from RAG import *
import json
import time
# import nest_asyncio
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# print(GROQ_API_KEY)
# llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
client = Groq(api_key=GROQ_API_KEY)
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
# import asyncio

st.set_page_config(layout="wide")

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "upload"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, How can I help you today?"}]
if "conversation_state" not in st.session_state:
        st.session_state["conversation_state"] = [{"role": "assistant", "content": "Hi, How can I help you today?"}]
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "rag" not in st.session_state:
    st.session_state.rag = None


def display_json_in_column(json_data, col):
    # Create a container in the specified column
    with col:
        form_header = f"Classified as - {json_data.get('classified_Form', 'N/A')}"
        file_header = f"File Name - {json_data.get('file', 'N/A')}"

        # Begin constructing the HTML content with dynamic headers
        html_content = f"""
            <style>
            .json-container {{
                width: 500px;
                height: 500px;
                overflow-y: auto;
                margin: 0 auto;
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 15px;
                padding: 10px;
                margin-bottom: 40px;
            }}
            .json-container h3, .json-container h2 {{
                color: black;
            }}
            </style>
            <div class='json-container'>
                <h2>{form_header}</h2>
                <h3>{file_header}</h3>
            """

        # Check if 'items' key exists, otherwise use the root dictionary
        data_to_display = json_data.get('items', json_data)

        if isinstance(data_to_display, dict):
            # Handle as a dictionary: iterate and display key-value pairs
            html_content += "".join([
                f"<p><strong>{key}:</strong> {(', '.join(value) if isinstance(value, list) else value)}</p>"
                for key, value in data_to_display.items() if key != 'classified_Form' and key != 'file'
            ])
        elif isinstance(data_to_display, str):
            # Handle as a string: convert newlines to <br> tags to maintain formatting
            formatted_text = data_to_display.replace("\n", "<br>")
            html_content += f"<p>{formatted_text}</p>"
        else:
            # Handle other types or when data_to_display is still the entire json_data
            html_content += "".join([
                f"<p><strong>{key}:</strong> {(', '.join(value) if isinstance(value, list) else value)}</p>"
                for key, value in (data_to_display.items() if isinstance(data_to_display, dict) else json_data.items()) if key != 'classified_Form' and key != 'file'
            ])
        
        # Close the HTML div tag
        html_content += "</div>"

        # Render the HTML content in the specified column
        st.markdown(html_content, unsafe_allow_html=True)

def csv_chat_interface(data):
    if st.button("Back to Upload"):
        st.session_state["current_page"] = "upload"
        st.session_state.clear()
        st.rerun()
    st.title("DocQA")
    
    for message in st.session_state.messages:
        image = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=image):
            st.markdown(message["content"])
    
    system_prompt = f'''You are a helpful assistant, you will use the provided context to answer user questions. You are great at reding json data.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.\n
Context:\n
{data}
'''
    print("System Prompt: ", system_prompt)
    if prompt := st.chat_input("User input"):
        st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        conversation_context = st.session_state["conversation_state"]
        conversation_context.append({"role": "user", "content": prompt})
        context = []
         # Add system prompt to context if desired
        context.append({"role": "system", "content": system_prompt})
         # Add conversation context to context
        context.extend(st.session_state["conversation_state"])
        # Use the extracted data directly instead of performing inference again
        # print(context)
        response = client.chat.completions.create(
            messages=context,  # Pass conversation context directly
            model="llama3-70b-8192",
            temperature=0,
            top_p=1,
            stop=None,
            stream=True,
        )
        
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            result = ""
            res_box = st.empty()
            for chunk in response:
                if chunk.choices[0].delta.content:
                    new_content = chunk.choices[0].delta.content
                    result += new_content   # Add a space to separate words
                    res_box.markdown(f'{result}')
        assistant_response = result
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        conversation_context.append({"role": "assistant", "content": assistant_response})
        # update_conversation_in_db(prompt,assistant_response)
        
def rag_chat_interface(rag):
    if st.button("Back to Upload"):
        st.session_state["current_page"] = "upload"
        st.session_state.clear()
        st.rerun()
    st.title("DocQA")
    
    for message in st.session_state.messages:
        image = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=image):
            st.markdown(message["content"])
    if prompt := st.chat_input("User input"):
        st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        res = rag(prompt)
        answer, docs = res["result"], res["source_documents"]
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(str(answer))
        st.session_state.messages.append({"role": "assistant", "content": str(answer)})
        # update_conversation_in_db(prompt, str(answer))

def upload():
    st.title('DocQA')
    st.subheader("These are types of forms used to fine-tune DONUT model")

    # Define the paths to your images
    image_paths = [
        "images/cropped_1099-Div.jpg",
        "images/cropped_1099-Int.jpg",
        "images/cropped_w2.jpg",
        "images/cropped_w3.jpg"
    ]
   
    # Define the captions for your images
    captions = ["1099-Div", "1099-Int", "W2", "W3"]

    # Display the images side-by-side with captions
    cols = st.columns(len(image_paths))
    for col, image_path, caption in zip(cols, image_paths, captions):
        col.image(image_path, caption=caption)
    
    st.markdown('''
# Instructions:
1. **Ensure all uploads are in PDF format**. This ensures compatibility and uniform processing across documents.
2. **Submit forms in portrait orientation only**. Landscape formats are not supported and may result in processing errors.
3. **Forms must have a minimum resolution of 1864x1440**. This is crucial for the clarity and legibility necessary for accurate parsing.
4. **Multiple documents can be uploaded simultaneously**; however, the combined size of these documents should not exceed 10MB.
5. **Donut model parses specific forms**: 1099-Div, 1099-Int, W2, and W3. Non-form documents are also processable.
6. **Upload only Forms at a time or Non forms at a time**: we dont accept both forms and Non forms simultaneoulsy.
            ''')
    st.subheader("Try it out")
    if 'uploaded_files'  not in st.session_state:
        st.session_state['uploaded_files'] = []
    st.session_state['uploaded_files'] = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    print(len(st.session_state['uploaded_files']))
    # print(type(uploaded_files))
    full_string = []
    all_data = []
    class_data = {}
    if  'inference_data' not in st.session_state \
        and 'non_form_inference_data' not in st.session_state \
            and 'processed' not in st.session_state \
                and 'non_form_inference_performed' not in st.session_state:
        # st.session_state["inference_performed"] = False
        st.session_state['inference_data'] = []
        st.session_state['non_form_inference_performed'] = False
        st.session_state['non_form_inference_data'] = []
        st.session_state['processed'] = False

    if st.session_state['uploaded_files'] and st.button('Start Processing'):
        if not st.session_state['processed']:
            st.session_state['processed'] = True
            with st.status("Looking for Files...", expanded=True) as status:
                st.write(f"Inferencing Classification Model..")
                for uploaded_file in st.session_state['uploaded_files']:
                    if uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                                    temp_file.write(uploaded_file.getvalue())
                                    temp_file.flush()
                                    pages = convert_from_path(temp_file.name, 300)
                                    img_classification = pages[0].resize((1024, 1024), Image.LANCZOS)
                                    st.success(f"classifying the File {uploaded_file.name}...", icon="✅")
                                    pred = predict(img_classification)
                                    class_data[uploaded_file.name] = pred
                                    if ('Non_Form' in class_data.values()) and ('1099_Int' in class_data.values() or \
                                                                                '1099_Div' in class_data.values() or \
                                                                                'w_2' in class_data.values() or \
                                                                                'w_3' in class_data.values() ):
                                        st.error('You can only upload only Forms type at a time or Non forms at time',  icon="🚨")
                                        time.sleep(5)
                                        st.session_state.clear()
                                        st.rerun()

            
                for uploaded_file in st.session_state['uploaded_files']:
                    if uploaded_file is not None:
                        st.write(f"Processing file {uploaded_file.name}...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                                temp_file.write(uploaded_file.getvalue())
                                temp_file.flush()
                                pages = convert_from_path(temp_file.name, 300)
                                img = pages[0].resize((1864, 1440), Image.LANCZOS)
                        if pred != "Non_Form":
                            # Check if inference has already been performed for this file
                            # if not st.session_state["inference_performed"]:
                            st.success("Infernecing the Donut Model...", icon='✅')
                            data_dict = inference(img)
                            data_dict['file'] = uploaded_file.name
                            data_dict['classified_Form'] = class_data[uploaded_file.name]
                            all_data.append(data_dict)
                            # st.session_state["inference_performed"] = True  # Set the flag to True to indicate inference has been performed
                            st.session_state['inference_data'] = all_data
                            
                        else:
                            # if not st.session_state['non_form_inference_performed']:
                            st.success("Starting the LLama_parse...", icon='✅')
                            text = extract_text(temp_file.name)
                            string_dict = {}
                            string_dict['items'] = text
                            string_dict['file'] = uploaded_file.name
                            string_dict['classified_Form'] = class_data[uploaded_file.name]
                            full_string.append(string_dict)
                            # st.session_state['non_form_inference_performed'] = True
                            st.session_state['non_form_inference_data'] = full_string        
                status.update(label="Parsing complete!", state="complete", expanded=False)

            result_list = st.session_state['inference_data'] + st.session_state['non_form_inference_data']
            chunks = [result_list[i:i + 3] for i in range(0, len(result_list), 3)]
            # print(chunks)
            # Iterate through each chunk and create a row of columns
            for chunk in chunks:
                columns = st.columns(3)  # Always create 3 columns for consistency
                for i in range(len(chunk)):
                    display_json_in_column(chunk[i], columns[i])
                for j in range(len(chunk), 3):  # Fill unused columns
                    with columns[j]:
                        st.write("")
    col1, col2, col3 = st.columns([4,1,4]) 
    if st.session_state['inference_data']:
        # print(all_data)
        # if len(all_data) != 0:
        #     all_data_string = "\n\n".join(json.dumps(data_dict) for data_dict in all_data)
        # else:
        all_data_string = "\n\n".join(json.dumps(data_dict) for data_dict in st.session_state['inference_data'])
        st.session_state.json_data = all_data_string
        
        with col2:
            if st.button("Start Chatting"):
                st.session_state["current_page"] = "csv_chat_ui"
                st.rerun()
            
    elif st.session_state['non_form_inference_data']:
        # if len(full_string) != 0:
        #     qa = rag("\n\n".join(json.dumps(data_dict) for data_dict in full_string))
        # else:
        if not st.session_state['non_form_inference_performed']:
            with st.spinner('Please wait for RAG-Setup to Complete...'):
                qa = rag("\n\n".join(json.dumps(data_dict) for data_dict in st.session_state['non_form_inference_data']))
                st.session_state['non_form_inference_performed'] = True
                st.session_state.rag = qa
                st.success('Done!')
       
        # col1, col2, col3 = st.columns([4,1,4]) 
        with col2:
            if st.button("Start Chatting"):
                st.session_state["current_page"] = "rag_ui"
                st.rerun()
    

def main():
    # if st.session_state["current_page"] == "login":
    #     showLoginPage()
    if st.session_state["current_page"] == "upload":
        upload()
    elif st.session_state["current_page"] == "csv_chat_ui":
        csv_chat_interface(st.session_state.get('json_data'))
    elif st.session_state["current_page"] == "rag_ui":
        rag_chat_interface(st.session_state.get('rag'))
if __name__ == '__main__':
    main()