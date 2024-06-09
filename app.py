import streamlit as st
from io import BytesIO
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
import torch
import torchvision

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro")

prompt = PromptTemplate(
    input_variables=["question", "elements"],
    template="""You are a helpful assistant that can answer question related to an image. You have the ability to see the image and answer questions about it. 
    I will give you a question and element about the image and you will answer the question.
        \n\n
        #Question: {question}
        #Elements: {elements}
        \n\n
        Your structured response:""",
    )

def convert_png_to_jpg(image):
    rgb_image = image.convert('RGB')
    byte_arr = BytesIO()
    rgb_image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return Image.open(byte_arr)

def vilt(image, query):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    encoding = processor(image, query, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    sol = model.config.id2label[idx]
    return sol

def blip(image, query):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    # unconditional image captioning
    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)
    sol = processor.decode(out[0], skip_special_tokens=True)
    return sol

def GIT(image, query):
    processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

    # file_path = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
    # image = Image.open(file_path).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    question = query

    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)

    generated_ids_1 = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids_1, skip_special_tokens=True)[0]

    return response[0] + " " + generated_caption

@st.cache_data(show_spinner="Processing image...")
def generate_table(uploaded_file):
    image = Image.open(uploaded_file)
    print("graph start")
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    print("graph start 1")
    inputs = processor(images=image, text="Generate underlying data table of the figure below and give the text as well:", return_tensors="pt")
    predictions = model.generate(**inputs, max_new_tokens=512)
    print("end")
    table = processor.decode(predictions[0], skip_special_tokens=True)
    print(table)
    return table

def process_query(image, query):
    blip_sol = blip(image, query)
    vilt_sol = vilt(image, query)
    GIT_sol = GIT(image, query)
    llm_sol = blip_sol + " " + vilt_sol + " " + GIT_sol
    print(llm_sol)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, elements=llm_sol)
    return response

def process_query_graph(data_table, query):
    prompt = PromptTemplate(
    input_variables=["question", "elements"],
    template="""You are a helpful assistant capable of answering questions related to graph images.
     You possess the ability to view the graph image and respond to inquiries about it. 
     I will provide you with a question and the associated data table of the graph, and you will answer the question
        \n\n
        #Question: {question}
        #Elements: {elements}
        \n\n
        Your structured response:""",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, elements=data_table)
    return response

def chart_with_Image():
    st.header("Chat with Image", divider='rainbow')
    uploaded_file = st.file_uploader('Upload your IMAGE', type=['png', 'jpeg', 'jpg'], key="imageUploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # ViLT model only supports JPG images
        if image.format == 'PNG':
            image = convert_png_to_jpg(image)

        st.image(image, caption='Uploaded Image.', width=300)

        cancel_button = st.button('Cancel')
        query = st.text_input('Ask a question to the IMAGE')

        if query:
            with st.spinner('Processing...'):
                answer = process_query(image, query)
                st.write(answer)

        if cancel_button:
            st.stop()

def chat_with_graph():
    st.header("Chat with Graph", divider='rainbow')
    uploaded_file = st.file_uploader('Upload your GRAPH', type=['png', 'jpeg', 'jpg'], key="graphUploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.format == 'PNG':
            image = convert_png_to_jpg(image)

        # data_table = generate_table(uploaded_file)

        st.image(image, caption='Uploaded Image.')
        data_table = generate_table(uploaded_file)  
        cancel_button = st.button('Cancel')
        query = st.text_input('Ask a question to the IMAGE')
        if query:
            with st.spinner('Processing...'):
                answer = process_query_graph(data_table, query)
                st.write(answer)

        if cancel_button:
            st.stop()

st.title("VisionQuery")
option = st.selectbox(
   "Who would you like to chart with?",
   ("Image", "Graph"),
   index=None,
   placeholder="Select contact method...",
)

st.write('You selected:', option)
if option == "Image":
    chart_with_Image()
elif option == "Graph":
    chat_with_graph()
