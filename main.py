import streamlit as st
import os
import base64
from io import BytesIO
from smolagents import CodeAgent, HfApiModel, Tool, tool
from PIL import Image
import tempfile
from gradio_client import Client, handle_file

st.set_page_config(page_title="BillEase Assistant", page_icon="💬", layout="centered")
st.title("BillEase Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

@tool
def analyze_image(image_path: str) -> str:
    """
    Analyze an image using Llama 3.2 Vision model and return a summary or error message.
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        A string containing either a description of the image or an error message
    """
    try:
        image_summarizer = Tool.from_space(
            space_id="mohsinmubaraksk/Llama-3.2-11-b-vision",
            name="image-summarizer",
            description="Generate a concise list of key details visible in an image",
            api_name="/predict"
        )
        
        result = image_summarizer(
            image=handle_file(image_path),
            question="""
            CRITICAL INSTRUCTION: If the image appears blurry, low quality, low resolution, or has ANY difficulty reading text or identifying details clearly, your ENTIRE response must be EXACTLY these 5 words only:

            "Please upload a clearer image"

            Do NOT attempt to describe blurry images. Do NOT list any details if clarity is questionable.

            ONLY if the image is perfectly clear and all text is easily readable:
            • List key details in bullet points
            • No introductions or conclusions
            • No explanations of what you see
            • Just facts in a concise list
            """
        )
        
        return result
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@st.cache_resource
def load_model():
    model = HfApiModel()
    return model

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return encoded_string
    except Exception as e:
        return f"[Image could not be encoded: {str(e)}]"

@st.cache_resource
def create_agent():
    model = load_model()
    return CodeAgent(tools=[analyze_image], model=model, add_base_tools=True)

try:
    agent = create_agent()
except Exception as e:
    st.error(f"Error creating agent: {str(e)}")
    agent = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"] is not None:
            st.image(message["image"])

with st.sidebar:
    st.title("About")
    st.markdown("""
    This assistant can answer queries about BillEase, a fintech company in the Philippines.
    
    You can also upload images for analysis using Llama 3.2 Vision.
    """)

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
image_path = None
image_analyzed = False

if uploaded_file is not None and not image_analyzed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        image_path = tmp_file.name
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if image_path and agent is not None:
        with st.spinner("Analyzing image..."):
            try:
                st.session_state.messages.append({"role": "user", "content": "Image uploaded", "image": uploaded_file})
                
                image_analysis = analyze_image(image_path)
                
                with st.chat_message("assistant"):
                    if image_analysis == "Please upload a clearer image":
                        st.markdown(image_analysis)
                    else:
                        st.markdown("I've analyzed the uploaded image:\n\n" + image_analysis)
                
                st.session_state.messages.append({"role": "assistant", "content": image_analysis, "image": None})
                
                image_analyzed = True
                st.session_state.image_analyzed = True
                
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")

if prompt := st.chat_input("Ask a question about BillEase or discuss the uploaded image..."):
    user_message = {"role": "user", "content": prompt}
    if uploaded_file is not None:
        user_message["image"] = uploaded_file
    
    st.session_state.messages.append(user_message)
    

    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file is not None:
            st.image(uploaded_file)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if agent is None:
                response = "Sorry, I'm having trouble initializing. Please try again later."
            else:
                try:
                    if image_path:
                        if 'image_analysis' in locals() and image_analysis:
                            context = f"The user has uploaded an image. Image analysis: {image_analysis}. User query: {prompt}"
                        else:
                            image_analysis = analyze_image(image_path)
                            context = f"The user has uploaded an image. Image analysis: {image_analysis}. User query: {prompt}"
                    else:
                        context = prompt
                        
                    response = agent.run(
                        f"""
                        You are a helpful assistant who answers queries about Billease, a fintech company in the Philippines.
                        
                        Context information:
                        - BillEase is a buy now, pay later app in the Philippines
                        - Users can shop online, pay bills, and get cash loans
                        - More info at: https://billease.ph/faq/
                        
                        User query: {context}
                        """
                    )
                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}"
            
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
                image_path = None
                
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response, "image": None})

if "image_analyzed" not in st.session_state:
    st.session_state.image_analyzed = False
else:
    image_analyzed = st.session_state.image_analyzed