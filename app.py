import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import random

# Initialize session state
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
    st.session_state.description = ""

# UI Layout
st.set_page_config(page_title="Text-to-Image Generator", layout="wide")
st.markdown("""
<style>
.main { background-color: #f0f2f6; padding: 20px; }
.sidebar .sidebar-content { background-color: #ffffff; border-radius: 10px; padding: 20px; }
.stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
.stButton>button:hover { background-color: #45a049; }
.image-container { border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: white; }
.description-box { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
h1, h2 { color: #333; font-family: 'Arial', sans-serif; }
.stTextArea textarea { border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Hardcoded sample prompts
sample_prompts = [
    "A mystical forest with glowing mushrooms",
    "A futuristic city at sunset",
    "A serene beach with pastel skies",
    "A steampunk airship in the clouds",
    "A dragon soaring over a medieval castle"
]

# Sidebar
with st.sidebar:
    st.header("Image Generation Settings")
    prompt = st.text_area("Enter your prompt:", placeholder="A futuristic city at sunset", height=100)
    style = st.selectbox("Art Style", ["Realism", "Watercolor", "Cyberpunk", "Anime", "Oil Painting"])
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.1, help="Controls how closely the image follows the prompt")
    image_size = st.selectbox("Image Size", ["256x256", "512x512"], index=0)
    num_steps = st.slider("Inference Steps", 10, 100, 30, 5, help="Higher steps improve quality but take longer")

    st.subheader("Sample Prompts")
    sample_prompt = st.selectbox("Choose a sample", [""] + sample_prompts)
    if sample_prompt and st.button("Use Sample"):
        prompt = sample_prompt

# Main content
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Generated Image")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                # Initialize Stable Diffusion
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    use_auth_token=st.secrets.get("HUGGINGFACE_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))
                )
                pipe = pipe.to("cpu")  # Render free tier uses CPU

                # Modify prompt with style
                styled_prompt = f"{prompt}, in {style.lower()} style"
                width, height = map(int, image_size.split("x"))
                image = pipe(
                    styled_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    width=width,
                    height=height
                ).images[0]

                # Save to session state
                st.session_state.generated_image = image

                # Rule-based description
                st.session_state.description = f"This image showcases a {style.lower()} depiction of {prompt.lower()}. The composition features vibrant colors and intricate details, capturing the essence of the described scene."

            except Exception as e:
                st.error(f"Error generating image: {str(e)}")

    # Display image
    if st.session_state.generated_image:
        st.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True, clamp=True)

with col2:
    st.header("Description")
    if st.session_state.description:
        st.markdown(f"<div class='description-box'>{st.session_state.description}</div>", unsafe_allow_html=True)
    else:
        st.info("Generate an image to see its description.")

# Download button
if st.session_state.generated_image:
    img_bytes = st.session_state.generated_image.convert("RGB")
    st.download_button("Download Image", img_bytes, "generated_image.png", "image/png")