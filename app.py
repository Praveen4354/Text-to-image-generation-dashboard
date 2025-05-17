import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import random
import signal
import time
import gc

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Image generation took too long!")

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

# Hardcoded sample prompts (simplified for faster processing)
sample_prompts = [
    "A tree",
    "A house",
    "A flower",
    "A cloud",
    "A star"
]

# Sidebar
with st.sidebar:
    st.header("Image Generation Settings")
    prompt = st.text_area("Enter your prompt:", placeholder="A tree", height=100)
    style = st.selectbox("Art Style", ["Realism", "Watercolor", "Cyberpunk", "Anime", "Oil Painting"])
    guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 5.0, 0.1, help="Lower values are faster")
    image_size = st.selectbox("Image Size", ["64x64", "32x32"], index=0)
    num_steps = st.slider("Inference Steps", 5, 10, 5, 1, help="Fewer steps are faster")

    st.subheader("Sample Prompts")
    sample_prompt = st.selectbox("Choose a sample", [""] + sample_prompts)
    if sample_prompt and st.button("Use Sample"):
        prompt = sample_prompt

# Main content
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Generated Image")
    if st.button("Generate Image"):
        with st.spinner("Generating image (may take 1-2 minutes)..."):
            try:
                # Set timeout (4 minutes)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(240)  # 4 minutes

                # Cache pipeline
                @st.cache_resource
                def load_pipeline():
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-1-base",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    # Enable memory-efficient attention
                    pipe.enable_attention_slicing()
                    return pipe

                pipe = load_pipeline()
                pipe = pipe.to("cpu")  # Render free tier uses CPU

                # Modify prompt with style (truncate to 50 chars for speed)
                styled_prompt = f"{prompt[:50]}, in {style.lower()} style"
                width, height = map(int, image_size.split("x"))
                image = pipe(
                    styled_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    width=width,
                    height=height
                ).images[0]

                # Disable timeout
                signal.alarm(0)

                # Save to session state
                st.session_state.generated_image = image

                # Rule-based description
                st.session_state.description = f"A {style.lower()} image of {prompt[:50].lower()} with vivid details."

                # Clean up memory
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except TimeoutError:
                st.error("Image generation timed out. Try 32x32 size or simpler prompt.")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    st.error("Memory limit exceeded. Try 32x32 size and 5 steps.")
                else:
                    st.error(f"Error generating image: {str(e)}")
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                if "authentication" in str(e).lower() or "license" in str(e).lower():
                    st.warning("Ensure you have accepted the license at https://huggingface.co/stabilityai/stable-diffusion-2-1-base.")

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
