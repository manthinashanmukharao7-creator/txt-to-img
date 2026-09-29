import io
import requests
import streamlit as st
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CLIP Text-Based Image Search",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #888;
    margin-bottom: 30px;
}

.search-box {
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🔎 CLIP Text-Based Image Search</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Search online images using natural-language descriptions'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD CLIP
# =========================================================

@st.cache_resource
def load_clip():

    model_name = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(
        model_name
    )

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    model.to(device)

    model.eval()

    return model, processor


with st.spinner("🧠 Loading CLIP model..."):

    model, processor = load_clip()


# =========================================================
# PEXELS SEARCH
# =========================================================

def search_pexels(query, per_page=15):

    api_key = st.secrets["PEXELS_API_KEY"]

    url = "https://api.pexels.com/v1/search"

    headers = {
        "Authorization": api_key
    }

    params = {
        "query": query,
        "per_page": per_page
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=20
        )

        if response.status_code != 200:

            st.error(
                f"Pexels API Error: {response.status_code}"
            )

            st.code(response.text)

            return []

        data = response.json()

        return data.get("photos", [])

    except Exception as e:

        st.error(
            f"Connection error: {str(e)}"
        )

        return []


# =========================================================
# DOWNLOAD IMAGE
# =========================================================

def download_image(url):

    try:

        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        image = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

        return image

    except Exception:

        return None


# =========================================================
# TEXT EMBEDDING
# =========================================================

@torch.no_grad()
def get_text_embedding(text):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    features = model.get_text_features(
        **inputs
    )

    features = features / features.norm(
        dim=-1,
        keepdim=True
    )

    return features


# =========================================================
# IMAGE EMBEDDING
# =========================================================

@torch.no_grad()
def get_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    features = model.get_image_features(
        pixel_values=pixel_values
    )

    features = features / features.norm(
        dim=-1,
        keepdim=True
    )

    return features


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Search Settings")

number_of_images = st.sidebar.slider(
    "Images to retrieve",
    min_value=5,
    max_value=20,
    value=10
)

number_of_results = st.sidebar.slider(
    "Results to display",
    min_value=1,
    max_value=10,
    value=6
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"🖥️ Device: **{device.upper()}**"
)

st.sidebar.write(
    "🤖 Model: **CLIP ViT-B/32**"
)

st.sidebar.write(
    "🌐 Source: **Pexels**"
)


# =========================================================
# SEARCH
# =========================================================

st.subheader("🔍 Search Images")

query = st.text_input(
    "Enter a natural-language description",
    placeholder="Example: a dog playing on a beach"
)

search_button = st.button(
    "🚀 Search Images",
    use_container_width=True
)


# =========================================================
# MAIN SEARCH
# =========================================================

if search_button:

    if not query.strip():

        st.warning(
            "⚠️ Please enter a search description."
        )

        st.stop()


    # -----------------------------------------------------
    # SEARCH ONLINE
    # -----------------------------------------------------

    with st.spinner(
        "🌐 Searching Pexels for images..."
    ):

        photos = search_pexels(
            query,
            number_of_images
        )


    if not photos:

        st.warning(
            "No images were found."
        )

        st.stop()


    st.success(
        f"✅ Found {len(photos)} online images."
    )


    # -----------------------------------------------------
    # CREATE TEXT EMBEDDING
    # -----------------------------------------------------

    with st.spinner(
        "🧠 Processing text with CLIP..."
    ):

        text_embedding = get_text_embedding(
            query
        )


    # -----------------------------------------------------
    # PROCESS IMAGES
    # -----------------------------------------------------

    results = []

    progress = st.progress(0)

    total = len(photos)


    for i, photo in enumerate(photos):

        image_url = photo.get(
            "src",
            {}
        ).get(
            "large"
        )


        if not image_url:

            continue


        image = download_image(
            image_url
        )


        if image is None:

            progress.progress(
                (i + 1) / total
            )

            continue


        try:

            image_embedding = get_image_embedding(
                image
            )


            # -------------------------------------------------
            # COSINE SIMILARITY
            # -------------------------------------------------

            similarity = torch.matmul(
                text_embedding,
                image_embedding.T
            ).item()


            results.append({
                "image": image,
                "score": similarity,
                "photo_url": photo.get(
                    "url",
                    "#"
                ),
                "photographer": photo.get(
                    "photographer",
                    "Unknown"
                ),
                "photographer_url": photo.get(
                    "photographer_url",
                    "#"
                )
            })


        except Exception:

            pass


        progress.progress(
            (i + 1) / total
        )


    progress.empty()


    # -----------------------------------------------------
    # SORT BY CLIP SCORE
    # -----------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    results = results[
        :number_of_results
    ]


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    if not results:

        st.error(
            "Unable to process the retrieved images."
        )

        st.stop()


    st.subheader(
        f"🎯 Top {len(results)} Results"
    )

    st.markdown(
        f'**Search query:** `{query}`'
    )


    # -----------------------------------------------------
    # IMAGE GRID
    # -----------------------------------------------------

    columns = st.columns(3)


    for index, result in enumerate(results):

        with columns[index % 3]:

            st.image(
                result["image"],
                use_container_width=True
            )

            st.markdown(
                f"### #{index + 1}"
            )

            st.metric(
                "CLIP Similarity",
                f"{result['score']:.4f}"
            )

            st.caption(
                f"📷 {result['photographer']}"
            )

            st.markdown(
                f"[🌐 View Pexels Photo]"
                f"({result['photo_url']})"
            )


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander("🧠 How does this project work?"):

    st.markdown("""
## Text-Based Image Search Using CLIP

### Step 1 — Natural Language Query

The user enters a description such as:

**"a dog playing on a beach"**

### Step 2 — Online Image Retrieval

The application uses the **Pexels API** to retrieve
online images matching the query.

### Step 3 — Text Embedding

CLIP converts the user's text into a numerical
representation called a **text embedding**.

### Step 4 — Image Embedding

CLIP converts every retrieved image into an
**image embedding**.

### Step 5 — Similarity Calculation

The application calculates the similarity between
the text and image embeddings.

### Step 6 — Ranking

Images are sorted from highest to lowest semantic
similarity.

### Step 7 — Dashboard

The highest-ranked images are displayed in the
Streamlit dashboard.

---

### Architecture

User Text
↓
Pexels Image Search
↓
Online Images
↓
CLIP Image Encoder
↓
Image Embeddings

User Text
↓
CLIP Text Encoder
↓
Text Embedding

Text Embedding + Image Embeddings
↓
Cosine Similarity
↓
Ranking
↓
Top Images
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built with Streamlit + Pexels API + OpenAI CLIP"
)