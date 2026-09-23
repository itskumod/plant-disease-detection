import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Dense, InputLayer


st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background: radial-gradient(
        ellipse at top left,
        #0a1f0d 0%,
        #0d1a10 40%,
        #060f07 100%
    );
    min-height: 100vh;
}

.hero-title {
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #56ab2f,
        #a8e063,
        #56ab2f
    );
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    margin-bottom: 0.2rem;
}

@keyframes shimmer {
    0% {
        background-position: 0%;
    }

    100% {
        background-position: 200%;
    }
}

.hero-subtitle {
    text-align: center;
    color: #4a7a45;
    font-size: 1rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: rgba(86,171,47,0.07);
    border: 1px solid rgba(86,171,47,0.2);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}

.stat-label {
    color: #4a7a45;
    font-size: 0.75rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stat-value {
    color: #a8e063;
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}

.result-card {
    background: linear-gradient(
        135deg,
        rgba(20,45,18,0.9),
        rgba(10,25,10,0.95)
    );
    border: 1px solid rgba(86,171,47,0.4);
    border-radius: 20px;
    padding: 1.8rem;
    margin-top: 1rem;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.5),
        0 0 0 1px rgba(86,171,47,0.1);
}

.result-label {
    color: #4a7a45;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.disease-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #c8f0b0;
    margin-bottom: 0.6rem;
    line-height: 1.3;
}

.badge-healthy {
    display: inline-block;
    background: rgba(80,200,100,0.15);
    border: 1px solid rgba(80,200,100,0.4);
    color: #80e890;
    padding: 3px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

.badge-disease {
    display: inline-block;
    background: rgba(200,90,70,0.15);
    border: 1px solid rgba(200,90,70,0.4);
    color: #e89070;
    padding: 3px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

.badge-warn {
    display: inline-block;
    background: rgba(200,160,50,0.15);
    border: 1px solid rgba(200,160,50,0.4);
    color: #e8c870;
    padding: 3px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 8px;
    margin: 1rem 0 0.3rem;
    overflow: hidden;
}

.conf-fill-green {
    height: 100%;
    background: linear-gradient(
        90deg,
        #2d7a28,
        #a8e063
    );
    border-radius: 50px;
}

.conf-fill-yellow {
    height: 100%;
    background: linear-gradient(
        90deg,
        #7a6a10,
        #e8c840
    );
    border-radius: 50px;
}

.conf-fill-red {
    height: 100%;
    background: linear-gradient(
        90deg,
        #7a2010,
        #e86040
    );
    border-radius: 50px;
}

.pred-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #6a9a65;
    font-size: 0.85rem;
}

.pred-bar {
    height: 3px;
    border-radius: 2px;
    margin-top: 3px;
    background: linear-gradient(
        90deg,
        #3a7a30,
        #a8e063
    );
}

.tip-box {
    background: rgba(200,160,50,0.07);
    border-left: 3px solid #c8a030;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.1rem;
    color: #c8b060;
    font-size: 0.87rem;
    line-height: 1.6;
    margin-top: 1rem;
}

.warn-low {
    background: rgba(200,70,50,0.07);
    border-left: 3px solid #c04030;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1rem;
    color: #e09080;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

.tab-header {
    color: #4a7a45;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
}

</style>
""", unsafe_allow_html=True)


# -------------------- TREATMENT TIPS --------------------

TIPS = {
    "healthy":
        "✅ Your plant looks healthy! Keep up good watering, sunlight, and air circulation.",

    "blight":
        "💧 Avoid overhead watering. Remove infected leaves immediately. Apply copper-based fungicide.",

    "rot":
        "🍂 Improve soil drainage, prune affected areas, and apply appropriate fungicide.",

    "rust":
        "🍃 Remove infected leaves, avoid wetting foliage, and apply sulfur-based spray.",

    "spot":
        "🔬 Apply neem oil or copper spray. Ensure good air circulation around the plant.",

    "mildew":
        "🌬️ Increase air circulation, reduce humidity, apply potassium bicarbonate spray.",

    "mold":
        "🌿 Reduce humidity, improve ventilation, apply chlorothalonil fungicide.",

    "virus":
        "⚠️ No cure available. Remove and destroy infected plants to prevent spread to others.",

    "mites":
        "🔴 Apply insecticidal soap or neem oil spray. Keep plants well-watered.",

    "greening":
        "🍊 No cure exists. Remove infected trees immediately to prevent spread.",

    "scorch":
        "🌡️ Ensure consistent watering and apply mulch around the base to retain moisture.",

    "bacterial":
        "🧫 Remove infected parts, avoid working with wet plants, apply copper-based bactericide.",
}


def get_tip(name):
    nl = name.lower()

    for kw, tip in TIPS.items():
        if kw in nl:
            return tip

    return (
        "🌱 Consult a local agronomist for targeted treatment "
        "advice specific to your region."
    )


# -------------------- PATCH LAYERS --------------------

class PatchedDense(Dense):

    def __init__(self, *args, **kwargs):

        kwargs.pop("quantization_config", None)

        super().__init__(*args, **kwargs)


class PatchedInputLayer(InputLayer):

    def __init__(self, *args, **kwargs):

        kwargs.pop("optional", None)

        if "batch_shape" in kwargs:

            batch_shape = kwargs.pop("batch_shape")

            kwargs["input_shape"] = batch_shape[1:]

        super().__init__(*args, **kwargs)


# -------------------- PATHS --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plant_disease_model.h5"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.npy"
)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():

    try:

        if not os.path.exists(MODEL_PATH):

            st.error(
                f"❌ Model file not found: {MODEL_PATH}"
            )

            return None

        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "Dense": PatchedDense,
                "InputLayer": PatchedInputLayer
            }
        )

    except Exception as e:

        st.error(
            f"❌ Model load error: {e}"
        )

        return None


# -------------------- LOAD CLASS NAMES --------------------

@st.cache_resource
def load_classes():

    try:

        if not os.path.exists(CLASS_NAMES_PATH):

            st.error(
                f"❌ class_names.npy not found: {CLASS_NAMES_PATH}"
            )

            return []

        class_dict = np.load(
            CLASS_NAMES_PATH,
            allow_pickle=True
        ).item()

        return list(class_dict.keys())

    except Exception as e:

        st.error(
            f"❌ class_names.npy error: {e}"
        )

        return []


# -------------------- PREPROCESS --------------------

def preprocess_image(image, h, w):

    image = image.convert("RGB").resize((w, h))

    img = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(
        img,
        axis=0
    )


# -------------------- FORMAT NAME --------------------

def fmt(name):

    return (
        name
        .replace(",_", " ")
        .replace("_", " ")
        .replace("  ", " ")
        .title()
    )


# -------------------- LOAD ASSETS --------------------

model = load_model()

class_names = load_classes()


if model is None or not class_names:

    st.stop()


input_height = model.input_shape[1]

input_width = model.input_shape[2]


# -------------------- HEADER --------------------

st.markdown(
    '<div class="hero-title">🌿 Plant Disease Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">AI-Powered Leaf Analysis</div>',
    unsafe_allow_html=True
)


# -------------------- STATS ROW --------------------

c1, c2, c3 = st.columns(3)


with c1:

    st.markdown(
        '''
        <div class="stat-card">
            <div class="stat-label">🧠 Model</div>
            <div class="stat-value">CNN</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f'''
        <div class="stat-card">
            <div class="stat-label">🌱 Classes</div>
            <div class="stat-value">{len(class_names)}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f'''
        <div class="stat-card">
            <div class="stat-label">📷 Input</div>
            <div class="stat-value">{input_width}×{input_height}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


# -------------------- INPUT TABS --------------------

tab1, tab2 = st.tabs(
    ["📤 Upload Image", "📷 Use Camera"]
)


image = None


with tab1:

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=[
            "jpg",
            "png",
            "jpeg",
            "webp"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )


with tab2:

    camera_photo = st.camera_input(
        "Take a photo of a leaf"
    )

    if camera_photo:

        image = Image.open(
            camera_photo
        )


# -------------------- PREDICTION --------------------

if image is not None:

    col_img, col_res = st.columns(
        [1, 1],
        gap="large"
    )


    with col_img:

        st.image(
            image,
            caption="📸 Uploaded Leaf",
            use_container_width=True
        )


    with col_res:

        with st.spinner(
            "🔬 Analyzing leaf..."
        ):

            processed = preprocess_image(
                image,
                input_height,
                input_width
            )

            prediction = model.predict(
                processed,
                verbose=0
            )[0]


        top_idx = int(
            np.argmax(prediction)
        )

        confidence = (
            float(prediction[top_idx]) * 100
        )

        disease = class_names[top_idx]

        is_healthy = (
            "healthy" in disease.lower()
        )


        # -------------------- CONFIDENCE --------------------

        if confidence >= 80:

            badge = (
                '<span class="badge-healthy">'
                + (
                    "✅ Healthy"
                    if is_healthy
                    else "⚠️ Disease Detected"
                )
                + "</span>"
            )

            fill = "conf-fill-green"


        elif confidence >= 50:

            badge = (
                '<span class="badge-warn">'
                "⚠️ Moderate Confidence"
                "</span>"
            )

            fill = "conf-fill-yellow"


        else:

            badge = (
                '<span class="badge-disease">'
                "❌ Low Confidence"
                "</span>"
            )

            fill = "conf-fill-red"


        # -------------------- RESULT CARD --------------------

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Diagnosis
                </div>

                <div class="disease-name">
                    {fmt(disease)}
                </div>

                {badge}

                <div class="conf-track">

                    <div
                        class="{fill}"
                        style="width:{confidence:.0f}%"
                    ></div>

                </div>

                <div
                    style="
                        color:#4a7a45;
                        font-size:0.82rem;
                    "
                >
                    Confidence:

                    <b style="color:#a8e063">
                        {confidence:.1f}%
                    </b>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------- LOW CONFIDENCE WARNING --------------------

        if confidence < 50:

            st.markdown(
                '''
                <div class="warn-low">
                    Try a clearer, well-lit close-up
                    photo of a single leaf.
                </div>
                ''',
                unsafe_allow_html=True
            )


        # -------------------- TOP PREDICTIONS --------------------

        st.markdown(
            '<div class="tab-header">Top Predictions</div>',
            unsafe_allow_html=True
        )


        top3 = np.argsort(
            prediction
        )[::-1][:3]


        medals = [
            "🥇",
            "🥈",
            "🥉"
        ]


        for i, idx in enumerate(top3):

            name = fmt(
                class_names[idx]
            )

            conf = (
                prediction[idx] * 100
            )


            st.markdown(
                f"""
                <div class="pred-row">

                    <span>
                        {medals[i]} {name}
                    </span>

                    <span
                        style="
                            color:#c8f0b0;
                            font-weight:600
                        "
                    >
                        {conf:.1f}%
                    </span>

                </div>

                <div
                    class="pred-bar"
                    style="width:{int(conf)}%"
                ></div>
                """,
                unsafe_allow_html=True
            )


        # -------------------- TREATMENT TIP --------------------

        st.markdown(
            f"""
            <div class="tip-box">

                <b>💊 Treatment Tip</b>

                <br>

                {get_tip(disease)}

            </div>
            """,
            unsafe_allow_html=True
        )


else:

    st.markdown(
        '''
        <div class="tip-box">
            👆 Upload a leaf image or use your
            camera to get started!
        </div>
        ''',
        unsafe_allow_html=True
    )


# -------------------- FOOTER --------------------

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div
        style="
            text-align:center;
            color:#2a4a25;
            font-size:0.78rem;
            letter-spacing:1px;
        "
    >
        Built with ❤️ using TensorFlow &amp; Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
