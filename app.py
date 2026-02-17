import streamlit as st
import json
from PIL import Image
import torch
from torchvision import transforms, models


@st.cache_resource
def load_model():
    import requests
    import os

    model_path = "efficient_best_mine.pth"

    if os.path.exists(model_path):
        os.remove(model_path)

    with st.spinner("Downloading model... please wait"):
        url = "https://media.githubusercontent.com/media/baqius/AnimalPedia/main/efficient_best_mine.pth"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    model = models.efficientnet_b0(weights=None)
    num_classes = 90
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_class_names_from_json():
    with open("animals_info.json", 'r') as f:
        data = json.load(f)
    return sorted([animal['common_name'] for animal in data])


def predict_animal(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top2_prob, top2_idx = torch.topk(probabilities, 2)

    class_names = load_class_names_from_json()

    return [
        {
            'animal': class_names[top2_idx[0][i].item()],
            'confidence': top2_prob[0][i].item()
        }
        for i in range(2)
    ]


@st.cache_data
def load_animal_data():
    with open("animals_info.json", 'r') as f:
        return json.load(f)


def search_animal_by_name(animal_name, data):
    animal_name_lower = animal_name.lower().strip()
    for animal in data:
        if animal['common_name'].lower() == animal_name_lower:
            return animal
    return None


def display_animal_info_compact(animal_data):
    import os, glob

    col_img, col_facts = st.columns([1, 3])

    with col_img:
        try:
            image_folder = "animal_images/"
            animal_name = animal_data['common_name'].lower()
            pattern = os.path.join(image_folder, f"{animal_name}*")
            matching_images = glob.glob(pattern)
            if not matching_images:
                pattern = os.path.join(image_folder, f"{animal_name.replace(' ', '_')}*")
                matching_images = glob.glob(pattern)
            if matching_images:
                st.image(matching_images[0], width=150)
        except Exception:
            pass

    with col_facts:
        st.markdown(f"**🐾 {animal_data['common_name'].title()}** — *{animal_data['scientific_name']}*")
        st.markdown(f"📍 {animal_data['found_in']}")
        st.markdown(f"💡 {animal_data['fun_fact']}")


def main():
    st.set_page_config(
        page_title="AnimalPedia",
        page_icon="🦁",
        layout="wide"
    )

    st.markdown("""
        <style>
        .main-header { font-size: 2rem; color: #2E7D32; text-align: center; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1rem; color: #616161; text-align: center; margin-bottom: 0.5rem; }
        div[data-testid="stExpander"] { border: none; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🦁 AnimalPedia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover fascinating facts about animals from around the world</p>', unsafe_allow_html=True)

    try:
        animal_data = load_animal_data()
    except FileNotFoundError:
        st.error("⚠️ animals_info.json file not found!")
        return

    st.caption(f"📊 {len(animal_data)} animals in database")
    st.markdown("---")

    col_search, col_upload = st.columns(2)

    with col_search:
        st.subheader("🔤 Search by Name")
        search_query = st.text_input("Animal name:", placeholder="e.g., lion, elephant, eagle", label_visibility="collapsed")
        search_button = st.button("🔍 Search", type="primary")

        if search_button or search_query:
            if search_query:
                result = search_animal_by_name(search_query, animal_data)
                if result:
                    st.success(f"✅ {result['common_name'].title()}")
                    display_animal_info_compact(result)
                else:
                    st.error(f"❌ '{search_query}' not found. Check spelling and try again.")
            else:
                st.warning("⚠️ Please enter an animal name.")

    with col_upload:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an animal image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            col_pic, col_results = st.columns([1, 1])

            with col_pic:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded", width=200)

            with col_results:
                with st.spinner("🔍 Analyzing..."):
                    try:
                        model = load_model()
                        predictions = predict_animal(image, model)

                        st.markdown("**🔮 Top 2 Predictions:**")
                        for i, pred in enumerate(predictions, 1):
                            emoji = "🥇" if i == 1 else "🥈"
                            st.write(f"{emoji} **{pred['animal'].title()}**")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            st.markdown("---")

            if 'predictions' in dir():
                for i, pred in enumerate(predictions, 1):
                    animal_info = search_animal_by_name(pred['animal'], animal_data)
                    if animal_info:
                        st.markdown(f"{'🥇' if i == 1 else '🥈'} **{pred['animal'].title()}**")
                        display_animal_info_compact(animal_info)
                        if i == 1:
                            st.markdown("---")

    st.markdown("---")
    st.caption("🌿 Built with Streamlit | Data contains information about 90 animals")


main()
