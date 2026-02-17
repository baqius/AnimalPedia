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

    if not os.path.exists(model_path):
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


def display_animal_info(animal_data, show_image=True):
    st.header(f"{animal_data['common_name'].title()}")

    if show_image:
        try:
            import os
            import glob
            image_folder = "animal_images/"
            animal_name = animal_data['common_name'].lower()

            pattern = os.path.join(image_folder, f"{animal_name}*")
            matching_images = glob.glob(pattern)

            if not matching_images:
                animal_name_underscore = animal_name.replace(" ", "_")
                pattern = os.path.join(image_folder, f"{animal_name_underscore}*")
                matching_images = glob.glob(pattern)

            if matching_images:
                st.image(matching_images[0], caption=f"{animal_data['common_name'].title()}", width=300)
        except Exception:
            pass

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Scientific Classification")
        st.info(f"**Scientific Name:**\n\n*{animal_data['scientific_name']}*")

    with col2:
        st.subheader("Habitat")
        st.success(f"**Found in:**\n\n{animal_data['found_in']}")

    st.subheader("💡 Fun Fact")
    st.warning(f"{animal_data['fun_fact']}")


def main():
    st.set_page_config(
        page_title="AnimalPedia",
        page_icon="🦁",
        layout="wide"
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #616161;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🦁 AnimalPedia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover fascinating facts about animals from around the world</p>', unsafe_allow_html=True)

    try:
        animal_data = load_animal_data()
    except FileNotFoundError:
        st.error("⚠️ animals_info.json file not found!")
        return

    st.info(f"📊 **Total Animals in Database:** {len(animal_data)}")
    st.markdown("---")

    col_search, col_upload = st.columns(2)

    with col_search:
        st.subheader("🔤 Search Animal by Name")

        search_query = st.text_input("Enter animal name:", placeholder="e.g., lion, elephant, eagle")
        search_button = st.button("🔍 Search", type="primary")

        if search_button or search_query:
            if search_query:
                result = search_animal_by_name(search_query, animal_data)
                if result:
                    st.success(f"✅ Found: {result['common_name'].title()}")
                    st.markdown("---")
                    display_animal_info(result)
                else:
                    st.error(f"❌ Animal '{search_query}' not found in database.")
                    st.info("💡 Please check your spelling and try again.")
            else:
                st.warning("⚠️ Please enter an animal name to search.")

    with col_upload:
        st.subheader("📸 Upload Animal Image for Recognition")
        st.info("🤖 **Note:** Upload an image of an animal to identify it using AI model.")

        uploaded_file = st.file_uploader(
            "Choose an animal image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of an animal"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            st.markdown("### 🔮 Recognition Results")

            with st.spinner("🔍 Analyzing image..."):
                try:
                    model = load_model()
                    predictions = predict_animal(image, model)

                    predicted_animal = predictions[0]['animal']
                    st.success(f"**Predicted Animal:** {predicted_animal.title()}")

                    st.markdown("### 📊 Top 2 Predictions:")
                    for i, pred in enumerate(predictions, 1):
                        emoji = "🥇" if i == 1 else "🥈"
                        st.write(f"{emoji} **{i}. {pred['animal'].title()}**")

                    st.markdown("---")

                    for i, pred in enumerate(predictions, 1):
                        animal_info = search_animal_by_name(pred['animal'], animal_data)
                        if animal_info:
                            st.markdown("## 🥇 First Prediction" if i == 1 else "## 🥈 Second Prediction")
                            display_animal_info(animal_info)
                            if i == 1:
                                st.markdown("---")
                        else:
                            st.warning(f"ℹ️ No additional information available for {pred['animal']}")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #616161; padding: 1rem;'>
            <p>🌿 Built with Streamlit | Data contains information about 90 animals</p>
        </div>
    """, unsafe_allow_html=True)
st.write(response.status_code)
st.write(response.headers)


main()
