import streamlit as st
import json
from PIL import Image
import io
import torch
from torchvision import transforms, models


 
@st.cache_resource
def load_model():
    """Load your trained EfficientNet model"""
    model = models.efficientnet_b0(pretrained=False)
    
    # Modify final layer to match your number of classes
    num_classes = 90  # Update based on your dataset
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Load with strict=False to ignore mismatched keys
    state_dict = torch.load(
        "Fundamental deep learning/Chapter 6 PreTrained ResNet CNN/efficient_best_mine.pth", 
        map_location='cpu'
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model



def load_class_names_from_json():
    """Extract class names from animals_info.json"""
    with open("Fundamental deep learning/Chapter 6 PreTrained ResNet CNN/animals_info.json", 'r') as f:
        data = json.load(f)
    
    # Get sorted list of common names
    class_names = sorted([animal['common_name'] for animal in data])
    return class_names

def predict_animal(image, model):
    """
    Predict animal from uploaded image - returns top 2 predictions
    Args:
        image: PIL Image
        model: Your trained model
    Returns:
        list of dicts with 'animal' and 'confidence' keys
    """
    # Define your transforms (should match training transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    img_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 2 predictions
        top2_prob, top2_idx = torch.topk(probabilities, 2)
    
    # Map predicted class index to animal name
    class_names = load_class_names_from_json()
    
    # Return top 2 predictions
    predictions = []
    for i in range(2):
        predictions.append({
            'animal': class_names[top2_idx[0][i].item()],
            'confidence': top2_prob[0][i].item()
        })
    
    return predictions

# ============================================================================


# Load animal information
@st.cache_data
def load_animal_data():
    """Load animal information from JSON file"""
    with open("Fundamental deep learning/Chapter 6 PreTrained ResNet CNN/animals_info.json", 'r') as f:
        data = json.load(f)
    return data


def search_animal_by_name(animal_name, data):
    """Search for animal information by name"""
    animal_name_lower = animal_name.lower().strip()
    for animal in data:
        if animal['common_name'].lower() == animal_name_lower:
            return animal
    return None


def display_animal_info(animal_data, show_image=True):
    """Display animal information in a nice format"""
    st.header(f"{animal_data['common_name'].title()}")
    
    # Try to find and display animal image from dataset
    if show_image:
        try:
            import os
            import glob
            image_folder = "Fundamental deep learning/Chapter 6 PreTrained ResNet CNN/animal_images/"
            animal_name = animal_data['common_name'].lower()
            
            # Search for images matching the animal name
            # This will match files like "lion.jpg", "lion_1.jpg", "elephant.png", etc.
            pattern = os.path.join(image_folder, f"{animal_name}*")
            matching_images = glob.glob(pattern)
            
            # Also try with spaces replaced by underscores
            if not matching_images:
                animal_name_underscore = animal_name.replace(" ", "_")
                pattern = os.path.join(image_folder, f"{animal_name_underscore}*")
                matching_images = glob.glob(pattern)
            
            if matching_images:
                # Display the first matching image with smaller size
                st.image(matching_images[0], caption=f"{animal_data['common_name'].title()}", width=300)
        except Exception as e:
            pass  # Silently skip if image not found
    
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
    # Page configuration
    st.set_page_config(
        page_title="AnimalPedia",
        page_icon="🦁",
        layout="wide"
    )
    
    # Custom CSS for better styling
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
    
    # Header
    st.markdown('<h1 class="main-header">🦁 AnimalPedia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover fascinating facts about animals from around the world</p>', unsafe_allow_html=True)
    
    # Load data
    try:
        animal_data = load_animal_data()
        animal_names = sorted([animal['common_name'] for animal in animal_data])
    except FileNotFoundError:
        st.error("⚠️ animals_info.json file not found! Please make sure the file is in the same directory as this script.")
        return
    
    # Info box
    st.info(f"📊 **Total Animals in Database:** {len(animal_data)}")
    
    # Main content area - both sections in columns
    st.markdown("---")
    
    col_search, col_upload = st.columns(2)
    
    # ========== COLUMN 1: SEARCH BY NAME ==========
    with col_search:
        st.subheader("🔤 Search Animal by Name")
        
        # Free text search
        search_query = st.text_input("Enter animal name:", placeholder="e.g., lion, elephant, eagle")
        
        search_button = st.button("🔍 Search", type="primary")
        
        if search_button or search_query:
            animal_to_search = search_query
            
            if animal_to_search:
                result = search_animal_by_name(animal_to_search, animal_data)
                
                if result:
                    st.success(f"✅ Found: {result['common_name'].title()}")
                    st.markdown("---")
                    display_animal_info(result)
                else:
                    st.error(f"❌ Animal '{animal_to_search}' not found in database.")
                    st.info("💡 Please check your spelling and try again.")
            else:
                st.warning("⚠️ Please enter an animal name to search.")
    
    # ========== COLUMN 2: UPLOAD IMAGE ==========
    with col_upload:
        st.subheader("📸 Upload Animal Image for Recognition")
        
        st.info("🤖 **Note:** Upload an image of an animal to identify it using AI model.")
        
        uploaded_file = st.file_uploader(
            "Choose an animal image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of an animal"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            st.image(image, caption="Uploaded Image", width=300)
            
            st.markdown("### 🔮 Recognition Results")
            
            with st.spinner("🔍 Analyzing image..."):
                try:
                    model = load_model()
                    predictions = predict_animal(image, model)
                    
                    # Get top prediction
                    top_prediction = predictions[0]
                    predicted_animal = top_prediction['animal']
                    
                    # Display top prediction
                    st.success(f"**Predicted Animal:** {predicted_animal.title()}")
                    
                    # Show top 2 predictions without confidence
                    st.markdown("### 📊 Top 2 Predictions:")
                    for i, pred in enumerate(predictions, 1):
                        emoji = "🥇" if i == 1 else "🥈"
                        st.write(f"{emoji} **{i}. {pred['animal'].title()}**")
                    
                    st.markdown("---")
                    
                    # Display information for both top 2 predictions
                    for i, pred in enumerate(predictions, 1):
                        animal_info = search_animal_by_name(pred['animal'], animal_data)
                        
                        if animal_info:
                            if i == 1:
                                st.markdown("## 🥇 First Prediction")
                            else:
                                st.markdown("## 🥈 Second Prediction")
                            display_animal_info(animal_info)
                            if i == 1:
                                st.markdown("---")
                        else:
                            st.warning(f"ℹ️ No additional information available for {pred['animal']}")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #616161; padding: 1rem;'>
            <p>🌿 Built with Streamlit | Data contains information about 90 animals</p>
        </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main() 

