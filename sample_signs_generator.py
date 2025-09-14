import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def create_sample_traffic_signs():
    """Create sample traffic sign images for demo purposes"""
    
    # Create Stop sign
    stop_sign = Image.new('RGB', (100, 100), 'red')
    draw = ImageDraw.Draw(stop_sign)
    # Draw octagon shape (simplified as circle for demo)
    draw.ellipse([10, 10, 90, 90], fill='red', outline='white', width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((35, 40), "STOP", fill='white', font=font)
    
    # Create Speed Limit sign
    speed_sign = Image.new('RGB', (100, 100), 'white')
    draw = ImageDraw.Draw(speed_sign)
    draw.ellipse([5, 5, 95, 95], fill='white', outline='red', width=5)
    draw.text((25, 35), "50", fill='black', font=font)
    draw.text((25, 55), "km/h", fill='black', font=font)
    
    # Create Warning sign  
    warning_sign = Image.new('RGB', (100, 100), 'yellow')
    draw = ImageDraw.Draw(warning_sign)
    # Draw triangle
    draw.polygon([(50, 10), (10, 90), (90, 90)], fill='yellow', outline='red')
    draw.text((45, 50), "!", fill='red', font=font)
    
    return {
        "Stop Sign": stop_sign,
        "Speed Limit (50km/h)": speed_sign, 
        "Warning Sign": warning_sign
    }

def get_sample_image_bytes(image):
    """Convert PIL image to bytes for download"""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

# Add this to your main app file or use as separate demo
if __name__ == "__main__":
    st.title("Sample Traffic Signs Generator")
    
    sample_signs = create_sample_traffic_signs()
    
    cols = st.columns(len(sample_signs))
    
    for i, (name, image) in enumerate(sample_signs.items()):
        with cols[i]:
            st.image(image, caption=name)
            st.download_button(
                label=f"Download {name}",
                data=get_sample_image_bytes(image),
                file_name=f"{name.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
