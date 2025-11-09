# 📖 MangaFlow: Multimodal Manga Translation Engine

<!-- Place a Demo GIF or Final Result Screenshot here once the project is functional. -->

A complete End-to-End Pipeline for automatic, high-quality manga (comic) translation. This project aims to demonstrate advanced AI capabilities by accurately processing images, localizing text bubbles, translating content, and seamlessly reconstructing the final image.

<!-- 🏆 Highlight: Placeholder for any future achievement (e.g., "Featured on...") -->

💡 Core Pipeline & Features

MangaFlow is designed around a multi-stage flow to handle the complexities of comic translation:

- Text Localization: Detect and accurately map the bounding boxes of text bubbles within the comic panel.

- Inpainting & Removal: Erase the original Japanese text from the image using advanced inpainting techniques.

- OCR: Extract the original text content from the detected bubbles.

- Translation: Utilize a state-of-the-art model for accurate and context-aware translation.

- Reconstruction: Render the translated text back into the cleaned bubble area, matching the original comic aesthetic (font, size, alignment).


⚙️ Installation & Setup

This section details how to clone the repository and set up the necessary environment.

1. Clone the Repository

```Bash
git clone https://github.com/akaiuun12/MangaFlow
cd MangaFlow
```

2. Create Virtual Environment

```Bash
python -m venv .venv
source .venv/bin/activate
```

3. Install Dependencies

```Bash
pip install -r requirements.txt
```

<!-- 
4. Model Weights Download (Crucial Step!)

[PLACEHOLDER: Add specific instructions or a Python script for downloading the models here.]

    Note: Due to the large file size and potential licensing restrictions, model weights (.pt, .pth files) are NOT included in this repository. Please follow the instructions to download the required weights (e.g., Detection model) into the designated /models folder. -->

🚀 Usage

<!-- [PLACEHOLDER: Add detailed commands on how to run the pipeline, the FastAPI server, or the Gradio interface.] -->

```Bash
# Example: To run the Gradio demo
python src/service/gradio_app.py
```

🤝 License & Acknowledgements

This project is released under the MIT License.

<!-- 
We extend our sincere thanks to the creators of the following essential tools and datasets:

    [deepghs/manga109_yolo] - For providing a well-trained detection model.

        ATTENTION: The specific license for this model's weights is currently unclear. This project is intended for non-commercial, educational, and research purposes only, and all users must respect the original author's terms of use.

    Manga109s Dataset - For the original data source used in training the core detection model.

    LaMA Cleaner (Apache 2.0)

    Hugging Face (Apache 2.0) -->

<!-- 
📝 TODO List

[PLACEHOLDER: This helps show you have a plan for future commits.]

    [ ] Complete the src/modules/detection implementation using Detectron2.

    [ ] Integrate an inpainting model for text removal.

    [ ] Implement the FastAPI service layer.

    [ ] Build the final Gradio demo interface. -->

👤 Author

[Akai Red / akaiuun12]
