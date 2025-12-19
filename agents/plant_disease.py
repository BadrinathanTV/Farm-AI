import os
import time
import io
import tempfile
from PIL import Image
from huggingface_hub import InferenceClient

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.config import settings

class PlantDiseaseAgent:
    """
    Agent for detecting plant diseases using a specialized computer vision model 
    via Hugging Face InferenceClient, and explaining the results using an LLM.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        
        # Initialize HF Client
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.repo_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        
        # Initialize the InferenceClient with the token
        if self.hf_token:
            self.client = InferenceClient(token=self.hf_token)
        else:
            self.client = None

        # Prompt for explaining the diagnosis
        self.explanation_prompt = ChatPromptTemplate.from_template(
            """You are an expert plant pathologist.
A specialized computer vision model has analyzed an image of a {crop_name} leaf and detected: **{disease_label}** with a confidence of {score:.1%}.

**Your Task:**
Provide a helpful, actionable diagnosis for the farmer.
1.  **Diagnosis**: Clearly state what the disease is (replace technical names like "Tomato___Early_blight" with natural language like "Early Blight in Tomato").
2.  **Symptoms**: Briefly describe what this disease usually looks like to confirm it matches what the farmer sees.
3.  **Treatment**: Provide organic and chemical treatment options.
4.  **Prevention**: Suggest 1-2 key preventive measures for the future.

Keep the response concise, encouraging, and easy to understand for a farmer.
"""
        )
        self.chain = self.explanation_prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---PLANT DISEASE AGENT---")
        
        image_data = state.get("image_data")
        
        if not image_data:
            return {"messages": [AIMessage(content="Please upload a clear photo of the affected plant leaf so I can examine it.")]}

        if not self.client:
             return {"messages": [AIMessage(content="I'm ready to help, but I need a **Hugging Face API Token** to access my vision model. Please add `HUGGINGFACEHUB_API_TOKEN` to your `.env` file.")]}

        try:
            print(f"Calling HF InferenceClient for model: {self.repo_id}")
            
            # Use the InferenceClient to classify the image
            # Save to temporary file to ensure robust handling by HF Client
            image = Image.open(io.BytesIO(image_data))
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp, format='JPEG')
                tmp_path = tmp.name
            
            try:
                predictions = self.client.image_classification(
                    image=tmp_path, 
                    model=self.repo_id
                )
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            if not predictions:
                raise ValueError("No predictions returned from the model.")

            # Get the top prediction
            top_prediction = predictions[0]
            label = top_prediction.label
            score = top_prediction.score
            
            print(f"Disease Detected: {label} ({score})")
            
            # Heuristic to guess crop name
            crop_name = "plant"
            if "___" in label:
                crop_name = label.split("___")[0].replace("_", " ")

            # Generate explanation
            explanation = self.chain.invoke({
                "crop_name": crop_name,
                "disease_label": label,
                "score": score
            })
            
            return {
                "messages": [AIMessage(content=explanation.content)],
                "detected_activity": f"Diagnosed {label} on {crop_name}"
            }

        except Exception as e:
            print(f"Error in PlantDiseaseAgent: {type(e).__name__} - {e}")
            return {"messages": [AIMessage(content=f"Error analyzing image: {str(e)}. Please try again.")]}
