import google.generativeai as genai
import PIL.Image
import os
import json
from datetime import datetime
from io import BytesIO
import fitz 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



# Configure the API key securely from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI app initialization
app = FastAPI()

def validate_pdf(filename: str, content: bytes) -> bool:
    """Validate if the file is a PDF"""
    if not filename.lower().endswith('.pdf'):
        return False
    
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        doc.close()
        return True
    except fitz.FitzError:
        return False

def convert_pdf_to_image(pdf_path: str) -> BytesIO:
    """Convert first page of PDF to image"""
    try:
        pdf_document = fitz.open(pdf_path)
        if pdf_document.page_count < 1:
            raise HTTPException(status_code=400, detail="PDF has no pages.")
        
        page = pdf_document.load_page(0)  # Load first page
        pix = page.get_pixmap()
        img_data = BytesIO(pix.tobytes("png"))
        pdf_document.close()
        return img_data
    except Exception as e:
        print(f"Error converting PDF to image: {str(e)}")
        return None

def extract_aadhaar_data(image_source: BytesIO) -> dict:
    """
    Extract Aadhaar card data using Google's Generative AI
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        img = PIL.Image.open(image_source)

        # Define prompt for Aadhaar extraction
        prompt = """
        Analyze this Aadhaar card image and extract the following details:
        - Full name
        - Date of birth
        - Gender
        - Aadhaar number
        - Address
        -S/O, D/O
        Return the information in this JSON format:
        {
            "name": "",
            "date_of_birth": "",
            "date_of_birth_year": "",
            "gender": "",
            "aadhaar_number": "",
            "address": "",
            "Parent": "",
            "confidence": 0-100
        }
        """

        response = model.generate_content([prompt, img])
        json_str = response.text.strip()
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to decode JSON response from AI.")
            return None
    
    except Exception as e:
        print(f"Error extracting Aadhaar data: {str(e)}")
        return None

@app.post("/extract-aadhaar-data/")
async def extract_aadhaar(aadhaar: UploadFile = File(...)):
    """
    Extract data from Aadhaar card and return JSON format
    """
    try:
        # Read Aadhaar content
        aadhaar_content = await aadhaar.read()

        # Validate PDF
        if not validate_pdf(aadhaar.filename, aadhaar_content):
            raise HTTPException(
                status_code=400,
                detail=f"File {aadhaar.filename} must be a valid PDF"
            )

        # Save PDF temporarily
        temp_filename = f"temp_aadhaar_{datetime.now().timestamp()}.pdf"
        with open(temp_filename, 'wb') as f:
            f.write(aadhaar_content)

        # Convert to image
        img_data = convert_pdf_to_image(temp_filename)
        if not img_data:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process {aadhaar.filename}"
            )

        # Extract Aadhaar data
        aadhaar_data = extract_aadhaar_data(img_data)
        if not aadhaar_data:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract data from {aadhaar.filename}"
            )

        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error deleting temporary file: {str(e)}")

        return JSONResponse(status_code=200, content=aadhaar_data)
    
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
