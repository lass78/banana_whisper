import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("medium")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    print ("testing2")

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    
    language = model_inputs.get('language', None)
    if language == None:
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio('input.mp3'))).to(model.device)
  
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)

    print("LANGUAGE: ", language)
    if (language == "da" or language == 'en'): 
        task = 'transcribe' 
    else: 
        task = 'translate'


    options = dict(language=language, beam_size=5, best_of=5, task=task)
    result = model.transcribe("input.mp3", **options)
    os.remove("input.mp3")
    # Return the results as a dictionary
    return result
