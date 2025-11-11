# Crime_Address_Extraction_and_Classification_Based_on_LLM_Data
Data in this repository is for the paper **Theft Address Extraction and Classification from Chinese Judicial Documents Based on Large Language Model**.

Below is an overview of each file in this repository.

  - `LAD_NER_Advance.py` Using Ollama to call LLMs for address extraction and classification
  - `json_extract.py` To serialize the model’s output into JSON format
  - `Error_Calcu.py` To compute the error of the model’s output
  - `Data` 
    - `Train_Set.json` Training data for fine-tuning the LLM
    - `Test_Set_500_json.csv` Test data for evaluating the model’s performance
